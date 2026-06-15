#include "a3i/storage/binary_column_store.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <system_error>

#include <fcntl.h>
#include <liburing.h>
#include <sys/stat.h>
#include <unistd.h>

namespace a3i {

namespace {

// Read an entire raw float64 column file into a heap vector. We verify the
// file size against the expected row count so a corrupt or truncated file
// is caught at load time rather than producing silent garbage reads.
std::vector<double> read_double_file(const std::filesystem::path& path,
                                     std::uint64_t expected_rows) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open column file: " + path.string());
    in.seekg(0, std::ios::end);
    const auto bytes = static_cast<std::uint64_t>(in.tellg());
    if (bytes != expected_rows * sizeof(double)) {
        throw std::runtime_error("column file size mismatch: " + path.string() +
                                 " (expected " + std::to_string(expected_rows * sizeof(double)) +
                                 " bytes, got " + std::to_string(bytes) + ")");
    }
    in.seekg(0, std::ios::beg);
    std::vector<double> out(expected_rows);
    if (expected_rows > 0) {
        in.read(reinterpret_cast<char*>(out.data()),
                static_cast<std::streamsize>(expected_rows * sizeof(double)));
        if (!in) throw std::runtime_error("read failed: " + path.string());
    }
    return out;
}

// pread the exact byte range [off, off+len) or throw; loops over short reads.
void pread_exact(int fd, void* buf, std::size_t len, std::uint64_t off) {
    char* p = static_cast<char*>(buf);
    while (len > 0) {
        const ssize_t got = ::pread(fd, p, len, static_cast<off_t>(off));
        if (got < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "pread");
        }
        if (got == 0) throw std::runtime_error("pread: unexpected end of file");
        p   += got;
        off += static_cast<std::uint64_t>(got);
        len -= static_cast<std::size_t>(got);
    }
}

// Tuning for the batched on-disk gather. Ranges are page-aligned because the
// device and page cache move whole pages anyway; only overlapping or exactly
// adjacent page ranges are merged — reading through a gap of unneeded pages
// trades bytes for fewer requests, which does not pay off when the device's
// random-read throughput does not fall with request count, so over-reading
// only wastes bandwidth. A single range is capped so coalescing cannot grow
// one read without bound. Ranges that fit a ring slot are issued
// asynchronously with up to kRingDepth requests in flight — scattered reads
// are latency-bound, so the device must see a deep queue to reach its
// random-read throughput. Ranges larger than a slot are dense contiguous runs
// where the kernel's readahead already delivers full bandwidth to a plain
// positional read.
constexpr std::uint64_t kPageBytes     = 4096;
constexpr std::uint64_t kMergeGapBytes = 0;
constexpr std::uint64_t kMaxRangeBytes = 8u << 20;
constexpr unsigned      kRingDepth     = 256;
constexpr std::size_t   kSlotBytes     = 64 * 1024;

// Access-path crossover: a scattered gather reads only the pages holding
// wanted rows but at the device's random-read rate, while a sequential scan of
// the rows' [min,max] span reads every page in the span at the (much higher)
// streaming rate and filters in memory. Reading frac of a span's rows touches,
// for rows scattered uniformly over the span's pages,
//   touched_fraction = 1 - (1 - frac)^rows_per_page  ~=  1 - e^(-frac*512)
// distinct pages. The gather is cheaper than the scan while
//   touched_fraction < random_bw / sequential_bw.
// That ratio is a per-device property (saturated random throughput over
// streaming throughput), not a code constant, so it is required from the
// environment variable A3I_RANDOM_SEQ_BW_RATIO -- there is deliberately no
// built-in default, because a wrong device assumption silently picks the wrong
// access path. It is read once and applied as a single value, not a per-call
// probe, so the decision adds no I/O and never perturbs the page cache of the
// columns about to be read. A lower ratio (a device whose random reads are
// relatively slower) lowers the scan crossover and sends more reads to the
// sequential-scan path; 1.0 disables the scan path entirely (always gather).
constexpr std::size_t kRowsPerPage   = kPageBytes / sizeof(double);  // 512
constexpr std::size_t kScanBlockRows = std::size_t{1} << 20;  // 8 MiB/measure

// The crossover ratio, resolved once from A3I_RANDOM_SEQ_BW_RATIO. Throws if it
// is unset or not in (0, 1] -- callers reach this only on an OnDisk store, so
// the cost-model input must be supplied explicitly for the target device.
double random_vs_sequential_bw() {
    static const double ratio = [] {
        const char* env = std::getenv("A3I_RANDOM_SEQ_BW_RATIO");
        if (env == nullptr || *env == '\0') {
            throw std::runtime_error(
                "A3I_RANDOM_SEQ_BW_RATIO is not set: the on-disk access-path "
                "cost model needs the device's random/sequential bandwidth "
                "ratio, a number in (0,1]");
        }
        char* end = nullptr;
        const double v = std::strtod(env, &end);
        if (end == env || v <= 0.0 || v > 1.0) {
            throw std::runtime_error(
                "A3I_RANDOM_SEQ_BW_RATIO must be a number in (0,1]; got: " +
                std::string(env));
        }
        return v;
    }();
    return ratio;
}

// One coalesced read: file bytes [file_off, file_off+len) cover the sorted
// gather positions [first, first+count).
struct ReadRange {
    std::uint64_t file_off = 0;
    std::uint32_t len      = 0;
    std::size_t   first    = 0;
    std::size_t   count    = 0;
};

}  // namespace

// Submission/completion context for batched reads. Owns the kernel ring and
// the per-slot staging buffers; one per store, reused across gathers (setup
// is not free, reads are frequent).
struct BinaryColumnStore::IoRing {
    io_uring          ring{};
    bool              live = false;
    std::vector<char> slots;  // kRingDepth * kSlotBytes staging bytes
    ~IoRing() {
        if (live) io_uring_queue_exit(&ring);
    }
};

BinaryColumnStore::BinaryColumnStore(const std::filesystem::path& manifest_path,
                                     std::vector<MeasureId> selected_measures,
                                     MeasureStorage measure_storage)
    : manifest_(read_manifest(manifest_path)),
      manifest_dir_(manifest_path.parent_path()),
      storage_mode_(measure_storage) {

    // Resolve which measure columns to expose. An empty selection means every
    // measure in manifest order; otherwise the caller's list fixes both the
    // membership and the local id order. Validate against the manifest so an
    // out-of-range id fails at open time rather than on first access.
    if (selected_measures.empty()) {
        exposed_measures_.resize(manifest_.measures.size());
        for (std::size_t i = 0; i < manifest_.measures.size(); ++i) {
            exposed_measures_[i] = static_cast<MeasureId>(i);
        }
    } else {
        for (MeasureId m : selected_measures) {
            if (m >= manifest_.measures.size()) {
                throw std::out_of_range(
                    "BinaryColumnStore: selected measure id out of range: " +
                    std::to_string(m));
            }
        }
        exposed_measures_ = std::move(selected_measures);
    }

    // Dimension columns are loaded eagerly: the adaptive index needs to read,
    // sort, and permute coordinates freely in RAM during grid construction and
    // cracking. The full column fits comfortably for typical dataset sizes; if
    // this ever needs to change the right place is here, not the access path.
    dim_columns_.reserve(manifest_.dimensions.size());
    for (const auto& d : manifest_.dimensions) {
        const auto path = manifest_dir_ / d.file;
        dim_columns_.push_back(read_double_file(path, manifest_.row_count));
    }

    if (storage_mode_ == MeasureStorage::Eager) {
        // In-memory backing: read each exposed column fully resident, exactly
        // as the dimensions are, so a read is a plain array index with no
        // I/O path. Meaningful only where the dataset fits in RAM.
        measure_resident_.reserve(exposed_measures_.size());
        for (std::size_t i = 0; i < exposed_measures_.size(); ++i) {
            const auto& mz = manifest_.measures[exposed_measures_[i]];
            measure_resident_.push_back(
                read_double_file(manifest_dir_ / mz.file, manifest_.row_count));
        }
        return;
    }

    // Out-of-core backing: keep each exposed column on disk behind an open
    // descriptor. All reads are explicit — sequential block reads stream
    // through the kernel's readahead, and scattered gathers are coalesced and
    // issued as batches of asynchronous reads (see gather_ondisk). Only the
    // exposed measures are opened; the rest stay untouched on disk.
    measure_files_.resize(exposed_measures_.size());
    for (std::size_t i = 0; i < exposed_measures_.size(); ++i) {
        const auto& mz = manifest_.measures[exposed_measures_[i]];
        const auto path = manifest_dir_ / mz.file;
        const auto bytes = manifest_.row_count * sizeof(double);

        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("cannot open measure file: " + path.string() + ": " +
                                     std::strerror(errno));
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error("fstat failed: " + path.string());
        }
        if (static_cast<std::uint64_t>(st.st_size) != bytes) {
            ::close(fd);
            throw std::runtime_error("measure file size mismatch: " + path.string());
        }
        measure_files_[i] = {fd, bytes};
    }
}

BinaryColumnStore::~BinaryColumnStore() {
    for (auto& c : measure_files_) {
        if (c.fd >= 0) ::close(c.fd);
    }
}

std::span<const double> BinaryColumnStore::dimension_column(DimensionId d) const {
    if (d >= dim_columns_.size()) {
        throw std::out_of_range("dimension_column: DimensionId out of range");
    }
    return std::span<const double>(dim_columns_[d]);
}

double BinaryColumnStore::measure_value(RowId r, MeasureId m) const {
    if (m >= measure_count()) {
        throw std::out_of_range("measure_value: MeasureId out of range");
    }
    if (r >= manifest_.row_count) {
        throw std::out_of_range("measure_value: RowId out of range");
    }
    if (storage_mode_ == MeasureStorage::Eager) {
        return measure_resident_[m][r];
    }
    double v = 0.0;
    pread_exact(measure_files_[m].fd, &v, sizeof(double),
                static_cast<std::uint64_t>(r) * sizeof(double));
    return v;
}

void BinaryColumnStore::gather(MeasureId m,
                               std::span<const RowId> row_ids,
                               std::span<double> out,
                               bool already_sorted) const {
    if (m >= measure_count()) {
        throw std::out_of_range("gather: MeasureId out of range");
    }
    if (row_ids.size() != out.size()) {
        throw std::invalid_argument("gather: row_ids and out must have equal size");
    }
    for (const RowId r : row_ids) {
        if (r >= manifest_.row_count) {
            throw std::out_of_range("gather: RowId out of range");
        }
    }
    if (row_ids.empty()) return;

    if (storage_mode_ == MeasureStorage::Eager) {
        const std::vector<double>& col = measure_resident_[m];
        for (std::size_t i = 0; i < row_ids.size(); ++i) {
            out[i] = col[row_ids[i]];
        }
        return;
    }
    const MeasureId ms[1]  = {m};
    double* const   outs[1] = {out.data()};
    gather_batch(ms, row_ids, outs, already_sorted);
}

void BinaryColumnStore::gather_all(std::span<const RowId> row_ids,
                                   std::vector<std::vector<double>>& outs,
                                   bool already_sorted,
                                   GatherPathStats* path_stats) const {
    const std::size_t k = measure_count();
    const std::size_t n = row_ids.size();
    for (const RowId r : row_ids) {
        if (r >= manifest_.row_count) {
            throw std::out_of_range("gather_all: RowId out of range");
        }
    }
    outs.resize(k);
    for (auto& o : outs) o.resize(n);
    if (n == 0 || k == 0) return;

    if (storage_mode_ == MeasureStorage::Eager) {
        for (std::size_t m = 0; m < k; ++m) {
            const std::vector<double>& col = measure_resident_[m];
            for (std::size_t i = 0; i < n; ++i) {
                outs[m][i] = col[row_ids[i]];
            }
        }
        return;
    }
    std::vector<MeasureId> ms(k);
    std::vector<double*>   ptrs(k);
    for (std::size_t m = 0; m < k; ++m) {
        ms[m]   = static_cast<MeasureId>(m);
        ptrs[m] = outs[m].data();
    }
    gather_batch(ms, row_ids, ptrs.data(), already_sorted, path_stats);
}

bool BinaryColumnStore::would_scan(RowId lo, RowId hi, std::size_t n) const {
    // Only the on-disk backing has a scan-vs-gather choice; an eager store
    // indexes a resident array, so there is no scan path.
    if (storage_mode_ != MeasureStorage::OnDisk || n == 0) return false;
    // Compare the scattered gather against a sequential scan of the rows'
    // [lo, hi] span. The n wanted rows scatter over the span's pages, touching
    // an expected fraction  1 - (1 - n/span_rows)^rows_per_page  ~=
    // 1 - e^(-frac*rows_per_page)  of them. Gathering reads exactly those pages
    // at the device's random-read rate; scanning reads every page in the span
    // at the streaming rate and filters in memory. The scan wins once the
    // touched fraction reaches the device's random/sequential bandwidth ratio.
    // touched_fraction is the expected value under uniform scatter (exact
    // there); an exact count from the coalesced ranges would be more precise
    // for clustered ids but the scan branch never builds them.
    const std::uint64_t span_rows = static_cast<std::uint64_t>(hi - lo) + 1;
    const double frac = static_cast<double>(n) / static_cast<double>(span_rows);
    const double touched_fraction =
        -std::expm1(-frac * static_cast<double>(kRowsPerPage));  // 1 - e^(-frac*512)
    return touched_fraction >= random_vs_sequential_bw();
}

void BinaryColumnStore::gather_batch(std::span<const MeasureId> ms,
                                     std::span<const RowId> row_ids,
                                     double* const* outs,
                                     bool already_sorted,
                                     GatherPathStats* path_stats) const {
    const std::size_t n = row_ids.size();
    const std::size_t k = ms.size();
    const std::uint64_t col_bytes = manifest_.row_count * sizeof(double);

    // Visit ids in ascending file order so nearby rows coalesce into shared
    // ranges; results are written through the permutation, so caller order is
    // preserved. When the caller guarantees the ids are already ascending
    // (`already_sorted`, e.g. the engine's k-way merge output) the O(n) order
    // check and the sort are both skipped; otherwise the ids are sorted via an
    // index permutation, leaving the caller's array untouched.
    std::vector<std::size_t> order;
    const bool sorted = already_sorted || std::is_sorted(row_ids.begin(), row_ids.end());
    if (!sorted) {
        order.resize(n);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return row_ids[a] < row_ids[b];
        });
    }
    const auto pos_of = [&](std::size_t j) { return sorted ? j : order[j]; };

    // Access-path choice (see would_scan): when the wanted rows are dense
    // enough over their [min,max] span that a sequential scan of the span beats
    // a scattered gather, scan; otherwise gather. The decision is per span and
    // independent of the column count.
    const RowId lo = row_ids[pos_of(0)];
    const RowId hi = row_ids[pos_of(n - 1)];
    const std::uint64_t span_rows = static_cast<std::uint64_t>(hi - lo) + 1;
    if (would_scan(lo, hi, n)) {
        const std::uint64_t bytes = scan_span(ms, row_ids, outs, order, lo, span_rows);
        if (path_stats != nullptr) {
            path_stats->scan_rows  += n;
            path_stats->scan_bytes += bytes;
        }
        return;
    }
    if (path_stats != nullptr) path_stats->gather_rows += n;

    // Coalesce the sorted byte offsets into page-aligned ranges, merging only
    // overlapping or adjacent pages (reading through a gap of unneeded pages
    // wastes bandwidth without reducing device time) and splitting when a
    // dense run would exceed the per-range cap. The columns share one row id
    // set, so one plan serves every requested measure.
    std::vector<ReadRange> ranges;
    for (std::size_t j = 0; j < n; ++j) {
        const std::uint64_t off  = static_cast<std::uint64_t>(row_ids[pos_of(j)]) * sizeof(double);
        const std::uint64_t pbeg = off & ~(kPageBytes - 1);
        const std::uint64_t pend = std::min(col_bytes,
                                            (off + sizeof(double) + kPageBytes - 1) & ~(kPageBytes - 1));
        if (!ranges.empty()) {
            ReadRange& cur = ranges.back();
            const std::uint64_t cur_end = cur.file_off + cur.len;
            if (pbeg <= cur_end + kMergeGapBytes &&
                pend - cur.file_off <= kMaxRangeBytes) {
                if (pend > cur_end) {
                    cur.len = static_cast<std::uint32_t>(pend - cur.file_off);
                }
                ++cur.count;
                continue;
            }
        }
        ranges.push_back({pbeg, static_cast<std::uint32_t>(pend - pbeg), j, 1});
    }

    // Every coalesced range is read in full once per measure column (whether it
    // goes through a plain pread or the ring), so the device bytes the gather
    // moves are fixed by the plan: total range bytes times the column count.
    if (path_stats != nullptr) {
        std::uint64_t range_bytes = 0;
        for (const ReadRange& rg : ranges) range_bytes += rg.len;
        path_stats->gather_bytes += range_bytes * static_cast<std::uint64_t>(k);
    }

    // Copy one finished range's values out of a staging buffer into caller
    // positions for one column.
    const auto scatter = [&](std::size_t c, const ReadRange& rg, const char* base) {
        for (std::size_t j = rg.first; j < rg.first + rg.count; ++j) {
            const std::size_t   pos = pos_of(j);
            const std::uint64_t off = static_cast<std::uint64_t>(row_ids[pos]) * sizeof(double);
            double v;
            std::memcpy(&v, base + (off - rg.file_off), sizeof(double));
            outs[c][pos] = v;
        }
    };

    // The ring is shared across gathers; set it up once. If the kernel
    // interface is unavailable every range is read synchronously instead —
    // same plan, same results, no queue depth.
    if (!ring_init_attempted_) {
        ring_init_attempted_ = true;
        auto r = std::make_unique<IoRing>();
        if (io_uring_queue_init(kRingDepth, &r->ring, 0) == 0) {
            r->live = true;
            ring_ = std::move(r);
        }
    }

    // Ranges wider than a ring slot are dense near-sequential runs: a plain
    // positional read already streams them at readahead bandwidth, and keeping
    // them out of the ring keeps the slot buffers small and fixed.
    std::vector<char> big;
    for (const ReadRange& rg : ranges) {
        if (rg.len <= kSlotBytes && ring_) continue;
        if (big.size() < rg.len) big.resize(rg.len);
        for (std::size_t c = 0; c < k; ++c) {
            pread_exact(measure_files_[ms[c]].fd, big.data(), rg.len, rg.file_off);
            scatter(c, rg, big.data());
        }
    }
    if (!ring_) return;

    // Slot-sized ranges go through the ring as one task per (range, column),
    // interleaved across columns, with a sliding window: completions are
    // scattered out and their slots reissued immediately, so the device keeps
    // a full queue from first submission to last drain instead of stalling on
    // batch barriers.
    std::vector<std::size_t> small;
    small.reserve(ranges.size());
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        if (ranges[i].len <= kSlotBytes) small.push_back(i);
    }
    const std::size_t total = small.size() * k;
    if (total == 0) return;

    io_uring* ring = &ring_->ring;
    if (ring_->slots.size() < static_cast<std::size_t>(kRingDepth) * kSlotBytes) {
        ring_->slots.resize(static_cast<std::size_t>(kRingDepth) * kSlotBytes);
    }
    char* slots = ring_->slots.data();

    std::vector<std::size_t>   slot_task(kRingDepth, 0);
    std::vector<std::uint32_t> slot_filled(kRingDepth, 0);
    std::vector<unsigned>      free_slots(kRingDepth);
    std::iota(free_slots.begin(), free_slots.end(), 0u);

    std::size_t next_task   = 0, done = 0;
    unsigned    inflight    = 0;
    unsigned    unsubmitted = 0;  // queued SQEs not yet handed to the kernel

    const auto submit_task = [&](unsigned slot, std::size_t task,
                                 std::uint32_t already) {
        const ReadRange& rg = ranges[small[task / k]];
        const int        fd = measure_files_[ms[task % k]].fd;
        io_uring_sqe* sqe = io_uring_get_sqe(ring);
        if (sqe == nullptr) {
            // Submission queue full of not-yet-submitted entries: hand them
            // to the kernel and retry (cannot recurse — the queue is empty
            // after a successful flush).
            const int rc = io_uring_submit(ring);
            if (rc < 0) throw std::system_error(-rc, std::generic_category(), "io_uring_submit");
            unsubmitted = 0;
            sqe = io_uring_get_sqe(ring);
            if (sqe == nullptr) throw std::runtime_error("gather: submission queue stuck");
        }
        io_uring_prep_read(sqe, fd, slots + static_cast<std::size_t>(slot) * kSlotBytes + already,
                           rg.len - already,
                           static_cast<off_t>(rg.file_off + already));
        io_uring_sqe_set_data64(sqe, slot);
        slot_task[slot]   = task;
        slot_filled[slot] = already;
        ++unsubmitted;
    };

    const auto handle_cqe = [&](io_uring_cqe* cqe) {
        const auto slot = static_cast<unsigned>(io_uring_cqe_get_data64(cqe));
        const int  res  = cqe->res;
        io_uring_cqe_seen(ring, cqe);
        if (res < 0) throw std::system_error(-res, std::generic_category(), "io_uring read");
        if (res == 0) throw std::runtime_error("gather: unexpected end of file");
        const std::size_t task = slot_task[slot];
        const ReadRange&  rg   = ranges[small[task / k]];
        slot_filled[slot] += static_cast<std::uint32_t>(res);
        if (slot_filled[slot] < rg.len) {
            submit_task(slot, task, slot_filled[slot]);  // short read: continue
            return;
        }
        scatter(task % k, rg, slots + static_cast<std::size_t>(slot) * kSlotBytes);
        free_slots.push_back(slot);
        --inflight;
        ++done;
    };

    while (done < total) {
        while (inflight < kRingDepth && next_task < total && !free_slots.empty()) {
            const unsigned slot = free_slots.back();
            free_slots.pop_back();
            submit_task(slot, next_task++, 0);
            ++inflight;
        }
        if (unsubmitted > 0) {
            const int rc = io_uring_submit(ring);
            if (rc < 0) throw std::system_error(-rc, std::generic_category(), "io_uring_submit");
            unsubmitted = 0;
        }
        io_uring_cqe* cqe = nullptr;
        const int rc = io_uring_wait_cqe(ring, &cqe);
        if (rc < 0) throw std::system_error(-rc, std::generic_category(), "io_uring_wait_cqe");
        handle_cqe(cqe);
        // Drain whatever else has already completed before topping up again.
        while (io_uring_peek_cqe(ring, &cqe) == 0) handle_cqe(cqe);
    }
}

std::uint64_t BinaryColumnStore::scan_span(std::span<const MeasureId> ms,
                                           std::span<const RowId> row_ids,
                                           double* const* outs,
                                           std::span<const std::size_t> order,
                                           RowId lo, std::uint64_t span_rows) const {
    const std::size_t n = row_ids.size();
    const std::size_t k = ms.size();
    const bool sorted = order.empty();
    const auto pos_of = [&](std::size_t j) { return sorted ? j : order[j]; };

    // Walk the sorted ids and the streamed span block in lockstep. The ids are
    // ascending in span order (j index), so a single forward cursor `j` over
    // them matches each block's rows; duplicates advance the cursor without
    // re-reading. Each measure column is read once, front-to-back over the
    // span, in large sequential blocks the kernel readahead streams at full
    // bandwidth. Blocks with no wanted row are skipped, so the bytes actually
    // read (accumulated here) can be well below the full span on sparse input.
    std::uint64_t bytes_read = 0;
    std::vector<double> buf;
    for (std::size_t c = 0; c < k; ++c) {
        const OnDiskColumn& col = measure_files_[ms[c]];
        std::size_t j = 0;  // cursor into the sorted ids
        for (std::uint64_t base = 0; base < span_rows; base += kScanBlockRows) {
            const std::uint64_t count = std::min<std::uint64_t>(kScanBlockRows, span_rows - base);
            const RowId block_begin = static_cast<RowId>(lo + base);
            // Advance to the first id within this block.
            while (j < n && row_ids[pos_of(j)] < block_begin) ++j;
            if (j >= n) break;  // no more wanted rows
            // If the next wanted row is past this block, skip the read entirely.
            if (row_ids[pos_of(j)] >= block_begin + count) continue;
            if (buf.size() < count) buf.resize(count);
            pread_exact(col.fd, buf.data(), count * sizeof(double),
                        static_cast<std::uint64_t>(block_begin) * sizeof(double));
            bytes_read += count * sizeof(double);
            while (j < n) {
                const std::size_t pos = pos_of(j);
                const RowId r = row_ids[pos];
                if (r >= block_begin + count) break;
                outs[c][pos] = buf[r - block_begin];
                ++j;
            }
        }
    }
    return bytes_read;
}

std::span<const double> BinaryColumnStore::read_rows(MeasureId m, RowId begin,
                                                     std::size_t count,
                                                     std::vector<double>& scratch) const {
    if (m >= measure_count()) {
        throw std::out_of_range("read_rows: MeasureId out of range");
    }
    if (static_cast<std::uint64_t>(begin) + count > manifest_.row_count) {
        throw std::out_of_range("read_rows: row range out of bounds");
    }
    if (count == 0) return {};
    if (storage_mode_ == MeasureStorage::Eager) {
        return std::span<const double>(measure_resident_[m]).subspan(begin, count);
    }
    scratch.resize(count);
    pread_exact(measure_files_[m].fd, scratch.data(), count * sizeof(double),
                static_cast<std::uint64_t>(begin) * sizeof(double));
    return std::span<const double>(scratch.data(), count);
}

std::span<const double> BinaryColumnStore::measure_column(MeasureId m) const {
    if (m >= measure_count()) {
        throw std::out_of_range("measure_column: MeasureId out of range");
    }
    if (storage_mode_ != MeasureStorage::Eager) {
        throw std::logic_error(
            "measure_column: column is on disk; sweep it in blocks with read_rows");
    }
    return std::span<const double>(measure_resident_[m]);
}

void BinaryColumnStore::advise_sequential(MeasureId m) const {
    if (m >= measure_count()) {
        throw std::out_of_range("advise_sequential: MeasureId out of range");
    }
    // Resident columns have no file access to advise.
    if (storage_mode_ != MeasureStorage::OnDisk) return;
    ::posix_fadvise(measure_files_[m].fd, 0, 0, POSIX_FADV_SEQUENTIAL);
}

const GlobalMeasureStats& BinaryColumnStore::global_stats(MeasureId m) const {
    if (m >= exposed_measures_.size()) {
        throw std::out_of_range("global_stats: MeasureId out of range");
    }
    return manifest_.measures[exposed_measures_[m]].global;
}

}  // namespace a3i
