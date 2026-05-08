#include "a3i/storage/binary_column_store.hpp"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <system_error>

#include <fcntl.h>
#include <sys/mman.h>
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

}  // namespace

BinaryColumnStore::BinaryColumnStore(const std::filesystem::path& manifest_path)
    : manifest_(read_manifest(manifest_path)),
      manifest_dir_(manifest_path.parent_path()) {

    // Dimension columns are loaded eagerly: the adaptive index needs to read,
    // sort, and permute coordinates freely in RAM during grid construction and
    // cracking. The full column fits comfortably for typical dataset sizes; if
    // this ever needs to change the right place is here, not the access path.
    dim_columns_.reserve(manifest_.dimensions.size());
    for (const auto& d : manifest_.dimensions) {
        const auto path = manifest_dir_ / d.file;
        dim_columns_.push_back(read_double_file(path, manifest_.row_count));
    }

    // Measure columns are memory-mapped rather than loaded. Sampling touches
    // scattered rows; loading the whole column wastes memory and startup time.
    // MADV_RANDOM tells the kernel not to read-ahead past the touched page —
    // appropriate for point-lookups. Callers that will sweep a contiguous range
    // should promote the relevant region to MADV_SEQUENTIAL / MADV_WILLNEED before calling gather().
    measure_regions_.resize(manifest_.measures.size());
    for (std::size_t i = 0; i < manifest_.measures.size(); ++i) {
        const auto path = manifest_dir_ / manifest_.measures[i].file;
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

        void* base = nullptr;
        if (bytes > 0) {
            base = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
            if (base == MAP_FAILED) {
                ::close(fd);
                throw std::runtime_error("mmap failed: " + path.string() + ": " +
                                         std::strerror(errno));
            }
            ::madvise(base, bytes, MADV_RANDOM);
        }

        auto& r = measure_regions_[i];
        r.base   = base;
        r.bytes  = bytes;
        r.fd     = fd;
        r.length = manifest_.row_count;
        r.data   = static_cast<const double*>(base);
    }
}

BinaryColumnStore::~BinaryColumnStore() {
    for (auto& r : measure_regions_) {
        if (r.base && r.bytes > 0) ::munmap(r.base, r.bytes);
        if (r.fd  >= 0)            ::close(r.fd);
    }
}

std::span<const double> BinaryColumnStore::dimension_column(DimensionId d) const {
    if (d >= dim_columns_.size()) {
        throw std::out_of_range("dimension_column: DimensionId out of range");
    }
    return std::span<const double>(dim_columns_[d]);
}

double BinaryColumnStore::measure_value(RowId r, MeasureId m) const {
    if (m >= measure_regions_.size()) {
        throw std::out_of_range("measure_value: MeasureId out of range");
    }
    const auto& reg = measure_regions_[m];
    if (r >= reg.length) {
        throw std::out_of_range("measure_value: RowId out of range");
    }
    return reg.data[r];
}

void BinaryColumnStore::gather(MeasureId m,
                               std::span<const RowId> row_ids,
                               std::span<double> out) const {
    if (m >= measure_regions_.size()) {
        throw std::out_of_range("gather: MeasureId out of range");
    }
    if (row_ids.size() != out.size()) {
        throw std::invalid_argument("gather: row_ids and out must have equal size");
    }
    const auto& reg = measure_regions_[m];

    // Read in the order given by the caller; out[i] = value at row_ids[i].
    // Reordering for read locality is the caller's responsibility and should
    // be decided with measurements in hand.
    for (std::size_t i = 0; i < row_ids.size(); ++i) {
        const RowId r = row_ids[i];
        if (r >= reg.length) {
            throw std::out_of_range("gather: RowId out of range");
        }
        out[i] = reg.data[r];
    }
}

const GlobalMeasureStats& BinaryColumnStore::global_stats(MeasureId m) const {
    if (m >= manifest_.measures.size()) {
        throw std::out_of_range("global_stats: MeasureId out of range");
    }
    return manifest_.measures[m].global;
}

}  // namespace a3i
