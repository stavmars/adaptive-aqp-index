#include "a3i/experiments/workload_generator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "a3i/storage/binary_column_store.hpp"
#include "a3i/util/rng.hpp"

namespace a3i {

namespace {

using nlohmann::json;

constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime  = 1099511628211ull;

std::uint64_t fnv1a(const std::string& s) {
    std::uint64_t h = kFnvOffset;
    for (unsigned char c : s) {
        h ^= c;
        h *= kFnvPrime;
    }
    return h;
}

// Format a double losslessly and locale-independently so the fingerprint and
// the CSV are byte-stable across runs and machines.
std::string fmt(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

std::string family_id(WorkloadFamily f) {
    return f == WorkloadFamily::UniformRandom ? "uniform_random" : "clustered";
}

std::string extent_id(ExtentMode m) {
    return m == ExtentMode::Calibrated ? "calibrated" : "closed_form";
}

// A spatial sample: `rows` points of `dims` coordinates, interleaved row-major
// (x0,y0,x1,y1,...), matching the on-disk reservoir layout.
struct Reservoir {
    std::size_t         dims = 0;
    std::vector<double> points;
    std::size_t rows() const { return dims ? points.size() / dims : 0; }
};

// Classic reservoir sampling (Algorithm R) over the in-memory dimension
// columns: one O(N) pass, O(target) memory, deterministic for a given seed.
// When the dataset has at most `target` rows every row is kept.
Reservoir build_reservoir(const BinaryColumnStore& store, std::size_t target,
                          std::uint64_t seed) {
    const std::size_t d = store.dimension_count();
    const std::size_t n = store.row_count();
    Reservoir res;
    res.dims = d;
    if (d == 0 || n == 0 || target == 0) return res;

    std::vector<std::span<const double>> cols;
    cols.reserve(d);
    for (std::size_t j = 0; j < d; ++j) cols.push_back(store.dimension_column(j));

    const std::size_t keep = std::min(target, n);
    std::vector<std::size_t> idx(keep);
    for (std::size_t i = 0; i < keep; ++i) idx[i] = i;

    Rng rng(seed);
    for (std::size_t i = keep; i < n; ++i) {
        std::uniform_int_distribution<std::size_t> pick(0, i);
        const std::size_t j = pick(rng);
        if (j < keep) idx[j] = i;
    }

    res.points.resize(keep * d);
    for (std::size_t i = 0; i < keep; ++i) {
        const std::size_t row = idx[i];
        for (std::size_t j = 0; j < d; ++j) res.points[i * d + j] = cols[j][row];
    }
    return res;
}

void write_reservoir(const Reservoir& res, const std::filesystem::path& bin,
                     const std::filesystem::path& sidecar,
                     const std::string& dataset_id, std::uint64_t seed) {
    std::filesystem::create_directories(bin.parent_path());
    std::ofstream out(bin, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write reservoir: " + bin.string());
    if (!res.points.empty()) {
        out.write(reinterpret_cast<const char*>(res.points.data()),
                  static_cast<std::streamsize>(res.points.size() * sizeof(double)));
    }
    out.close();

    json j;
    j["dataset_id"] = dataset_id;
    j["dims"]       = res.dims;
    j["rows"]       = res.rows();
    j["seed"]       = seed;
    std::ofstream sj(sidecar, std::ios::trunc);
    sj << j.dump(2) << '\n';
}

// Read a cached reservoir if its sidecar matches the requested dims/rows/seed;
// returns false (leaving `res` empty) when absent or stale, so the caller
// rebuilds it.
bool read_reservoir(const std::filesystem::path& bin,
                    const std::filesystem::path& sidecar, std::size_t dims,
                    std::size_t want_rows, std::uint64_t seed, Reservoir& res) {
    std::ifstream sj(sidecar);
    if (!sj) return false;
    json j;
    try {
        sj >> j;
    } catch (const json::exception&) {
        return false;
    }
    if (j.value("dims", std::size_t{0}) != dims) return false;
    if (j.value("seed", std::uint64_t{0}) != seed) return false;
    const std::size_t rows = j.value("rows", std::size_t{0});
    // A smaller dataset legitimately yields fewer than the requested rows; the
    // recorded count is authoritative as long as the seed and dims agree.
    if (want_rows != 0 && rows > want_rows) return false;

    std::ifstream in(bin, std::ios::binary);
    if (!in) return false;
    res.dims = dims;
    res.points.resize(rows * dims);
    if (rows > 0) {
        in.read(reinterpret_cast<char*>(res.points.data()),
                static_cast<std::streamsize>(res.points.size() * sizeof(double)));
        if (!in) return false;
    }
    return true;
}

double fraction_inside(const Reservoir& res, const HyperRect& rect) {
    const std::size_t r = res.rows();
    if (r == 0) return 0.0;
    std::size_t hits = 0;
    for (std::size_t i = 0; i < r; ++i) {
        std::span<const double> p(res.points.data() + i * res.dims, res.dims);
        if (rect.contains_point(p)) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(r);
}

// A rectangle covering fraction `t` of each dimension's extent, centred at `c`.
// A window that would overflow a domain edge is shifted inward,
// so its extent (and hence selectivity) is preserved at the edges;
// since `t <= 1` the shifted window always fits. Monotone in `t`, which
// calibration relies on.
HyperRect rect_at(const std::vector<double>& c, const HyperRect& domain,
                  double t) {
    HyperRect rect;
    rect.dims.resize(domain.dims.size());
    for (std::size_t j = 0; j < domain.dims.size(); ++j) {
        const double lo = domain.dims[j].low;
        const double hi = domain.dims[j].high;
        const double half = 0.5 * t * (hi - lo);
        double centre = c[j];
        if (centre - half < lo) centre = lo + half;
        else if (centre + half > hi) centre = hi - half;
        rect.dims[j].low  = centre - half;
        rect.dims[j].high = centre + half;
    }
    return rect;
}

// Bisection on the per-dimension extent fraction until the sample selectivity
// lands within tolerance of the target, capped at a fixed iteration count;
// selectivity is monotone in the fraction, so bisection converges.
HyperRect calibrate(const std::vector<double>& c, const HyperRect& domain,
                    const Reservoir& res, double target, double& achieved) {
    constexpr int    kMaxIters = 20;
    constexpr double kTol = 0.20;  // achieved within +/-20% of target
    double lo = 0.0, hi = 1.0;
    HyperRect best = rect_at(c, domain, hi);
    achieved = fraction_inside(res, best);
    double best_err = std::abs(achieved - target);

    for (int it = 0; it < kMaxIters; ++it) {
        const double mid = 0.5 * (lo + hi);
        HyperRect rect = rect_at(c, domain, mid);
        const double f = fraction_inside(res, rect);
        const double err = std::abs(f - target);
        if (err < best_err) {
            best_err = err;
            best = rect;
            achieved = f;
        }
        if (target > 0.0 && std::abs(f - target) <= kTol * target) {
            best = rect;
            achieved = f;
            break;
        }
        if (f < target) lo = mid; else hi = mid;
    }
    return best;
}

// Closed-form rectangle for uniform data: an equal share s^(1/d) of every
// dimension's extent, centred at `c` and clipped to the domain.
HyperRect closed_form(const std::vector<double>& c, const HyperRect& domain,
                      double target) {
    const double t = std::pow(target, 1.0 / static_cast<double>(domain.dims.size()));
    return rect_at(c, domain, t);
}

std::vector<double> draw_centre(const WorkloadConfig& cfg, const HyperRect& domain,
                                Rng& rng) {
    const std::size_t d = domain.dims.size();
    std::vector<double> c(d);
    if (cfg.family == WorkloadFamily::UniformRandom) {
        for (std::size_t j = 0; j < d; ++j) {
            std::uniform_real_distribution<double> u(domain.dims[j].low,
                                                     domain.dims[j].high);
            c[j] = u(rng);
        }
    } else {
        for (std::size_t j = 0; j < d; ++j) {
            const double extent = domain.dims[j].high - domain.dims[j].low;
            std::normal_distribution<double> g(cfg.focus[j], 0.05 * extent);
            double v = g(rng);
            v = std::clamp(v, domain.dims[j].low, domain.dims[j].high);
            c[j] = v;
        }
    }
    return c;
}

// The centre sequence for the whole workload. For a uniform-random family on a
// calibrated dataset the centres follow the data distribution: they are drawn
// from the spatial reservoir without replacement (partial Fisher-Yates over the
// reservoir indices, reused cyclically once one pass is exhausted), so query
// rectangles track where the data actually is instead of landing in empty
// regions of a skewed domain. Without a reservoir (closed-form, uniform data)
// the centres are uniform over the domain. Clustered centres are Gaussian
// around the configured focus. The draw order is fully determined by the seed.
std::vector<std::vector<double>> build_centres(const WorkloadConfig& cfg,
                                               const HyperRect& domain,
                                               const Reservoir* res, Rng& rng) {
    const std::size_t d = domain.dims.size();
    std::vector<std::vector<double>> centres;
    centres.reserve(cfg.count);

    if (cfg.family == WorkloadFamily::UniformRandom && res && res->rows() > 0) {
        const std::size_t n = res->rows();
        const std::size_t len = std::max(cfg.count, n);
        std::vector<std::size_t> order(len);
        for (std::size_t i = 0; i < len; ++i) order[i] = i % n;
        const std::size_t picks = std::min(cfg.count, n);
        for (std::size_t i = 0; i < picks; ++i) {
            std::uniform_int_distribution<std::size_t> u(i, n - 1);
            std::swap(order[i], order[u(rng)]);
        }
        for (std::size_t q = 0; q < cfg.count; ++q) {
            const double* p = res->points.data() + order[q] * res->dims;
            centres.emplace_back(p, p + d);
        }
        return centres;
    }

    for (std::size_t q = 0; q < cfg.count; ++q) {
        centres.push_back(draw_centre(cfg, domain, rng));
    }
    return centres;
}

}  // namespace

std::uint64_t workload_fingerprint(const WorkloadConfig& config,
                                   const HyperRect& domain) {
    std::string s;
    s += "gen=" + family_id(config.family);
    s += ";dataset=" + config.dataset_id;
    s += ";extent=" + extent_id(config.extent_mode);
    s += ";sel=" + fmt(config.selectivity);
    s += ";focus=";
    for (double f : config.focus) s += fmt(f) + ",";
    s += ";seed=" + std::to_string(config.seed);
    s += ";count=" + std::to_string(config.count);
    s += ";domain=";
    for (const auto& r : domain.dims) s += fmt(r.low) + "," + fmt(r.high) + ",";
    return fnv1a(s);
}

WorkloadReport generate_workload(const WorkloadConfig& config,
                                 const std::filesystem::path& manifest_path,
                                 const std::filesystem::path& out_dir,
                                 bool force) {
    BinaryColumnStore store(manifest_path);
    const HyperRect domain = store.manifest().domain_bounds;
    const std::size_t d = domain.dims.size();
    if (d == 0) throw std::runtime_error("workload: dataset has no dimensions");
    if (config.selectivity <= 0.0 || config.selectivity > 1.0) {
        throw std::invalid_argument("workload: selectivity must be in (0, 1]");
    }
    if (config.family == WorkloadFamily::Clustered && config.focus.size() != d) {
        throw std::invalid_argument(
            "workload: clustered family needs one focus coordinate per dimension");
    }

    std::filesystem::create_directories(out_dir);
    const auto csv_path = out_dir / (config.name + ".csv");
    const auto meta_path = out_dir / (config.name + ".metadata.json");
    const std::uint64_t fp = workload_fingerprint(config, domain);

    WorkloadReport report;
    report.csv_path = csv_path;
    report.metadata_path = meta_path;
    report.fingerprint = fp;
    report.count = config.count;

    // Reuse an up-to-date file: its first line records the fingerprint.
    if (!force && std::filesystem::exists(csv_path)) {
        std::ifstream in(csv_path);
        std::string line;
        if (std::getline(in, line)) {
            const std::string prefix = "fingerprint=";
            if (line.rfind(prefix, 0) == 0 &&
                line.substr(prefix.size()) == std::to_string(fp)) {
                report.regenerated = false;
                return report;
            }
        }
    }

    // Calibrated sizing needs a spatial sample; build and cache it if missing
    // or stale. ClosedForm needs none.
    Reservoir res;
    std::filesystem::path bin_path;
    if (config.extent_mode == ExtentMode::Calibrated) {
        const auto res_dir = out_dir / "reservoirs";
        bin_path = res_dir / (config.dataset_id + ".bin");
        const auto res_sidecar = res_dir / (config.dataset_id + ".json");
        if (!read_reservoir(bin_path, res_sidecar, d, config.reservoir_rows,
                            config.reservoir_seed, res)) {
            res = build_reservoir(store, config.reservoir_rows, config.reservoir_seed);
            write_reservoir(res, bin_path, res_sidecar, config.dataset_id,
                            config.reservoir_seed);
        }
        report.reservoir_path = bin_path;
    }

    Rng rng(config.seed);
    const Reservoir* res_ptr =
        config.extent_mode == ExtentMode::Calibrated ? &res : nullptr;
    const std::vector<std::vector<double>> centres =
        build_centres(config, domain, res_ptr, rng);

    std::vector<HyperRect> rects;
    rects.reserve(config.count);
    double achieved_sum = 0.0;
    for (std::size_t q = 0; q < config.count; ++q) {
        const std::vector<double>& c = centres[q];
        HyperRect rect;
        if (config.extent_mode == ExtentMode::Calibrated) {
            double achieved = 0.0;
            rect = calibrate(c, domain, res, config.selectivity, achieved);
            achieved_sum += achieved;
        } else {
            rect = closed_form(c, domain, config.selectivity);
        }
        rects.push_back(std::move(rect));
    }
    if (config.extent_mode == ExtentMode::Calibrated && config.count > 0) {
        report.achieved_selectivity_mean = achieved_sum / static_cast<double>(config.count);
    }

    // Write the CSV: fingerprint line, header, then one rectangle per row.
    {
        std::ofstream out(csv_path, std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write workload: " + csv_path.string());
        out << "fingerprint=" << fp << '\n';
        for (std::size_t j = 0; j < d; ++j) out << "lower_" << j << ',';
        for (std::size_t j = 0; j < d; ++j) {
            out << "upper_" << j;
            out << (j + 1 < d ? ',' : '\n');
        }
        for (const auto& rect : rects) {
            for (std::size_t j = 0; j < d; ++j) out << fmt(rect.dims[j].low) << ',';
            for (std::size_t j = 0; j < d; ++j) {
                out << fmt(rect.dims[j].high);
                out << (j + 1 < d ? ',' : '\n');
            }
        }
    }

    // Sidecar metadata: the full configuration plus what calibration achieved.
    {
        json meta;
        meta["dataset_id"]  = config.dataset_id;
        meta["name"]        = config.name;
        meta["family"]      = family_id(config.family);
        meta["extent_mode"] = extent_id(config.extent_mode);
        meta["selectivity"] = config.selectivity;
        meta["focus"]       = config.focus;
        meta["seed"]        = config.seed;
        meta["count"]       = config.count;
        meta["dimensions"]  = d;
        meta["fingerprint"] = fp;
        json bounds_lo = json::array(), bounds_hi = json::array();
        for (const auto& r : domain.dims) {
            bounds_lo.push_back(r.low);
            bounds_hi.push_back(r.high);
        }
        meta["domain_bounds"] = {{"low", bounds_lo}, {"high", bounds_hi}};
        if (config.extent_mode == ExtentMode::Calibrated) {
            meta["reservoir_rows"] = res.rows();
            meta["reservoir_seed"] = config.reservoir_seed;
            meta["achieved_selectivity_mean"] = report.achieved_selectivity_mean;
        }
        std::ofstream out(meta_path, std::ios::trunc);
        out << meta.dump(2) << '\n';
    }

    report.regenerated = true;
    return report;
}

}  // namespace a3i
