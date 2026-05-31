// The workload generator's contract: deterministic output, a fingerprint that
// tracks its inputs and gates regeneration, and rectangles sized to the target
// selectivity (closed-form on uniform data, calibrated against a spatial
// sample on arbitrary data).

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "a3i/experiments/workload_generator.hpp"
#include "a3i/storage/binary_column_store.hpp"
#include "a3i/tools/csv_to_parquet.hpp"
#include "a3i/tools/parquet_to_columns_pipeline.hpp"

namespace fs = std::filesystem;

namespace {

using namespace a3i;

class TempDir {
public:
    TempDir() {
        std::random_device rd;
        std::mt19937_64    rng(rd());
        path_ = fs::temp_directory_path() / ("a3i_wl_" + std::to_string(rng()));
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    fs::path operator/(const std::string& s) const { return path_ / s; }
    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

// A uniform 2-D dataset: an n x n grid of points over [0, n) x [0, n), with one
// measure. Uniform so the closed-form selectivity is also exercisable.
struct Fixture {
    TempDir                          tmp;
    std::optional<BinaryColumnStore> store;
    fs::path                         manifest_path;
    static constexpr int             kN = 100;  // 100 x 100 = 10000 rows

    Fixture() {
        const auto csv = tmp / "data.csv";
        {
            std::ofstream o(csv, std::ios::binary | std::ios::trunc);
            o << "x,y,m\n";
            for (int x = 0; x < kN; ++x)
                for (int y = 0; y < kN; ++y)
                    o << x << ',' << y << ',' << (x + y) << '\n';
        }
        const auto parquet = tmp / "data.parquet";
        CsvToParquetOptions po;
        po.input_path = csv;
        po.output_path = parquet;
        po.has_header = true;
        po.delimiter = ',';
        po.overwrite = true;
        csv_to_parquet(po);

        ConvertOptions opts;
        opts.input_parquet = parquet;
        opts.output_dir = tmp / "prepared";
        opts.dataset_id = "wl_fixture";
        opts.dimensions = {{"x", 0.0, kN}, {"y", 0.0, kN}};
        opts.measures = {"m"};
        opts.overwrite = true;
        const auto report = run_parquet_to_columns(opts);
        manifest_path = report.manifest_path;
        store.emplace(manifest_path);
    }

    WorkloadConfig base() const {
        WorkloadConfig cfg;
        cfg.dataset_id = "wl_fixture";
        cfg.name = "wl";
        cfg.family = WorkloadFamily::UniformRandom;
        cfg.extent_mode = ExtentMode::Calibrated;
        cfg.selectivity = 0.1;
        cfg.seed = 0;
        cfg.count = 200;
        return cfg;
    }
};

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::size_t count_rows(const std::string& csv) {
    // Subtract the fingerprint line and the header line.
    std::size_t lines = 0;
    for (char c : csv) if (c == '\n') ++lines;
    return lines >= 2 ? lines - 2 : 0;
}

}  // namespace

TEST(WorkloadGen, DeterministicBytesAndFingerprint) {
    Fixture f;
    const auto out_a = f.tmp / "a";
    const auto out_b = f.tmp / "b";

    const auto ra = generate_workload(f.base(), f.manifest_path, out_a);
    const auto rb = generate_workload(f.base(), f.manifest_path, out_b);

    EXPECT_TRUE(ra.regenerated);
    EXPECT_TRUE(rb.regenerated);
    EXPECT_EQ(ra.fingerprint, rb.fingerprint);
    EXPECT_EQ(ra.count, static_cast<std::size_t>(200));
    EXPECT_EQ(read_file(ra.csv_path), read_file(rb.csv_path));
    EXPECT_EQ(count_rows(read_file(ra.csv_path)), static_cast<std::size_t>(200));

    // The fingerprint is on the first line and matches the report.
    std::ifstream in(ra.csv_path);
    std::string first;
    std::getline(in, first);
    EXPECT_EQ(first, "fingerprint=" + std::to_string(ra.fingerprint));
}

TEST(WorkloadGen, FingerprintGatesRegeneration) {
    Fixture f;
    const auto out = f.tmp / "g";

    const auto first = generate_workload(f.base(), f.manifest_path, out);
    EXPECT_TRUE(first.regenerated);

    // Unchanged inputs: the up-to-date file is reused, not rewritten.
    const auto reused = generate_workload(f.base(), f.manifest_path, out);
    EXPECT_FALSE(reused.regenerated);
    EXPECT_EQ(reused.fingerprint, first.fingerprint);

    // A changed input (seed) changes the fingerprint and forces regeneration.
    WorkloadConfig changed = f.base();
    changed.seed = 1;
    const auto regen = generate_workload(changed, f.manifest_path, out);
    EXPECT_TRUE(regen.regenerated);
    EXPECT_NE(regen.fingerprint, first.fingerprint);

    // --force regenerates even when nothing changed.
    const auto forced = generate_workload(changed, f.manifest_path, out, /*force=*/true);
    EXPECT_TRUE(forced.regenerated);
    EXPECT_EQ(forced.fingerprint, regen.fingerprint);
}

TEST(WorkloadGen, CalibratedHitsTargetSelectivity) {
    Fixture f;
    WorkloadConfig cfg = f.base();
    cfg.selectivity = 0.1;
    const auto r = generate_workload(cfg, f.manifest_path, f.tmp / "cal");
    // Mean achieved selectivity within +/-20% of the target.
    EXPECT_GE(r.achieved_selectivity_mean, 0.8 * cfg.selectivity);
    EXPECT_LE(r.achieved_selectivity_mean, 1.2 * cfg.selectivity);
    EXPECT_TRUE(fs::exists(r.reservoir_path));
}

TEST(WorkloadGen, ClosedFormUniformSelectivity) {
    Fixture f;
    WorkloadConfig cfg = f.base();
    cfg.extent_mode = ExtentMode::ClosedForm;
    cfg.selectivity = 0.25;
    const auto r = generate_workload(cfg, f.manifest_path, f.tmp / "cf");
    EXPECT_TRUE(r.regenerated);
    EXPECT_TRUE(r.reservoir_path.empty());

    // On a uniform grid a closed-form rectangle of side sqrt(0.25)=0.5 of each
    // extent that fits inside the domain covers ~25% of the points. Verify a
    // mid-domain rectangle directly from the CSV by counting grid points.
    std::ifstream in(r.csv_path);
    std::string line;
    std::getline(in, line);  // fingerprint
    std::getline(in, line);  // header
    int checked = 0;
    while (std::getline(in, line) && checked < 5) {
        std::istringstream ss(line);
        double lo0, lo1, hi0, hi1;
        char c;
        ss >> lo0 >> c >> lo1 >> c >> hi0 >> c >> hi1;
        // Only assert on rectangles fully inside the domain (not clipped).
        if (lo0 <= 0.0 || lo1 <= 0.0 || hi0 >= Fixture::kN || hi1 >= Fixture::kN)
            continue;
        const double area = (hi0 - lo0) * (hi1 - lo1);
        const double frac = area / (double(Fixture::kN) * Fixture::kN);
        EXPECT_NEAR(frac, 0.25, 0.02);
        ++checked;
    }
}

TEST(WorkloadGen, ClusteredRequiresFocusPerDimension) {
    Fixture f;
    WorkloadConfig cfg = f.base();
    cfg.family = WorkloadFamily::Clustered;
    cfg.focus = {50.0};  // only one coordinate for a 2-D dataset
    EXPECT_THROW(generate_workload(cfg, f.manifest_path, f.tmp / "bad"),
                 std::invalid_argument);

    cfg.focus = {50.0, 50.0};
    EXPECT_NO_THROW(generate_workload(cfg, f.manifest_path, f.tmp / "ok"));
}
