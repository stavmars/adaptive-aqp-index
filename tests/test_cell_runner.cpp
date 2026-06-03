// The cell runner: replaying a workload through one named method writes the
// results header, the scan oracle and an exact substrate
// agree on every aggregate, and the measure subset bounds the row count.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <cmath>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "a3i/experiments/cell_runner.hpp"
#include "a3i/tools/csv_to_parquet.hpp"
#include "a3i/tools/parquet_to_columns_pipeline.hpp"

namespace fs = std::filesystem;

namespace {

using namespace a3i;

constexpr int kGrid = 20;  // 20 x 20 = 400 rows

class TempDir {
public:
    TempDir() {
        std::random_device rd;
        std::mt19937_64    rng(rd());
        path_ = fs::temp_directory_path() / ("a3i_cell_" + std::to_string(rng()));
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    fs::path operator/(const std::string& s) const { return path_ / s; }

private:
    fs::path path_;
};

// A grid dataset with two measures, built through the real conversion path so
// the runner sees a genuine manifest and binary columns.
struct Fixture {
    TempDir  tmp;
    fs::path manifest_path;
    fs::path workload_path;

    Fixture() {
        const auto csv = tmp / "data.csv";
        {
            std::ofstream o(csv, std::ios::binary | std::ios::trunc);
            o << "x,y,m0,m1\n";
            for (int x = 0; x < kGrid; ++x) {
                for (int y = 0; y < kGrid; ++y) {
                    o << x << ',' << y << ',' << (x * 100 + y) << ',';
                    if (x < 15) o << (x + y);
                    o << '\n';
                }
            }
        }
        const auto parquet = tmp / "data.parquet";
        {
            CsvToParquetOptions po;
            po.input_path  = csv;
            po.output_path = parquet;
            po.has_header  = true;
            po.delimiter   = ',';
            po.overwrite   = true;
            csv_to_parquet(po);
        }
        ConvertOptions opts;
        opts.input_parquet = parquet;
        opts.output_dir    = tmp / "prepared";
        opts.dataset_id    = "cell_fixture";
        opts.dimensions    = {{"x", 0.0, kGrid}, {"y", 0.0, kGrid}};
        opts.measures      = {"m0", "m1"};
        opts.overwrite     = true;
        manifest_path      = run_parquet_to_columns(opts).manifest_path;

        workload_path = tmp / "wl.csv";
        std::ofstream w(workload_path, std::ios::trunc);
        w << "fingerprint=424242\n";
        w << "lower_0,lower_1,upper_0,upper_1\n";
        w << "0,0,10,10\n";
        w << "5,5,20,20\n";
        w << "2,3,18,19\n";
    }

    CellConfig config(const std::string& method, std::size_t nm) const {
        CellConfig cfg;
        cfg.manifest_path = manifest_path;
        cfg.workload_path = workload_path;
        cfg.method        = method;
        cfg.num_measures  = nm;
        cfg.qresults_path = tmp / (method + ".csv");
        cfg.runmeta_path  = tmp / (method + ".meta.json");
        return cfg;
    }
};

// Parse a results CSV into header tokens + rows of cells. Fields may be quoted
// (the JSON columns are), with doubled quotes escaping a literal quote.
struct Csv {
    std::vector<std::string>              header;
    std::vector<std::vector<std::string>> rows;
    std::string                           header_line;
};

std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::string field;
    bool in_quotes = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') { field += '"'; ++i; }
                else in_quotes = false;
            } else {
                field += c;
            }
        } else if (c == '"') {
            in_quotes = true;
        } else if (c == ',') {
            out.push_back(std::move(field));
            field.clear();
        } else {
            field += c;
        }
    }
    out.push_back(std::move(field));
    return out;
}

Csv read_csv(const fs::path& p) {
    std::ifstream in(p);
    Csv csv;
    std::getline(in, csv.header_line);
    csv.header = split(csv.header_line);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) csv.rows.push_back(split(line));
    }
    return csv;
}

std::size_t col(const Csv& csv, const std::string& name) {
    for (std::size_t i = 0; i < csv.header.size(); ++i)
        if (csv.header[i] == name) return i;
    return csv.header.size();
}

// The estimate of one (aggregate, measure) from a row's `aggregates` JSON cell.
double estimate_of(const std::string& aggregates_cell, const std::string& aggregate,
                   const std::string& measure) {
    const auto arr = nlohmann::json::parse(aggregates_cell);
    for (const auto& e : arr) {
        if (e.at("aggregate") == aggregate && e.at("measure") == measure) {
            const auto& v = e.at("estimate");
            return v.is_null() ? std::nan("") : v.get<double>();
        }
    }
    throw std::runtime_error("estimate not found: " + aggregate + "/" + measure);
}

}  // namespace

TEST(CellRunner, HeaderMatchesSchema) {
    Fixture fx;
    const auto rep = run_cell(fx.config("scan", 1));
    const Csv csv  = read_csv(rep.qresults_path);
    EXPECT_EQ(csv.header_line, std::string(qresults_header()));
    // The engine-only stop_reason and the dropped phase timers stay out.
    EXPECT_EQ(col(csv, "stop_reason"), csv.header.size());
    EXPECT_EQ(col(csv, "t_locate_ms"), csv.header.size());
    EXPECT_EQ(col(csv, "t_exactify_ms"), csv.header.size());
    // Geometry and estimates ride in single JSON columns.
    EXPECT_LT(col(csv, "query_rect"), csv.header.size());
    EXPECT_LT(col(csv, "aggregates"), csv.header.size());
}

TEST(CellRunner, OneRowPerQueryRegardlessOfMeasures) {
    Fixture fx;
    // 3 queries -> 3 rows whatever the measure count; the per-measure estimates
    // live inside each row's aggregates cell (3 per measure + one COUNT_STAR).
    const auto one  = run_cell(fx.config("scan", 1));
    const Csv  c1   = read_csv(one.qresults_path);
    EXPECT_EQ(one.measures_served, 1u);
    EXPECT_EQ(c1.rows.size(), 3u);
    EXPECT_EQ(nlohmann::json::parse(c1.rows[0][col(c1, "aggregates")]).size(), 4u);

    const auto all = run_cell(fx.config("scan", 0));
    const Csv  c2  = read_csv(all.qresults_path);
    EXPECT_EQ(all.measures_served, 2u);
    EXPECT_EQ(c2.rows.size(), 3u);
    EXPECT_EQ(nlohmann::json::parse(c2.rows[0][col(c2, "aggregates")]).size(), 7u);
}

TEST(CellRunner, CountStarMeasureIsAsterisk) {
    Fixture fx;
    const auto rep = run_cell(fx.config("scan", 2));
    const Csv  csv = read_csv(rep.qresults_path);
    const std::size_t a = col(csv, "aggregates");
    for (const auto& r : csv.rows) {
        bool seen = false;
        for (const auto& e : nlohmann::json::parse(r[a])) {
            if (e.at("aggregate") == "COUNT_STAR") {
                EXPECT_EQ(e.at("measure"), "*");
                seen = true;
            }
        }
        EXPECT_TRUE(seen);
    }
}

TEST(CellRunner, QueryRectMatchesWorkload) {
    Fixture fx;
    const auto rep = run_cell(fx.config("scan", 1));
    const Csv  csv = read_csv(rep.qresults_path);
    const std::size_t rc = col(csv, "query_rect");
    // The first workload row is "0,0,10,10".
    const auto rect = nlohmann::json::parse(csv.rows[0][rc]);
    EXPECT_EQ(rect.at("lower"), (std::vector<double>{0.0, 0.0}));
    EXPECT_EQ(rect.at("upper"), (std::vector<double>{10.0, 10.0}));
}

TEST(CellRunner, ScanAndExactSubstrateAgree) {
    Fixture fx;
    const Csv scan = read_csv(run_cell(fx.config("scan", 2)).qresults_path);
    const Csv kd   = read_csv(run_cell(fx.config("kd", 2)).qresults_path);  // static_kd, exact
    ASSERT_EQ(scan.rows.size(), kd.rows.size());

    const std::size_t a = col(scan, "aggregates");
    const std::vector<std::pair<std::string, std::string>> keys = {
        {"SUM", "m0"}, {"COUNT", "m0"}, {"AVG", "m0"},
        {"SUM", "m1"}, {"COUNT", "m1"}, {"AVG", "m1"}, {"COUNT_STAR", "*"}};
    for (std::size_t i = 0; i < scan.rows.size(); ++i) {
        for (const auto& [agg, mea] : keys) {
            const double s = estimate_of(scan.rows[i][a], agg, mea);
            const double k = estimate_of(kd.rows[i][a], agg, mea);
            if (std::isnan(s)) { EXPECT_TRUE(std::isnan(k)); continue; }
            // Both exact; only summation order differs (global vs per-partition).
            EXPECT_NEAR(s, k, 1e-9 * (1.0 + std::abs(s))) << agg << "/" << mea;
        }
    }
}


TEST(CellRunner, RunmetaRecordsResolvedConfig) {
    Fixture fx;
    auto cfg        = fx.config("a3i", 1);
    cfg.error_bound = 0.05;
    cfg.run_id      = 3;
    const auto rep  = run_cell(cfg);

    std::ifstream in(rep.runmeta_path);
    nlohmann::json meta;
    in >> meta;
    EXPECT_EQ(meta["method"], "a3i");
    EXPECT_EQ(meta["substrate"], "adaptive_kd");
    EXPECT_EQ(meta["num_measures"], 1);
    EXPECT_EQ(meta["run_id"], 3);
    EXPECT_EQ(meta["sampling_seed"], 3);
    EXPECT_EQ(meta["queries_executed"], 3);
    EXPECT_EQ(meta["workload_fingerprint"], "424242");
}

TEST(CellRunner, UnknownMethodThrows) {
    Fixture fx;
    EXPECT_THROW(run_cell(fx.config("nope", 1)), std::invalid_argument);
}

TEST(CellRunner, MethodCatalogIsSingleSourceOfTruth) {
    // The catalog is what external drivers consume; it must list every runnable
    // method with the substrate label that lands in the results path, and flag
    // exactly the methods that consume the per-query error bound.
    const auto& cat = a3i::method_catalog();
    std::map<std::string, a3i::MethodInfo> by_name;
    for (const auto& m : cat) by_name[m.name] = m;

    EXPECT_EQ(by_name.at("scan").substrate, "n/a");
    EXPECT_FALSE(by_name.at("scan").approx);
    EXPECT_EQ(by_name.at("kd").substrate, "static_kd");
    EXPECT_EQ(by_name.at("a3i").substrate, "adaptive_kd");

    std::map<std::string, bool> expected_approx = {
        {"scan", false}, {"kd", false}, {"kd_agg", false},
        {"adkd", false}, {"adkd_agg", false}, {"adkd_sampling", true},
        {"a3i", true},
    };
    ASSERT_EQ(by_name.size(), expected_approx.size());
    for (const auto& [name, approx] : expected_approx) {
        ASSERT_TRUE(by_name.count(name)) << "catalog missing method " << name;
        EXPECT_EQ(by_name.at(name).approx, approx) << "approx mismatch for " << name;
    }
}

