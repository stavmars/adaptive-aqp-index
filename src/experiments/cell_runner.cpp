#include "a3i/experiments/cell_runner.hpp"

#include <chrono>
#include <cstdio>
#include <span>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "a3i/access_path/substrate_factory.hpp"
#include "a3i/core/query.hpp"
#include "a3i/core/schema.hpp"
#include "a3i/engine/method.hpp"
#include "a3i/engine/query_engine.hpp"
#include "a3i/experiments/truth_store.hpp"
#include "a3i/storage/binary_column_store.hpp"
#include "a3i/storage/index_table.hpp"
#include "a3i/storage/manifest.hpp"
#include "a3i/util/version.hpp"

namespace a3i {

namespace {

using nlohmann::json;
using Clock = std::chrono::steady_clock;

// The results header. One row per query; the per-measure aggregate estimates and
// the query rectangle each travel as a single JSON-encoded column, so the column
// set is fixed regardless of the measure count or dimensionality.
constexpr const char* kHeader =
    "query_ordinal,method,substrate,dataset,workload,query_rect,aggregates,"
    "target_satisfied,status,exactify_cause,pre_exactification_error_bound,"
    "sampling_seed,latency_ms,measure_reads,sampled_rows,"
    "exactified_rows,frontier_partitions,partitions_refined,exact_contributors,"
    "reusable_sampled_strata,reusable_absent_strata,query_local_strata,"
    "adaptive_rounds";

// Locale-independent, byte-stable, round-tripping double formatting.
std::string fmt(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

// Wrap a field for CSV: quote it and double any embedded quote. Used for the
// JSON columns, whose payload contains commas and quotes.
std::string csv_quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else          out += c;
    }
    out += '"';
    return out;
}

// One named method = a substrate plus a behavior (or the scan oracle).
struct MethodSpec {
    bool        is_scan = false;
    std::string substrate_id;     ///< Registered substrate; empty for scan.
    Behavior    behavior = Behavior::Plain;
    std::string substrate_label;  ///< Value written to the `substrate` column.
};

// The single source of truth for the selectable method set. A method is a
// substrate plus a behavior (or the scan oracle); its name is the catalog's
// authority, the substrate column is derived ("n/a" for scan, else the
// substrate id), and whether it is approximate follows from the behavior alone
// (`behavior_is_approx`). Every other view of this mapping (the public catalog,
// the runner's spec resolution, and any external tool that reads
// `a3i_run --describe-methods`) is derived from here.
struct MethodEntry {
    const char* name;
    bool        is_scan;
    const char* substrate_id;     // registered substrate id; "" for scan
    Behavior    behavior;
};

const MethodEntry kMethods[] = {
    {"scan",          true,  "",            Behavior::Plain},
    {"kd",            false, "static_kd",   Behavior::Plain},
    {"kd_agg",        false, "static_kd",   Behavior::Agg},
    {"adkd",          false, "adaptive_kd", Behavior::Plain},
    {"adkd_agg",      false, "adaptive_kd", Behavior::Agg},
    {"adkd_sampling", false, "adaptive_kd", Behavior::Sampling},
    {"a3i",           false, "adaptive_kd", Behavior::A3i},
};

// The label written to the `substrate` column / results path.
std::string substrate_label_of(const MethodEntry& e) {
    return e.is_scan ? "n/a" : e.substrate_id;
}

MethodSpec resolve_method(const std::string& name) {
    for (const MethodEntry& e : kMethods) {
        if (name == e.name)
            return {e.is_scan, e.substrate_id, e.behavior, substrate_label_of(e)};
    }
    throw std::invalid_argument("run_cell: unknown method '" + name + "'");
}

const char* aggregate_name(AggregateOp op) {
    switch (op) {
        case AggregateOp::Sum:          return "SUM";
        case AggregateOp::CountMeasure: return "COUNT";
        case AggregateOp::Avg:          return "AVG";
        case AggregateOp::CountStar:    return "COUNT_STAR";
    }
    return "";
}

// A workload file: a fingerprint stamp, a header that fixes the dimensionality,
// then one rectangle per row as `lower_0,...,lower_{d-1},upper_0,...,upper_{d-1}`.
struct Workload {
    std::string            fingerprint;  ///< The raw line-1 value, for provenance.
    std::size_t            dims = 0;
    std::vector<HyperRect> rects;
};

Workload read_workload(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("run_cell: cannot open workload: " + path.string());

    Workload wl;
    std::string line;
    if (!std::getline(in, line)) {
        throw std::invalid_argument("run_cell: empty workload: " + path.string());
    }
    const std::string fp_prefix = "fingerprint=";
    if (line.rfind(fp_prefix, 0) == 0) wl.fingerprint = line.substr(fp_prefix.size());

    if (!std::getline(in, line)) {
        throw std::invalid_argument("run_cell: workload has no header: " + path.string());
    }
    std::size_t lower_cols = 0;
    {
        std::stringstream hs(line);
        std::string col;
        while (std::getline(hs, col, ',')) {
            if (col.rfind("lower_", 0) == 0) ++lower_cols;
        }
    }
    if (lower_cols == 0) {
        throw std::invalid_argument("run_cell: workload header has no lower_ columns");
    }
    wl.dims = lower_cols;
    const std::size_t expected = 2 * lower_cols;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<double> vals;
        vals.reserve(expected);
        std::stringstream rs(line);
        std::string cell;
        while (std::getline(rs, cell, ',')) vals.push_back(std::stod(cell));
        if (vals.size() != expected) {
            throw std::invalid_argument("run_cell: workload row has wrong column count");
        }
        HyperRect rect;
        rect.dims.resize(lower_cols);
        for (std::size_t j = 0; j < lower_cols; ++j) {
            rect.dims[j].low  = vals[j];
            rect.dims[j].high = vals[lower_cols + j];
        }
        wl.rects.push_back(std::move(rect));
    }
    return wl;
}

// The query rectangle as `{"lower":[...],"upper":[...]}`.
json rect_json(const HyperRect& rect) {
    json lower = json::array();
    json upper = json::array();
    for (const auto& r : rect.dims) {
        lower.push_back(r.low);
        upper.push_back(r.high);
    }
    return json{{"lower", std::move(lower)}, {"upper", std::move(upper)}};
}

// The query's aggregate answers as a JSON array, one object per
// (aggregate, measure); a non-finite estimate or interval bound serialises as
// null (the JSON spelling of NaN).
json aggregates_json(const std::vector<AggregateEstimate>& aggregates,
                     const std::vector<std::string>&       measure_names) {
    json arr = json::array();
    for (const auto& e : aggregates) {
        const std::string measure =
            e.op == AggregateOp::CountStar ? "*" : measure_names[e.measure_id];
        arr.push_back(json{
            {"aggregate", aggregate_name(e.op)},
            {"measure", measure},
            {"estimate", e.estimate},
            {"ci_low", e.ci_low},
            {"ci_high", e.ci_high},
            {"relative_half_width", e.relative_half_width},
            {"effective_df", e.effective_df},
            {"exact", e.exact},
        });
    }
    return arr;
}

void write_row(std::ofstream& out, std::uint64_t ordinal, const std::string& method,
               const std::string& substrate, const std::string& dataset,
               const std::string& workload, const HyperRect& rect,
               const std::vector<AggregateEstimate>& aggregates,
               const std::vector<std::string>& measure_names, const QueryMetrics& m) {
    out << ordinal << ',' << method << ',' << substrate << ',' << dataset << ','
        << workload << ','
        << csv_quote(rect_json(rect).dump()) << ','
        << csv_quote(aggregates_json(aggregates, measure_names).dump()) << ','
        << (m.target_satisfied ? "true" : "false") << ',' << m.status << ','
        << m.exactify_cause << ',' << fmt(m.pre_exactification_error_bound) << ','
        << m.sampling_seed << ',' << fmt(m.latency_ms) << ','
        << m.measure_reads << ',' << m.sampled_rows << ','
        << m.exactified_rows << ',' << m.frontier_partitions << ','
        << m.partitions_refined << ',' << m.exact_contributors << ','
        << m.reusable_sampled_strata << ',' << m.reusable_absent_strata << ','
        << m.query_local_strata << ',' << m.adaptive_rounds << '\n';
}

// Distinct yet reproducible sampling seed material per (run, query). For run 0
// it reduces to the query index, keeping single-run results stable.
std::uint64_t seed_for(std::uint64_t run_id, std::size_t query_index) {
    return (run_id << 40) ^ static_cast<std::uint64_t>(query_index);
}

}  // namespace

const char* qresults_header() { return kHeader; }

const std::vector<MethodInfo>& method_catalog() {
    static const std::vector<MethodInfo> catalog = [] {
        std::vector<MethodInfo> v;
        for (const MethodEntry& e : kMethods)
            v.push_back({e.name, substrate_label_of(e), behavior_is_approx(e.behavior)});
        return v;
    }();
    return catalog;
}

CellReport run_cell(const CellConfig& config) {
    const MethodSpec spec = resolve_method(config.method);
    const Manifest manifest = read_manifest(config.manifest_path);
    const Workload wl = read_workload(config.workload_path);

    const std::size_t d = manifest.dimensions.size();
    if (wl.dims != d) {
        throw std::invalid_argument(
            "run_cell: workload dimensionality does not match the dataset");
    }

    // Expose the first-k measures (the subset policy lives here, in the runner).
    const std::size_t total_measures = manifest.measures.size();
    std::size_t nm = config.num_measures == 0 ? total_measures
                                              : config.num_measures;
    if (nm > total_measures) nm = total_measures;
    std::vector<MeasureId> selected;
    if (nm < total_measures) {
        selected.reserve(nm);
        for (std::size_t i = 0; i < nm; ++i) selected.push_back(static_cast<MeasureId>(i));
    }
    BinaryColumnStore store(config.manifest_path, selected, config.measure_storage);
    const std::size_t served = store.measure_count();

    DatasetSchema schema;
    for (const auto& dim : manifest.dimensions) schema.dimension_names.push_back(dim.name);
    for (std::size_t i = 0; i < served; ++i) schema.measure_names.push_back(manifest.measures[i].name);
    schema.domain_bounds        = manifest.domain_bounds;
    schema.object_count         = manifest.row_count;
    schema.binary_manifest_path = config.manifest_path.string();

    const std::string dataset_label =
        config.dataset_id.empty() ? manifest.dataset_id : config.dataset_id;
    const std::string workload_label =
        config.workload_name.empty() ? config.workload_path.stem().string()
                                      : config.workload_name;

    std::size_t to_run = wl.rects.size();
    if (config.max_queries != 0 && config.max_queries < to_run) to_run = config.max_queries;

    // The engine path needs an in-memory point table and a substrate; the scan
    // oracle needs neither.
    IndexTable table = [&] {
        // View the store's resident dimension columns directly; the AoS buffer
        // is the only allocation, so no transient duplicate of every dimension
        // column inflates the build-time peak.
        std::vector<std::span<const double>> cols;
        cols.reserve(d);
        for (std::size_t i = 0; i < d; ++i) {
            cols.push_back(store.dimension_column(static_cast<DimensionId>(i)));
        }
        return IndexTable::from_columns(
            std::span<const std::span<const double>>(cols));
    }();

    std::unique_ptr<AdaptiveAccessPath> substrate;
    std::unique_ptr<QueryEngine>        engine;
    double init_ms = 0.0;
    if (!spec.is_scan) {
        SubstrateConfig scfg;
        scfg.refinement_threshold = config.refinement_threshold;
        scfg.stochastic_cracking  = config.stochastic_cracking;
        scfg.leaf_min_size        = config.leaf_min_size;
        // Size the index to the observed data extent (per-dimension min/max),
        // not the declared domain; the declared domain is the workload's.
        scfg.data_bounds.dims.reserve(manifest.dimensions.size());
        for (const auto& dim : manifest.dimensions) {
            scfg.data_bounds.dims.push_back(Range{dim.min, dim.max});
        }
        substrate = SubstrateFactory::instance().create(spec.substrate_id, scfg);
        substrate->prepare(table);
        EngineConfig ecfg = behavior_config(spec.behavior);
        ecfg.sort_gather_by_row_id = config.sort_gather_by_row_id;
        engine = std::make_unique<QueryEngine>(store, table, *substrate,
                                               std::move(ecfg));
        // Build (and, for aggregating behaviors, precompute summaries) up front
        // so the cost is not folded into the first query's latency.
        const auto t0 = Clock::now();
        engine->initialize();
        init_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }

    std::filesystem::create_directories(config.qresults_path.parent_path());
    std::ofstream out(config.qresults_path, std::ios::trunc);
    if (!out) throw std::runtime_error("run_cell: cannot write results: " +
                                       config.qresults_path.string());
    out << kHeader << '\n';

    for (std::size_t q = 0; q < to_run; ++q) {
        const RangeQuery query{wl.rects[q], {config.error_bound, config.confidence}};
        const auto t0 = Clock::now();
        QueryResult res = spec.is_scan
                              ? exact_scan(store, schema, query)
                              : engine->execute(query, seed_for(config.run_id, q));
        const double latency =
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        QueryMetrics& m = res.metrics;
        m.query_ordinal = q;
        m.method        = config.method;
        m.substrate     = spec.substrate_label;
        m.sampling_seed = config.run_id;
        m.latency_ms    = latency;

        write_row(out, q, config.method, spec.substrate_label, dataset_label,
                  workload_label, wl.rects[q], res.aggregates, schema.measure_names, m);
    }
    out.close();

    // Sidecar: the fully resolved configuration plus provenance, so a result
    // file is self-describing and version-mixing is detectable.
    {
        json meta;
        meta["method"]               = config.method;
        meta["substrate"]            = spec.substrate_label;
        meta["dataset"]              = dataset_label;
        meta["workload"]             = workload_label;
        meta["workload_fingerprint"] = wl.fingerprint;
        meta["num_measures"]         = served;
        meta["error_bound"]          = config.error_bound;
        meta["confidence"]           = config.confidence;
        meta["refinement_threshold"] = config.refinement_threshold;
        meta["leaf_min_size"]        = config.leaf_min_size;
        meta["stochastic_cracking"]  = config.stochastic_cracking;
        meta["sort_gather_by_row_id"] = config.sort_gather_by_row_id;
        meta["measure_storage"]      =
            config.measure_storage == MeasureStorage::Eager ? "eager" : "ondisk";
        meta["run_id"]               = config.run_id;
        meta["sampling_seed"]        = config.run_id;
        meta["max_queries"]          = config.max_queries;
        meta["queries_executed"]     = to_run;
        meta["cold"]                 = config.cold;
        meta["init_ms"]              = init_ms;
        meta["manifest_path"]        = config.manifest_path.string();
        meta["workload_path"]        = config.workload_path.string();
        meta["engine_build_version"] = std::string(version());
        meta["converter_version"]    = manifest.converter_version;

        std::filesystem::create_directories(config.runmeta_path.parent_path());
        std::ofstream mout(config.runmeta_path, std::ios::trunc);
        if (!mout) throw std::runtime_error("run_cell: cannot write runmeta: " +
                                            config.runmeta_path.string());
        mout << meta.dump(2) << '\n';
    }

    CellReport report;
    report.qresults_path    = config.qresults_path;
    report.runmeta_path     = config.runmeta_path;
    report.queries_executed = to_run;
    report.measures_served  = served;
    return report;
}

}  // namespace a3i
