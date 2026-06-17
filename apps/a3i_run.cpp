// Run one experiment cell: a single (method, dataset, workload, run).
//
// Replays a workload of rectangles through one named method and writes a
// results CSV plus a JSON sidecar. See --help for usage. The cell
// logic lives in the a3i_core library; this is a thin argument front end.

#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

#include "a3i/experiments/cell_runner.hpp"
#include "a3i/util/version.hpp"

namespace {

void print_usage(std::ostream& os) {
    os <<
"Usage: a3i_run --manifest <manifest.json> --workload <workload.csv>\n"
"               --method <name> --qresults <out.csv> --runmeta <out.json>\n"
"               [--dataset <id>] [--workload-name <id>]\n"
"               [--num-measures <k>] [--error-bound <eps>] [--confidence <c>]\n"
"               [--partition-size <N>]\n"
"               [--stochastic-cracking] [--no-sort-gather] [--in-memory]\n"
"               [--run-id <R>] [--max-queries <N>] [--cold true|false]\n"
"\n"
"Methods:\n"
"  scan           exact-scan oracle (no substrate)\n"
"  kd | kd_agg | static_kd + plain | aggregating | accuracy-aware\n"
"  akd | akd_agg | akd_sampling | a3i_akd\n"
"                 adaptive_kd + plain | aggregating | sampling | accuracy-aware\n"
"\n"
"  --describe-methods   print the method catalog as JSON and exit\n"
"  --version            print the engine build version and exit\n"
"\n"
"Writes one CSV row per query (the per-measure aggregate estimates and the query\n"
"rectangle each in a single JSON column), and a runmeta sidecar with the resolved\n"
"configuration and provenance. The accuracy target is applied per query; the\n"
"workload rows carry only geometry.\n"
"\n"
"Example:\n"
"  a3i_run --manifest /data/prepared/taxi/manifest.json \\\n"
"     --workload experiments/workloads/taxi_clustered.csv --method a3i \\\n"
"     --error-bound 0.05 --num-measures 1 \\\n"
"     --qresults results/taxi/q.csv --runmeta results/taxi/q.meta.json\n";
}

bool eat_flag(int& i, int argc, char** argv, std::string_view name) {
    if (i < argc && argv[i] == name) { ++i; return true; }
    return false;
}

bool eat_kv(int& i, int argc, char** argv, std::string_view name, std::string& out) {
    if (i < argc && argv[i] == name) {
        if (i + 1 >= argc) throw std::runtime_error(std::string(name) + " requires a value");
        out = argv[i + 1];
        i += 2;
        return true;
    }
    return false;
}

bool parse_bool(const std::string& s) {
    if (s == "true" || s == "1")  return true;
    if (s == "false" || s == "0") return false;
    throw std::runtime_error("expected true or false, got: " + s);
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc <= 1 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_usage(std::cout);
        return 0;
    }
    if (std::string(argv[1]) == "--version") {
        std::cout << a3i::version() << '\n';
        return 0;
    }
    if (std::string(argv[1]) == "--describe-methods") {
        // Emit the method catalog as JSON so external drivers read the
        // method->substrate / approx mapping from here instead of restating it.
        const auto& methods = a3i::method_catalog();
        std::cout << "[";
        for (std::size_t i = 0; i < methods.size(); ++i) {
            const auto& m = methods[i];
            std::cout << (i ? "," : "")
                      << "{\"method\":\"" << m.name << "\","
                      << "\"substrate\":\"" << m.substrate << "\","
                      << "\"approx\":" << (m.approx ? "true" : "false") << "}";
        }
        std::cout << "]\n";
        return 0;
    }

    a3i::CellConfig cfg;
    std::string manifest, workload, qresults, runmeta;
    std::string nm_str, eb_str, conf_str, psize_str, run_str, maxq_str, cold_str;

    int i = 1;
    while (i < argc) {
        std::string val;
        if (eat_kv(i, argc, argv, "--manifest", val))             { manifest = val; continue; }
        if (eat_kv(i, argc, argv, "--workload", val))             { workload = val; continue; }
        if (eat_kv(i, argc, argv, "--method", val))               { cfg.method = val; continue; }
        if (eat_kv(i, argc, argv, "--qresults", val))             { qresults = val; continue; }
        if (eat_kv(i, argc, argv, "--runmeta", val))              { runmeta = val; continue; }
        if (eat_kv(i, argc, argv, "--dataset", val))              { cfg.dataset_id = val; continue; }
        if (eat_kv(i, argc, argv, "--workload-name", val))        { cfg.workload_name = val; continue; }
        if (eat_kv(i, argc, argv, "--num-measures", val))         { nm_str = val; continue; }
        if (eat_kv(i, argc, argv, "--error-bound", val))          { eb_str = val; continue; }
        if (eat_kv(i, argc, argv, "--confidence", val))           { conf_str = val; continue; }
        if (eat_kv(i, argc, argv, "--partition-size", val))       { psize_str = val; continue; }
        if (eat_kv(i, argc, argv, "--run-id", val))               { run_str = val; continue; }
        if (eat_kv(i, argc, argv, "--max-queries", val))          { maxq_str = val; continue; }
        if (eat_kv(i, argc, argv, "--cold", val))                 { cold_str = val; continue; }
        if (eat_flag(i, argc, argv, "--stochastic-cracking"))     { cfg.stochastic_cracking = true; continue; }
        if (eat_flag(i, argc, argv, "--no-sort-gather"))          { cfg.sort_gather_by_row_id = false; continue; }
        if (eat_flag(i, argc, argv, "--in-memory"))               { cfg.measure_storage = a3i::MeasureStorage::Eager; continue; }
        throw std::runtime_error("unrecognized argument: " + std::string(argv[i]));
    }

    if (manifest.empty())    throw std::runtime_error("--manifest is required");
    if (workload.empty())    throw std::runtime_error("--workload is required");
    if (cfg.method.empty())  throw std::runtime_error("--method is required");
    if (qresults.empty())    throw std::runtime_error("--qresults is required");
    if (runmeta.empty())     throw std::runtime_error("--runmeta is required");

    cfg.manifest_path = manifest;
    cfg.workload_path = workload;
    cfg.qresults_path = qresults;
    cfg.runmeta_path  = runmeta;

    if (!nm_str.empty())   cfg.num_measures         = std::stoull(nm_str);
    if (!eb_str.empty())   cfg.error_bound          = std::stod(eb_str);
    if (!conf_str.empty()) cfg.confidence           = std::stod(conf_str);
    if (!psize_str.empty()) cfg.partition_size = static_cast<std::uint32_t>(std::stoul(psize_str));
    if (!run_str.empty())  cfg.run_id               = std::stoull(run_str);
    if (!maxq_str.empty()) cfg.max_queries          = std::stoull(maxq_str);
    if (!cold_str.empty()) cfg.cold                 = parse_bool(cold_str);

    const auto rep = a3i::run_cell(cfg);
    std::cout << "wrote " << rep.qresults_path << '\n'
              << "  queries=" << rep.queries_executed
              << "  measures=" << rep.measures_served
              << "  runmeta=" << rep.runmeta_path << '\n';
    return 0;
}
catch (const std::exception& e) {
    std::cerr << "a3i_run: " << e.what() << '\n';
    return 1;
}
