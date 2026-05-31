// Deterministic query-workload generator -> CSV (+ JSON sidecar).
//
// Produces a sequence of query rectangles for a prepared dataset, sized to a
// target selectivity (calibrated against a cached spatial sample, or closed
// form for uniform data). See --help for usage. A thin scripts/ wrapper exposes the same
// interface.

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "a3i/experiments/workload_generator.hpp"

namespace {

void print_usage(std::ostream& os) {
    os <<
"Usage: generate_workload --manifest <manifest.json> --out-dir <dir>\n"
"                         --dataset <id> --name <workload_id>\n"
"                         --family uniform_random|clustered\n"
"                         --extent calibrated|closed_form\n"
"                         --selectivity <fraction in (0,1]>\n"
"                         [--focus <c0,c1,...>]   (required for clustered)\n"
"                         [--seed <N>] [--count <N>]\n"
"                         [--reservoir-rows <N>] [--reservoir-seed <N>]\n"
"                         [--force]\n"
"\n"
"Writes <out-dir>/<name>.csv (fingerprint line, header, one rectangle per\n"
"row) and <out-dir>/<name>.metadata.json. Calibrated sizing builds and\n"
"caches a spatial sample under <out-dir>/reservoirs/<dataset>.bin. An\n"
"up-to-date file (matching fingerprint) is reused unless --force is given.\n"
"\n"
"Example:\n"
"  generate_workload --manifest /data/prepared/taxi/manifest.json \\\n"
"     --out-dir experiments/workloads --dataset taxi --name taxi_clustered \\\n"
"     --family clustered --extent calibrated --selectivity 0.01 \\\n"
"     --focus -73.985,40.758\n";
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

std::vector<double> parse_focus(const std::string& s) {
    std::vector<double> out;
    std::size_t pos = 0;
    while (pos <= s.size()) {
        const auto next = s.find(',', pos);
        const auto end = (next == std::string::npos) ? s.size() : next;
        if (end > pos) out.push_back(std::stod(s.substr(pos, end - pos)));
        if (next == std::string::npos) break;
        pos = next + 1;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc <= 1 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_usage(std::cout);
        return 0;
    }

    std::string manifest, out_dir, family, extent, focus_str, sel_str;
    std::string seed_str, count_str, rrows_str, rseed_str;
    a3i::WorkloadConfig cfg;
    bool force = false;

    int i = 1;
    while (i < argc) {
        std::string val;
        if (eat_kv(i, argc, argv, "--manifest", val))       { manifest = val; continue; }
        if (eat_kv(i, argc, argv, "--out-dir", val))        { out_dir = val; continue; }
        if (eat_kv(i, argc, argv, "--dataset", val))        { cfg.dataset_id = val; continue; }
        if (eat_kv(i, argc, argv, "--name", val))           { cfg.name = val; continue; }
        if (eat_kv(i, argc, argv, "--family", val))         { family = val; continue; }
        if (eat_kv(i, argc, argv, "--extent", val))         { extent = val; continue; }
        if (eat_kv(i, argc, argv, "--selectivity", val))    { sel_str = val; continue; }
        if (eat_kv(i, argc, argv, "--focus", val))          { focus_str = val; continue; }
        if (eat_kv(i, argc, argv, "--seed", val))           { seed_str = val; continue; }
        if (eat_kv(i, argc, argv, "--count", val))          { count_str = val; continue; }
        if (eat_kv(i, argc, argv, "--reservoir-rows", val)) { rrows_str = val; continue; }
        if (eat_kv(i, argc, argv, "--reservoir-seed", val)) { rseed_str = val; continue; }
        if (eat_flag(i, argc, argv, "--force"))             { force = true; continue; }
        throw std::runtime_error("unrecognized argument: " + std::string(argv[i]));
    }

    if (manifest.empty())       throw std::runtime_error("--manifest is required");
    if (out_dir.empty())        throw std::runtime_error("--out-dir is required");
    if (cfg.dataset_id.empty()) throw std::runtime_error("--dataset is required");
    if (cfg.name.empty())       throw std::runtime_error("--name is required");
    if (sel_str.empty())        throw std::runtime_error("--selectivity is required");

    if      (family == "uniform_random") cfg.family = a3i::WorkloadFamily::UniformRandom;
    else if (family == "clustered")      cfg.family = a3i::WorkloadFamily::Clustered;
    else throw std::runtime_error("--family must be uniform_random or clustered");

    if      (extent == "calibrated")  cfg.extent_mode = a3i::ExtentMode::Calibrated;
    else if (extent == "closed_form") cfg.extent_mode = a3i::ExtentMode::ClosedForm;
    else throw std::runtime_error("--extent must be calibrated or closed_form");

    cfg.selectivity = std::stod(sel_str);
    if (!focus_str.empty())  cfg.focus = parse_focus(focus_str);
    if (!seed_str.empty())   cfg.seed = std::stoull(seed_str);
    if (!count_str.empty())  cfg.count = std::stoull(count_str);
    if (!rrows_str.empty())  cfg.reservoir_rows = std::stoull(rrows_str);
    if (!rseed_str.empty())  cfg.reservoir_seed = std::stoull(rseed_str);

    const auto rep = a3i::generate_workload(cfg, manifest, out_dir, force);
    std::cout << (rep.regenerated ? "wrote " : "reused ") << rep.csv_path << '\n'
              << "  count=" << rep.count
              << "  fingerprint=" << rep.fingerprint;
    if (cfg.extent_mode == a3i::ExtentMode::Calibrated) {
        std::cout << "  achieved_selectivity_mean=" << rep.achieved_selectivity_mean;
    }
    std::cout << '\n';
    return 0;
}
catch (const std::exception& e) {
    std::cerr << "generate_workload: " << e.what() << '\n';
    return 1;
}
