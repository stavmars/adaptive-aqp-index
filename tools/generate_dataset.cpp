// Seeded synthetic dataset generator -> Parquet.
//
// Writes a deterministic, byte-reproducible Parquet file from a (seed, rows,
// column-spec) configuration. Used only offline in data preparation. See
// --help for usage. A thin scripts/ wrapper exposes the same interface.

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "a3i/tools/synthetic_generator.hpp"

namespace {

void print_usage(std::ostream& os) {
    os <<
"Usage: generate_dataset --output <parquet> --seed <N> --rows <N>\n"
"                        --column <name>:<dist>:<p0>:<p1>   (repeatable)\n"
"                        [--overwrite]\n"
"\n"
"Generates a deterministic Parquet: the same (seed, rows, columns)\n"
"yields a byte-identical file. Each column draws `rows` values from a\n"
"distribution:\n"
"  uniform:<lo>:<hi>     values in [lo, hi)\n"
"  normal:<mean>:<sd>    Gaussian(mean, sd)\n"
"  lognormal:<mu>:<sigma> exp of Gaussian(mu, sigma)\n"
"\n"
"Example:\n"
"  generate_dataset --output /data/parquet/synth.parquet --seed 42 --rows 1000000 \\\n"
"     --column x:uniform:0:1000 --column y:uniform:0:1000 \\\n"
"     --column m:normal:50:15\n";
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

std::vector<std::string> split(std::string_view s, char sep) {
    std::vector<std::string> out;
    std::size_t pos = 0;
    while (pos <= s.size()) {
        const auto next = s.find(sep, pos);
        const auto end = (next == std::string_view::npos) ? s.size() : next;
        out.emplace_back(s.substr(pos, end - pos));
        if (next == std::string_view::npos) break;
        pos = next + 1;
    }
    return out;
}

a3i::SyntheticColumnSpec parse_column(std::string_view s) {
    const auto parts = split(s, ':');
    if (parts.size() != 4) {
        throw std::runtime_error(
            "--column expects <name>:<dist>:<p0>:<p1>, got '" + std::string(s) + "'");
    }
    a3i::SyntheticColumnSpec spec;
    spec.name = parts[0];
    if      (parts[1] == "uniform")   spec.distribution = a3i::SyntheticDistribution::Uniform;
    else if (parts[1] == "normal")    spec.distribution = a3i::SyntheticDistribution::Normal;
    else if (parts[1] == "lognormal") spec.distribution = a3i::SyntheticDistribution::LogNormal;
    else throw std::runtime_error("unknown distribution '" + parts[1] +
                                  "' (use uniform|normal|lognormal)");
    spec.param0 = std::stod(parts[2]);
    spec.param1 = std::stod(parts[3]);
    return spec;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc <= 1 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_usage(std::cout);
        return 0;
    }

    a3i::GeneratorConfig config;
    std::string output, seed_str, rows_str;
    bool overwrite = false;

    int i = 1;
    while (i < argc) {
        std::string val;
        if (eat_kv(i, argc, argv, "--output", val)) { output = val; continue; }
        if (eat_kv(i, argc, argv, "--seed",   val)) { seed_str = val; continue; }
        if (eat_kv(i, argc, argv, "--rows",   val)) { rows_str = val; continue; }
        if (eat_kv(i, argc, argv, "--column", val)) { config.columns.push_back(parse_column(val)); continue; }
        if (eat_flag(i, argc, argv, "--overwrite")) { overwrite = true; continue; }
        throw std::runtime_error("unrecognized argument: " + std::string(argv[i]));
    }

    if (output.empty())   throw std::runtime_error("--output is required");
    if (seed_str.empty()) throw std::runtime_error("--seed is required");
    if (rows_str.empty()) throw std::runtime_error("--rows is required");
    config.seed = std::stoull(seed_str);
    config.rows = std::stoull(rows_str);

    const auto rep = a3i::generate_parquet(config, output, overwrite);
    std::cout << "wrote " << rep.output_path << '\n'
              << "  rows=" << rep.rows
              << "  fingerprint=" << rep.fingerprint << '\n';
    return 0;
}
catch (const std::exception& e) {
    std::cerr << "generate_dataset: " << e.what() << '\n';
    return 1;
}
