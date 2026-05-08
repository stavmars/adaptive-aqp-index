// Offline CSV -> binary column converter.
//
// Run once per dataset (or once per --max-rows variant); never invoked
// from the runner. Writes columns/<...>.bin and manifest.json under the
// resolved output directory. See --help for usage and examples.

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "a3i/tools/csv_to_columns_pipeline.hpp"

namespace {

void print_usage(std::ostream& os) {
    os <<
"Usage: convert_csv_to_columns --input <csv> --dataset-id <id>\n"
"                              (--output-dir <dir> | --prepared-root <root>)\n"
"                              [--has-header] [--delimiter <ch>] [--null-string <s>]\n"
"                              --dimension <name>:<lo>:<hi>   (one or more, repeatable)\n"
"                              --measure   <name>             (one or more, repeatable)\n"
"                              [--validation-filter '<name><op><value>']  (repeatable)\n"
"                              [--max-rows <N>] [--overwrite]\n"
"\n"
"Resolves columns by name. With --has-header, names come from the header line.\n"
"Without --has-header, names are synthesized DuckDB-style: 'column' + zero-padded\n"
"index (e.g. column00..column17 for an 18-column file; column0..column9 for 10).\n"
"\n"
"Output directory precedence: --output-dir > --prepared-root/<dataset-id> > error.\n"
"Per-column file paths in the manifest are relative to the manifest's directory.\n"
"\n"
"Validation filters DROP rows matching the predicate, e.g. 'NUMBER_OBSERVERS<1.0'.\n"
"Operators: < <= > >= == !=  (single = and 0 != are not accepted).\n"
"\n"
"Examples (one-off per dataset):\n"
"\n"
"  # 1M-row prefix of synth10 (headerless; DuckDB-style names)\n"
"  convert_csv_to_columns --input /data/synth10_10M.csv \\\n"
"     --dataset-id synth10_1M --prepared-root /data/prepared --delimiter , \\\n"
"     --dimension column0:0:1000 --dimension column1:0:1000 \\\n"
"     --measure column2 --measure column3 --measure column4 --measure column5 \\\n"
"     --measure column6 --measure column7 --measure column8 --measure column9 \\\n"
"     --max-rows 1000000\n"
"\n"
"  # taxi (headered)\n"
"  convert_csv_to_columns --input /data/yellow_tripdata_2013_2014_cleaned.csv \\\n"
"     --dataset-id taxi --prepared-root /data/prepared --has-header --delimiter , \\\n"
"     --dimension pickup_lon:-74.106216:-73.842545 \\\n"
"     --dimension pickup_lat:40.676993:40.839788 \\\n"
"     --measure fare_amount --measure total_amount \\\n"
"     --measure trip_distance --measure passenger_count\n";
}

// --- Argument helpers ----------------------------------------------------

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

a3i::DimensionRequest parse_dimension(std::string_view s) {
    const auto parts = split(s, ':');
    if (parts.size() != 3) {
        throw std::runtime_error("--dimension expects <name>:<lo>:<hi>, got '" + std::string(s) + "'");
    }
    a3i::DimensionRequest d;
    d.name = parts[0];
    d.low  = std::stod(parts[1]);
    d.high = std::stod(parts[2]);
    if (!(d.low < d.high)) {
        throw std::runtime_error("--dimension lo must be < hi: '" + std::string(s) + "'");
    }
    return d;
}

a3i::ValidationFilter parse_filter(std::string_view s) {
    using Op = a3i::ValidationFilter::Op;
    static const std::vector<std::pair<std::string, Op>> ops = {
        {">=", Op::Ge}, {"<=", Op::Le}, {"==", Op::Eq}, {"!=", Op::Ne},
        {">",  Op::Gt}, {"<",  Op::Lt},
    };
    for (const auto& [tok, op] : ops) {
        const auto pos = s.find(tok);
        if (pos != std::string_view::npos) {
            a3i::ValidationFilter f;
            f.name  = std::string(s.substr(0, pos));
            f.op    = op;
            f.value = std::stod(std::string(s.substr(pos + tok.size())));
            if (f.name.empty()) throw std::runtime_error("--validation-filter missing name");
            return f;
        }
    }
    throw std::runtime_error("--validation-filter must use one of < <= > >= == !=");
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc <= 1 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_usage(std::cout);
        return 0;
    }

    a3i::ConvertOptions opts;
    std::string input_csv, output_dir, prepared_root, dataset_id;
    std::string delimiter_str = ",";
    std::string max_rows_str;

    int i = 1;
    while (i < argc) {
        std::string val;
        if (eat_kv(i, argc, argv, "--input",         val)) { input_csv = val; continue; }
        if (eat_kv(i, argc, argv, "--output-dir",    val)) { output_dir = val; continue; }
        if (eat_kv(i, argc, argv, "--prepared-root", val)) { prepared_root = val; continue; }
        if (eat_kv(i, argc, argv, "--dataset-id",    val)) { dataset_id = val; continue; }
        if (eat_kv(i, argc, argv, "--delimiter",     val)) { delimiter_str = val; continue; }
        if (eat_kv(i, argc, argv, "--null-string",   val)) { opts.null_string = val; continue; }
        if (eat_kv(i, argc, argv, "--max-rows",      val)) { max_rows_str = val; continue; }
        if (eat_kv(i, argc, argv, "--dimension",     val)) { opts.dimensions.push_back(parse_dimension(val)); continue; }
        if (eat_kv(i, argc, argv, "--measure",       val)) { opts.measures.push_back(val); continue; }
        if (eat_kv(i, argc, argv, "--validation-filter", val)) { opts.validation_filters.push_back(parse_filter(val)); continue; }
        if (eat_flag(i, argc, argv, "--has-header")) { opts.has_header = true; continue; }
        if (eat_flag(i, argc, argv, "--overwrite"))  { opts.overwrite  = true; continue; }
        throw std::runtime_error("unrecognized argument: " + std::string(argv[i]));
    }

    if (input_csv.empty())  throw std::runtime_error("--input is required");
    if (dataset_id.empty()) throw std::runtime_error("--dataset-id is required");
    if (output_dir.empty()) {
        if (prepared_root.empty()) {
            throw std::runtime_error("either --output-dir or --prepared-root is required");
        }
        output_dir = (std::filesystem::path(prepared_root) / dataset_id).string();
    }

    // --delimiter accepts: literal char, "\t", "tab".
    char delim = ',';
    if (delimiter_str == "\\t" || delimiter_str == "tab") delim = '\t';
    else if (delimiter_str.size() == 1)                    delim = delimiter_str[0];
    else throw std::runtime_error("--delimiter must be a single char (use 'tab' or '\\t' for TSV)");

    opts.input_csv  = input_csv;
    opts.output_dir = output_dir;
    opts.dataset_id = dataset_id;
    opts.delimiter  = delim;
    if (!max_rows_str.empty()) opts.max_rows = std::stoull(max_rows_str);

    const auto rep = a3i::run_csv_to_columns(opts);
    std::cout << "wrote " << rep.manifest_path << '\n'
              << "  rows_written=" << rep.rows_written
              << "  rows_read=" << rep.rows_read
              << "  rows_filtered_out=" << rep.rows_filtered_out << '\n';
    return 0;
}
catch (const std::exception& e) {
    std::cerr << "convert_csv_to_columns: " << e.what() << '\n';
    return 1;
}
