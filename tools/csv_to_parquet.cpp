// Offline CSV/TSV -> Parquet converter (one-off per dataset).
//
// Faithfully converts a delimited text file to a typed, columnar Parquet
// artifact with no projection and no row filtering. Headered inputs keep
// their column names; headerless inputs get synthesized positional names.
// See --help for usage.

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "a3i/tools/csv_to_parquet.hpp"

namespace {

void print_usage(std::ostream& os) {
    os <<
"Usage: csv_to_parquet --input <csv> --output <parquet>\n"
"                      [--has-header] [--delimiter <ch>] [--null-string <s>]\n"
"                      [--timestamp-format <fmt>]... [--overwrite]\n"
"\n"
"Converts a delimited text file to a typed Parquet file. Column types\n"
"are inferred (integers, doubles, booleans, timestamps, strings). No filtering\n"
"or projection happens here -- the output is the immutable, unfiltered dataset.\n"
"\n"
"With --has-header the first row supplies column names. Without it, names are\n"
"synthesized as 'col' + zero-padded index (col00..col17 for\n"
"an 18-column file; col0..col9 for 10). To give a headerless source\n"
"meaningful names, prepend a header line to the file and pass --has-header.\n"
"\n"
"--delimiter accepts a single char, or 'tab'/'\\t' for TSV.\n"
"--null-string marks an extra token as null (empty fields are always null).\n"
"--timestamp-format adds a strptime pattern for timestamp inference; repeat it\n"
"  to cover a column whose rows mix formats (ISO8601 is always tried too).\n"
"\n"
"Example:\n"
"  csv_to_parquet --input /data/raw/taxi.csv --output /data/parquet/taxi.parquet \\\n"
"     --has-header --delimiter , --timestamp-format '%m/%d/%Y %I:%M:%S %p'\n";
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

}  // namespace

int main(int argc, char** argv) try {
    if (argc <= 1 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_usage(std::cout);
        return 0;
    }

    a3i::CsvToParquetOptions opts;
    std::string input, output, delimiter_str = ",";

    int i = 1;
    while (i < argc) {
        std::string val;
        if (eat_kv(i, argc, argv, "--input",       val)) { input = val; continue; }
        if (eat_kv(i, argc, argv, "--output",      val)) { output = val; continue; }
        if (eat_kv(i, argc, argv, "--delimiter",   val)) { delimiter_str = val; continue; }
        if (eat_kv(i, argc, argv, "--null-string", val)) { opts.null_string = val; continue; }
        if (eat_kv(i, argc, argv, "--timestamp-format", val)) { opts.timestamp_formats.push_back(val); continue; }
        if (eat_flag(i, argc, argv, "--has-header")) { opts.has_header = true; continue; }
        if (eat_flag(i, argc, argv, "--overwrite"))  { opts.overwrite  = true; continue; }
        throw std::runtime_error("unrecognized argument: " + std::string(argv[i]));
    }

    if (input.empty())  throw std::runtime_error("--input is required");
    if (output.empty()) throw std::runtime_error("--output is required");

    char delim = ',';
    if (delimiter_str == "\\t" || delimiter_str == "tab") delim = '\t';
    else if (delimiter_str.size() == 1)                   delim = delimiter_str[0];
    else throw std::runtime_error("--delimiter must be a single char (use 'tab' or '\\t' for TSV)");

    opts.input_path  = input;
    opts.output_path = output;
    opts.delimiter   = delim;

    const auto rep = a3i::csv_to_parquet(opts);
    std::cout << "wrote " << output << '\n'
              << "  rows=" << rep.rows
              << "  columns=" << rep.column_names.size() << '\n';
    return 0;
}
catch (const std::exception& e) {
    std::cerr << "csv_to_parquet: " << e.what() << '\n';
    return 1;
}
