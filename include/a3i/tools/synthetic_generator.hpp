// Seeded synthetic dataset generator.
//
// Produces a Parquet artifact from a (seed, rows, column-spec)
// configuration. Generation is fully deterministic: the same configuration
// yields a byte-identical Parquet file, so a generated dataset is
// reproducible if deleted and is shared identically across systems by
// copying the one file (never by re-deriving randomness elsewhere).
//
// Used only offline, in data preparation; the query path never generates
// data.

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace a3i {

/// Per-column value distribution.
///   Uniform   : values in [param0, param1).
///   Normal    : mean = param0, stddev = param1.
///   LogNormal : underlying normal mean = param0, stddev = param1.
enum class SyntheticDistribution { Uniform, Normal, LogNormal };

struct SyntheticColumnSpec {
    std::string           name;
    SyntheticDistribution distribution = SyntheticDistribution::Uniform;
    double                param0 = 0.0;
    double                param1 = 1.0;
};

struct GeneratorConfig {
    std::uint64_t                    seed = 0;
    std::uint64_t                    rows = 0;
    std::vector<SyntheticColumnSpec> columns;
};

struct GeneratorReport {
    std::filesystem::path output_path;
    std::uint64_t         rows = 0;
    std::uint64_t         fingerprint = 0;  ///< Stable digest of the written file's bytes.
};

/// FNV-1a 64-bit digest over a file's raw bytes. Used to assert that two
/// generations of the same configuration are byte-identical.
std::uint64_t file_fingerprint(const std::filesystem::path& path);

/// Generate the Parquet described by `config` at `output_path`.
/// Deterministic in (seed, rows, columns). Throws std::runtime_error on I/O
/// errors, std::invalid_argument on malformed options. Refuses to overwrite
/// an existing file unless `overwrite` is set.
GeneratorReport generate_parquet(const GeneratorConfig& config,
                                 const std::filesystem::path& output_path,
                                 bool overwrite = false);

}  // namespace a3i
