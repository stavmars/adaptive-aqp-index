// Description of a prepared dataset: dimension and measure columns, the
// row count, the domain bounds, and per-measure global statistics. The
// manifest is the engine's only handle on the binary columns the
// offline converter produced (no header lives in the .bin files
// themselves). Column file paths are relative to the manifest's own
// directory so a prepared directory is relocatable.

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "a3i/core/geometry.hpp"

namespace a3i {

/// Per-measure aggregate stats computed once during conversion. Used by
/// the engine as priors and floors; never recomputed at query time.
struct GlobalMeasureStats {
    std::uint64_t non_nan_count = 0;
    std::uint64_t nan_count = 0;
    double sum = 0.0;
    double sum_sq = 0.0;
    double min = 0.0;
    double max = 0.0;
};

/// One entry per dimension column.
struct DimensionDescriptor {
    std::uint16_t logical_id = 0;   ///< Dense index; also the DimensionId.
    std::string name;               ///< Source column name (header or synthesized positional fallback).
    std::uint32_t source_index = 0; ///< Original CSV column position (provenance only).
    std::string file;               ///< Path to the .bin, relative to the manifest dir.
    double min = 0.0;               ///< Observed min over non-NaN values.
    double max = 0.0;               ///< Observed max over non-NaN values.
};

/// One entry per measure column.
struct MeasureDescriptor {
    std::uint16_t logical_id = 0;   ///< Dense index; also the MeasureId.
    std::string name;
    std::uint32_t source_index = 0;
    std::string file;
    GlobalMeasureStats global;
};

/// On-disk dataset description.
struct Manifest {
    std::string dataset_id;
    std::uint64_t row_count = 0;
    std::string endianness = "little";
    std::string dtype = "float64";
    std::vector<DimensionDescriptor> dimensions;
    std::vector<MeasureDescriptor> measures;
    HyperRect domain_bounds;
    std::string null_encoding = "NaN";
    std::vector<std::string> applied_drop_if; ///< Drop predicates applied at conversion, as "name op value" strings.
    std::string source_parquet;        ///< Absolute path of the Parquet the converter consumed.
    std::uint64_t source_bytes = 0;    ///< Size of the source Parquet in bytes (cheap staleness check).
    std::int64_t source_mtime = 0;     ///< Source last-write time as Unix epoch seconds (cheap staleness check).
    std::optional<std::string> parent_dataset_id; ///< Set for a --max-rows subset: the full dataset it is a prefix of.
    std::optional<std::uint64_t> max_rows;        ///< Set when --max-rows capped the prepared row count.
    std::string converter_version;
    std::string created_utc;           ///< ISO-8601, e.g. "2026-05-28T12:34:56Z".
};

/// Read a manifest JSON file. Throws std::runtime_error on I/O or schema
/// errors. The path is stored verbatim by the caller; per-column `file`
/// fields stay relative.
Manifest read_manifest(const std::filesystem::path& manifest_path);

/// Write the manifest as pretty JSON. Overwrites if it exists.
void write_manifest(const std::filesystem::path& manifest_path, const Manifest& manifest);

}  // namespace a3i
