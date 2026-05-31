// Deterministic query-workload generation.
//
// A workload is a sequence of axis-aligned rectangles over a dataset's
// dimension domain, materialised as a CSV (one rectangle per row) plus a JSON
// sidecar. The same configuration always reproduces the same file: the first
// CSV line carries a fingerprint of the generator inputs, so a later call with
// unchanged inputs reuses the file and any change regenerates it in place.
//
// Two families are supported. `uniform_random` draws independent centres over
// the whole domain; `clustered` draws centres from a Gaussian around an
// explicit focus. Each rectangle is sized to a target selectivity, either by
// calibrating against a cached spatial sample of the data (`calibrated`) or by
// a closed-form area split for uniform data (`closed_form`). The accuracy
// target is deliberately NOT part of a workload row — the error bound and
// confidence are fixed per experiment and applied by the runner, so one
// workload file drives exact and approximate runs alike.

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "a3i/core/geometry.hpp"

namespace a3i {

enum class WorkloadFamily { UniformRandom, Clustered };

/// How a rectangle is sized to the target selectivity.
///   Calibrated : resize against a cached spatial sample until the fraction of
///                sample points inside is within tolerance of the target.
///   ClosedForm : take an axis-aligned share of each dimension's extent
///                (`selectivity^(1/d)`); correct only for uniform data.
enum class ExtentMode { Calibrated, ClosedForm };

struct WorkloadConfig {
    std::string    dataset_id;             ///< Prepared dataset this targets.
    std::string    name;                   ///< Workload id; the output file stem.
    WorkloadFamily family      = WorkloadFamily::UniformRandom;
    ExtentMode     extent_mode = ExtentMode::Calibrated;
    double         selectivity = 0.01;     ///< Target fraction of rows in (0, 1].
    std::vector<double> focus;             ///< Centre of the cluster (Clustered only),
                                           ///< one coordinate per dimension.
    std::uint64_t  seed  = 0;              ///< Seed for centre draws.
    std::size_t    count = 500;            ///< Number of queries.

    // Spatial sample used for Calibrated sizing (ignored for ClosedForm).
    std::size_t    reservoir_rows = 65536;
    std::uint64_t  reservoir_seed = 0;
};

struct WorkloadReport {
    std::filesystem::path csv_path;
    std::filesystem::path metadata_path;
    std::filesystem::path reservoir_path;   ///< Empty when ClosedForm.
    std::uint64_t         fingerprint = 0;
    std::size_t           count = 0;
    bool                  regenerated = false;       ///< False if an up-to-date file was reused.
    double                achieved_selectivity_mean = 0.0;  ///< Diagnostic (Calibrated only).
};

/// 64-bit FNV-1a digest of the canonical generator inputs. The domain bounds
/// are folded in directly: a change to the dataset's extent (which moves every
/// rectangle) regenerates the workload even under an unchanged dataset id.
/// Exposed so the determinism contract is unit-testable without file I/O.
std::uint64_t workload_fingerprint(const WorkloadConfig& config,
                                   const HyperRect& domain);

/// Generate (or reuse) the workload for `config` from a prepared dataset's
/// manifest, writing `<out_dir>/<name>.csv` and `<out_dir>/<name>.metadata.json`.
/// For Calibrated sizing a spatial sample is built once and cached under
/// `<out_dir>/reservoirs/<dataset_id>.bin` (rebuilt only if its recorded row
/// count or seed differs). If a CSV already exists whose first-line fingerprint
/// matches the current inputs it is reused unless `force` is set.
WorkloadReport generate_workload(const WorkloadConfig& config,
                                 const std::filesystem::path& manifest_path,
                                 const std::filesystem::path& out_dir,
                                 bool force = false);

}  // namespace a3i
