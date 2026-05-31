#include "a3i/storage/manifest.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace a3i {

namespace {

using nlohmann::json;

json to_json(const GlobalMeasureStats& g) {
    return json{
        {"non_nan_count", g.non_nan_count},
        {"nan_count", g.nan_count},
        {"sum", g.sum},
        {"sum_sq", g.sum_sq},
        {"min", g.min},
        {"max", g.max},
    };
}

GlobalMeasureStats from_json_stats(const json& j) {
    GlobalMeasureStats g;
    g.non_nan_count = j.at("non_nan_count").get<std::uint64_t>();
    g.nan_count     = j.at("nan_count").get<std::uint64_t>();
    g.sum           = j.at("sum").get<double>();
    g.sum_sq        = j.at("sum_sq").get<double>();
    g.min           = j.at("min").get<double>();
    g.max           = j.at("max").get<double>();
    return g;
}

}  // namespace

void write_manifest(const std::filesystem::path& manifest_path, const Manifest& m) {
    json doc;
    doc["dataset_id"] = m.dataset_id;
    doc["row_count"]  = m.row_count;
    doc["endianness"] = m.endianness;
    doc["dtype"]      = m.dtype;

    json dims = json::array();
    for (const auto& d : m.dimensions) {
        dims.push_back({
            {"logical_id",   d.logical_id},
            {"name",         d.name},
            {"source_index", d.source_index},
            {"file",         d.file},
            {"min",          d.min},
            {"max",          d.max},
        });
    }
    doc["dimensions"] = std::move(dims);

    json measures = json::array();
    for (const auto& mz : m.measures) {
        measures.push_back({
            {"logical_id",   mz.logical_id},
            {"name",         mz.name},
            {"source_index", mz.source_index},
            {"file",         mz.file},
            {"global",       to_json(mz.global)},
        });
    }
    doc["measures"] = std::move(measures);

    json low = json::array();
    json high = json::array();
    for (const auto& r : m.domain_bounds.dims) { low.push_back(r.low); high.push_back(r.high); }
    doc["domain_bounds"] = { {"low", low}, {"high", high} };

    doc["null_encoding"]     = m.null_encoding;
    doc["applied_drop_if"]   = m.applied_drop_if;
    doc["source_parquet"]    = m.source_parquet;
    doc["source_bytes"]      = m.source_bytes;
    doc["source_mtime"]      = m.source_mtime;
    if (m.parent_dataset_id) doc["parent_dataset_id"] = *m.parent_dataset_id;
    if (m.max_rows)          doc["max_rows"]          = *m.max_rows;
    doc["converter_version"] = m.converter_version;
    doc["created_utc"]       = m.created_utc;

    std::filesystem::create_directories(manifest_path.parent_path());
    std::ofstream out(manifest_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot open manifest for writing: " + manifest_path.string());
    }
    out << std::setw(2) << doc << '\n';
}

Manifest read_manifest(const std::filesystem::path& manifest_path) {
    std::ifstream in(manifest_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open manifest: " + manifest_path.string());
    }
    json doc;
    try {
        in >> doc;
    } catch (const std::exception& e) {
        throw std::runtime_error("manifest is not valid JSON (" + manifest_path.string() +
                                 "): " + e.what());
    }

    Manifest m;
    try {
        m.dataset_id = doc.at("dataset_id").get<std::string>();
        m.row_count  = doc.at("row_count").get<std::uint64_t>();
        m.endianness = doc.value("endianness", std::string{"little"});
        m.dtype      = doc.value("dtype", std::string{"float64"});

        for (const auto& d : doc.at("dimensions")) {
            DimensionDescriptor dd;
            dd.logical_id   = d.at("logical_id").get<std::uint16_t>();
            dd.name         = d.at("name").get<std::string>();
            dd.source_index = d.at("source_index").get<std::uint32_t>();
            dd.file         = d.at("file").get<std::string>();
            dd.min          = d.at("min").get<double>();
            dd.max          = d.at("max").get<double>();
            m.dimensions.push_back(std::move(dd));
        }

        for (const auto& mz : doc.at("measures")) {
            MeasureDescriptor md;
            md.logical_id   = mz.at("logical_id").get<std::uint16_t>();
            md.name         = mz.at("name").get<std::string>();
            md.source_index = mz.at("source_index").get<std::uint32_t>();
            md.file         = mz.at("file").get<std::string>();
            md.global       = from_json_stats(mz.at("global"));
            m.measures.push_back(std::move(md));
        }

        const auto& db = doc.at("domain_bounds");
        const auto& los = db.at("low");
        const auto& his = db.at("high");
        if (los.size() != his.size()) {
            throw std::runtime_error("domain_bounds low/high size mismatch");
        }
        m.domain_bounds.dims.reserve(los.size());
        for (std::size_t i = 0; i < los.size(); ++i) {
            m.domain_bounds.dims.push_back(Range{los[i].get<double>(), his[i].get<double>()});
        }

        m.null_encoding     = doc.value("null_encoding", std::string{"NaN"});
        if (doc.contains("applied_drop_if") && doc["applied_drop_if"].is_array()) {
            m.applied_drop_if = doc["applied_drop_if"].get<std::vector<std::string>>();
        }
        m.source_parquet    = doc.value("source_parquet", std::string{});
        m.source_bytes      = doc.value("source_bytes", std::uint64_t{0});
        m.source_mtime      = doc.value("source_mtime", std::int64_t{0});
        if (doc.contains("parent_dataset_id") && !doc["parent_dataset_id"].is_null()) {
            m.parent_dataset_id = doc["parent_dataset_id"].get<std::string>();
        }
        if (doc.contains("max_rows") && !doc["max_rows"].is_null()) {
            m.max_rows = doc["max_rows"].get<std::uint64_t>();
        }
        m.converter_version = doc.value("converter_version", std::string{});
        m.created_utc       = doc.value("created_utc", std::string{});
    } catch (const std::exception& e) {
        throw std::runtime_error("manifest schema error (" + manifest_path.string() +
                                 "): " + e.what());
    }

    // Cross-field consistency.
    if (m.dimensions.size() != m.domain_bounds.dims.size()) {
        throw std::runtime_error("manifest: domain_bounds size != dimensions count");
    }
    for (std::uint16_t i = 0; i < m.dimensions.size(); ++i) {
        if (m.dimensions[i].logical_id != i) {
            throw std::runtime_error("manifest: dimension logical_id is not its position");
        }
    }
    for (std::uint16_t i = 0; i < m.measures.size(); ++i) {
        if (m.measures[i].logical_id != i) {
            throw std::runtime_error("manifest: measure logical_id is not its position");
        }
    }

    return m;
}

}  // namespace a3i
