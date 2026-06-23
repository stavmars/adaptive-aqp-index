// Grid substrate: a coarse equi-width grid with an adaptive KD-tree per tile.
//
// ensure_built() partitions the table once into partitions_per_dimension^d
// equi-width tiles by a counting sort over the tile each point falls in, then
// installs them as the children of a single root. Each tile is the root of a
// local adaptive KD-tree: a partial tile cracks toward the query during
// evaluation, exactly as the adaptive substrate cracks its root.
//
// Tiles are fixed at build time and addressed arithmetically, so the descent is
// routed only to the tiles overlapping a query (overlapping_children) and every
// tile's summary can be precomputed at initialization
// (has_prebuilt_partitions()). The counting sort permutes rows, so the read
// path sorts per stratum.

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "a3i/access_path/adaptive_access_path.hpp"
#include "a3i/access_path/substrate_factory.hpp"
#include "a3i/core/geometry.hpp"
#include "a3i/core/types.hpp"
#include "a3i/storage/index_table.hpp"
#include "a3i/substrates/kd_tree.hpp"

namespace a3i {

class BinaryColumnStore;

class GridAkdAccessPath : public AdaptiveAccessPath {
public:
    explicit GridAkdAccessPath(SubstrateConfig config);

    void prepare(IndexTable& table) override;
    void ensure_built() override;
    std::vector<PartitionId> roots() const override;
    std::vector<PartitionId> children(PartitionId id) const override;
    std::vector<PartitionId> overlapping_children(PartitionId id,
                                                  const HyperRect& q) const override;
    bool is_leaf(PartitionId id) const override;
    Containment classify(PartitionId id, const HyperRect& q) const override;
    std::vector<PartitionId> refine(PartitionId id, const HyperRect& q,
                                    IndexTable& table) override;
    PartitionView partition(PartitionId id) const override;
    std::vector<PartitionId> active_partitions() const override;
    std::optional<PartitionId> parent(PartitionId id) const override;
    bool supports_refine() const override { return true; }
    bool has_prebuilt_partitions() const override { return true; }
    const std::vector<PartitionId>* row_owner_map() const override {
        return built_ ? &owner_ : nullptr;
    }
    bool builds_table_from_dimension_store() const override { return true; }
    void set_dimension_store(BinaryColumnStore* store) override { store_ = store; }

private:
    // Tile index of value `v` on `axis`, clamped to [0, k_-1].
    std::uint32_t cell_index(DimensionId axis, double v) const;
    // Flatten/decode per-axis tile indices to/from the row-major flat tile id.
    std::uint32_t flatten(const std::vector<std::uint32_t>& idx) const;
    HyperRect tile_bounds(std::uint32_t flat_cell) const;
    // Tile the prepared resident table and reorder it in place.
    void build_from_table();
    // Tile from the store's resident dimension columns, release them, and
    // scatter the table from storage so only the final order is materialized.
    void build_from_dimension_store();
    // Install the root with one child per tile from the prefix-sum offsets and
    // record the RowId -> tile owner map (shared by both build paths).
    void install_tiles(std::size_t n, const std::vector<IndexPos>& offsets,
                       const std::vector<std::uint32_t>& cell);

    SubstrateConfig config_;
    IndexTable*     table_ = nullptr;  ///< Non-owning; prepared at load time.
    BinaryColumnStore* store_ = nullptr;  ///< Set only for the out-of-core build.
    bool            built_ = false;
    KdTree          tree_;             ///< Node 0 is the root; nodes 1..G are tiles.

    DimensionId           dims_ = 0;
    std::uint32_t         k_    = 1;   ///< Tiles per axis.
    std::uint32_t         tiles_ = 1;  ///< G = k_^dims_.
    std::vector<double>   lo_;         ///< Per-axis domain low.
    std::vector<double>   width_;      ///< Per-axis tile width.
    std::vector<double>   inv_width_;  ///< 1/width_ (0 for a degenerate axis).
    HyperRect             domain_;     ///< Root bounds.
    std::vector<PartitionId> owner_;   ///< RowId -> tile PartitionId (1 + flat cell).
};

}  // namespace a3i
