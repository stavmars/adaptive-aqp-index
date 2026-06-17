// Tests for the in-memory index table and the dataset schema.

#include <array>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "a3i/core/geometry.hpp"
#include "a3i/core/schema.hpp"
#include "a3i/core/types.hpp"
#include "a3i/storage/index_table.hpp"

namespace {

using a3i::DatasetSchema;
using a3i::DimensionId;
using a3i::HyperRect;
using a3i::IndexPos;
using a3i::IndexTable;
using a3i::MeasureId;
using a3i::Range;
using a3i::RowId;

// Two dimensions, five points; rows are intentionally in non-trivial order so
// swaps and accessors are tested against a recognizable pattern.
std::vector<std::vector<double>> sample_columns_2d() {
    return {
        {10.0, 11.0, 12.0, 13.0, 14.0},  // dim 0
        {20.0, 21.0, 22.0, 23.0, 24.0},  // dim 1
    };
}

TEST(IndexTable, FromColumnsReportsSizeAndDimensions) {
    const auto columns = sample_columns_2d();
    const auto table = IndexTable::from_columns(columns);

    EXPECT_EQ(table.size(), 5u);
    EXPECT_EQ(table.dimensions(), 2u);
}

TEST(IndexTable, FromColumnsAssignsContiguousRowIds) {
    const auto columns = sample_columns_2d();
    const auto table = IndexTable::from_columns(columns);

    for (IndexPos pos = 0; pos < table.size(); ++pos) {
        EXPECT_EQ(table.row_id(pos), static_cast<RowId>(pos));
    }
}

TEST(IndexTable, DimAccessorReturnsInterleavedValues) {
    const auto columns = sample_columns_2d();
    const auto table = IndexTable::from_columns(columns);

    for (IndexPos pos = 0; pos < table.size(); ++pos) {
        EXPECT_DOUBLE_EQ(table.dim(pos, 0), columns[0][pos]);
        EXPECT_DOUBLE_EQ(table.dim(pos, 1), columns[1][pos]);
    }
}

TEST(IndexTable, PointReturnsContiguousDimensionBlock) {
    const auto columns = sample_columns_2d();
    const auto table = IndexTable::from_columns(columns);

    for (IndexPos pos = 0; pos < table.size(); ++pos) {
        const auto p = table.point(pos);
        ASSERT_EQ(p.size(), 2u);
        EXPECT_DOUBLE_EQ(p[0], columns[0][pos]);
        EXPECT_DOUBLE_EQ(p[1], columns[1][pos]);
    }
}

TEST(IndexTable, FromDimensionReaderMatchesFromColumns) {
    // Streaming the columns in blocks must build the same interleaved table.
    const auto columns = sample_columns_2d();
    const auto via_columns = IndexTable::from_columns(columns);
    const auto via_reader = IndexTable::from_dimension_reader(
        static_cast<DimensionId>(columns.size()), columns[0].size(),
        [&](DimensionId axis, std::size_t off, std::size_t count,
            std::span<double> out) {
            for (std::size_t i = 0; i < count; ++i) out[i] = columns[axis][off + i];
        });

    ASSERT_EQ(via_reader.size(), via_columns.size());
    ASSERT_EQ(via_reader.dimensions(), via_columns.dimensions());
    for (IndexPos pos = 0; pos < via_reader.size(); ++pos) {
        EXPECT_EQ(via_reader.row_id(pos), via_columns.row_id(pos));
        for (DimensionId a = 0; a < via_reader.dimensions(); ++a) {
            EXPECT_DOUBLE_EQ(via_reader.dim(pos, a), via_columns.dim(pos, a));
        }
    }
}

TEST(IndexTable, FromDimensionReaderStreamsAcrossMultipleChunks) {
    // A small chunk size forces several blocks, including a partial last one;
    // the streamed result must still match the single-shot interleave.
    const auto columns = sample_columns_2d();
    const auto expected = IndexTable::from_columns(columns);
    const auto via_chunks = IndexTable::from_dimension_reader(
        static_cast<DimensionId>(columns.size()), columns[0].size(),
        [&](DimensionId axis, std::size_t off, std::size_t count,
            std::span<double> out) {
            for (std::size_t i = 0; i < count; ++i) out[i] = columns[axis][off + i];
        },
        /*chunk_rows=*/2);  // 5 rows -> blocks of 2, 2, 1

    ASSERT_EQ(via_chunks.size(), expected.size());
    for (IndexPos pos = 0; pos < via_chunks.size(); ++pos) {
        EXPECT_EQ(via_chunks.row_id(pos), expected.row_id(pos));
        for (DimensionId a = 0; a < via_chunks.dimensions(); ++a) {
            EXPECT_DOUBLE_EQ(via_chunks.dim(pos, a), expected.dim(pos, a));
        }
    }
}

TEST(IndexTable, FromDimensionReaderRejectsBadArguments) {
    const auto noop = [](DimensionId, std::size_t, std::size_t, std::span<double>) {};
    EXPECT_THROW(IndexTable::from_dimension_reader(0, 5, noop),
                 std::invalid_argument);  // zero dimensions
    EXPECT_THROW(IndexTable::from_dimension_reader(2, 5, noop, /*chunk_rows=*/0),
                 std::invalid_argument);  // zero chunk size
}

TEST(IndexTable, SwapPositionsMovesPointBlockAndRowIdTogether) {
    // I13: every permutation must keep dimensions and row_id aligned.
    const auto columns = sample_columns_2d();
    auto table = IndexTable::from_columns(columns);

    const auto p0_before = std::array<double, 2>{table.dim(0, 0), table.dim(0, 1)};
    const auto p3_before = std::array<double, 2>{table.dim(3, 0), table.dim(3, 1)};
    const RowId id0_before = table.row_id(0);
    const RowId id3_before = table.row_id(3);

    table.swap_positions(0, 3);

    EXPECT_DOUBLE_EQ(table.dim(0, 0), p3_before[0]);
    EXPECT_DOUBLE_EQ(table.dim(0, 1), p3_before[1]);
    EXPECT_DOUBLE_EQ(table.dim(3, 0), p0_before[0]);
    EXPECT_DOUBLE_EQ(table.dim(3, 1), p0_before[1]);
    EXPECT_EQ(table.row_id(0), id3_before);
    EXPECT_EQ(table.row_id(3), id0_before);
}

TEST(IndexTable, SwapPositionsWithSelfIsANoOp) {
    const auto columns = sample_columns_2d();
    auto table = IndexTable::from_columns(columns);

    const auto p2_before = std::array<double, 2>{table.dim(2, 0), table.dim(2, 1)};
    const RowId id2_before = table.row_id(2);

    table.swap_positions(2, 2);

    EXPECT_DOUBLE_EQ(table.dim(2, 0), p2_before[0]);
    EXPECT_DOUBLE_EQ(table.dim(2, 1), p2_before[1]);
    EXPECT_EQ(table.row_id(2), id2_before);
}

TEST(IndexTable, MutableSpansHaveExpectedSizes) {
    const auto columns = sample_columns_2d();
    auto table = IndexTable::from_columns(columns);

    EXPECT_EQ(table.points().size(),
              static_cast<std::size_t>(table.size()) * table.dimensions());
    EXPECT_EQ(table.row_ids().size(), table.size());
}

TEST(IndexTable, MutableSpansAllowInPlaceEdits) {
    const auto columns = sample_columns_2d();
    auto table = IndexTable::from_columns(columns);

    // Overwrite via the mutable interleaved span: point 1 -> (99, 100).
    auto points = table.points();
    points[1 * 2 + 0] = 99.0;
    points[1 * 2 + 1] = 100.0;

    auto ids = table.row_ids();
    ids[1] = 42;

    EXPECT_DOUBLE_EQ(table.dim(1, 0), 99.0);
    EXPECT_DOUBLE_EQ(table.dim(1, 1), 100.0);
    EXPECT_EQ(table.row_id(1), 42u);
}

TEST(IndexTable, FromColumnsRejectsEmptyColumnSet) {
    const std::vector<std::vector<double>> none;
    EXPECT_THROW(IndexTable::from_columns(none), std::invalid_argument);
}

TEST(IndexTable, FromColumnsRejectsMismatchedColumnLengths) {
    const std::vector<std::vector<double>> ragged = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0},
    };
    EXPECT_THROW(IndexTable::from_columns(ragged), std::invalid_argument);
}

TEST(IndexTable, FromColumnsAcceptsEmptyDataset) {
    const std::vector<std::vector<double>> empty_cols = {
        {}, {}, {}
    };
    const auto table = IndexTable::from_columns(empty_cols);
    EXPECT_EQ(table.size(), 0u);
    EXPECT_EQ(table.dimensions(), 3u);
    EXPECT_TRUE(table.points().empty());
    EXPECT_TRUE(table.row_ids().empty());
}

TEST(IndexTable, RawConstructorChecksConsistency) {
    // Consistent: 6 doubles for 3 points x 2 dims, 3 row_ids.
    EXPECT_NO_THROW(IndexTable(std::vector<double>{1, 2, 3, 4, 5, 6},
                               std::vector<RowId>{0, 1, 2},
                               /*dimensions=*/2));

    // points length not divisible by dimensions.
    EXPECT_THROW(IndexTable(std::vector<double>{1, 2, 3, 4, 5},
                            std::vector<RowId>{0, 1},
                            /*dimensions=*/2),
                 std::invalid_argument);

    // row_ids count disagrees with implied point count.
    EXPECT_THROW(IndexTable(std::vector<double>{1, 2, 3, 4},
                            std::vector<RowId>{0, 1, 2},
                            /*dimensions=*/2),
                 std::invalid_argument);

    // zero dimensions is rejected.
    EXPECT_THROW(IndexTable(std::vector<double>{},
                            std::vector<RowId>{},
                            /*dimensions=*/0),
                 std::invalid_argument);
}

TEST(DatasetSchema, DefaultIsEmpty) {
    DatasetSchema schema;
    EXPECT_TRUE(schema.dimension_names.empty());
    EXPECT_TRUE(schema.measure_names.empty());
    EXPECT_TRUE(schema.domain_bounds.dims.empty());
    EXPECT_EQ(schema.object_count, 0u);
    EXPECT_TRUE(schema.binary_manifest_path.empty());
}

TEST(DatasetSchema, HoldsCallerSuppliedFields) {
    DatasetSchema schema;
    schema.dimension_names = {"x", "y"};
    schema.measure_names   = {"fare", "distance"};
    schema.domain_bounds   = HyperRect{{Range{0.0, 1.0}, Range{-5.0, 5.0}}};
    schema.object_count    = 1000;
    schema.binary_manifest_path = "/tmp/manifest.json";

    EXPECT_EQ(schema.dimension_names.size(), 2u);
    EXPECT_EQ(schema.measure_names.size(), 2u);
    // MeasureId / DimensionId are the dense positions in these vectors.
    EXPECT_EQ(static_cast<MeasureId>(schema.measure_names.size() - 1), MeasureId{1});
    EXPECT_EQ(static_cast<DimensionId>(schema.dimension_names.size() - 1), DimensionId{1});
    EXPECT_EQ(schema.domain_bounds.dims.size(), 2u);
    EXPECT_DOUBLE_EQ(schema.domain_bounds.dims[1].low, -5.0);
    EXPECT_EQ(schema.object_count, 1000u);
    EXPECT_EQ(schema.binary_manifest_path, "/tmp/manifest.json");
}

TEST(Types, IdWidthsAreFixed) {
    // Id widths are fixed by the storage layout; lock them down so a typo
    // on one platform does not silently corrupt on-disk files.
    static_assert(sizeof(a3i::RowId) == 4, "RowId must be 32-bit");
    static_assert(sizeof(a3i::IndexPos) == 4, "IndexPos must be 32-bit");
    static_assert(sizeof(a3i::PartitionId) == 4, "PartitionId must be 32-bit");
    static_assert(sizeof(a3i::MeasureId) == 2, "MeasureId must be 16-bit");
    static_assert(sizeof(a3i::DimensionId) == 2, "DimensionId must be 16-bit");
    SUCCEED();
}

}  // namespace
