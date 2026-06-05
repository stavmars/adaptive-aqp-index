// Substrate-independence harness. The query engine and every cross-cutting
// invariant must hold for any registered substrate, so this test iterates
// the factory's registry rather than naming a substrate. As more
// substrates register, they automatically join this battery; for now it
// establishes the shape the later differential checks plug into.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "a3i/access_path/substrate_factory.hpp"
#include "a3i/core/geometry.hpp"
#include "a3i/storage/index_table.hpp"

namespace {

using namespace a3i;

IndexTable make_table() {
    std::vector<double> xs = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> ys = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    return IndexTable::from_columns({xs, ys});
}

SubstrateConfig config() {
    SubstrateConfig cfg;
    return cfg;
}

void check_partition_invariants(const AdaptiveAccessPath& path, std::size_t n) {
    auto active = path.active_partitions();
    ASSERT_FALSE(active.empty());
    std::vector<bool> covered(n, false);
    for (PartitionId id : active) {
        PartitionView pv = path.partition(id);
        EXPECT_TRUE(pv.active);
        EXPECT_LE(pv.begin, pv.end);
        EXPECT_LE(pv.end, static_cast<IndexPos>(n));
        for (IndexPos p = pv.begin; p < pv.end; ++p) {
            ASSERT_FALSE(covered[p]);
            covered[p] = true;
        }
    }
    for (std::size_t p = 0; p < n; ++p) EXPECT_TRUE(covered[p]);
}

TEST(SubstrateIndependence, RegistryIsNonEmpty) {
    auto ids = SubstrateFactory::instance().registered_ids();
    EXPECT_FALSE(ids.empty());
    EXPECT_TRUE(SubstrateFactory::instance().is_registered("adaptive_kd"));
}

TEST(SubstrateIndependence, EverySubstrateUpholdsInvariants) {
    auto& factory = SubstrateFactory::instance();
    for (const std::string& id : factory.registered_ids()) {
        SCOPED_TRACE(id);
        IndexTable table = make_table();
        auto path = factory.create(id, config());
        ASSERT_NE(path, nullptr);
        path->prepare(table);
        path->ensure_built();

        check_partition_invariants(*path, table.size());

        // A whole-domain query must classify every active partition as
        // fully contained and leave nothing partial.
        std::size_t contained = 0;
        std::size_t partial   = 0;
        const HyperRect whole{{{-1.0, 11.0}, {-1.0, 11.0}}};
        for (PartitionId pid : path->active_partitions()) {
            switch (path->classify(pid, whole)) {
                case Containment::Contained: ++contained; break;
                case Containment::Partial:   ++partial;   break;
                case Containment::Disjoint:               break;
            }
        }
        EXPECT_EQ(contained, path->active_partitions().size());
        EXPECT_EQ(partial, 0u);

        // A disjoint query touches nothing.
        const HyperRect far{{{20.0, 30.0}, {20.0, 30.0}}};
        for (PartitionId pid : path->active_partitions()) {
            EXPECT_EQ(path->classify(pid, far), Containment::Disjoint);
        }
    }
}

TEST(SubstrateIndependence, UnknownIdThrows) {
    EXPECT_THROW(SubstrateFactory::instance().create("no_such_substrate", config()),
                 std::invalid_argument);
}

}  // namespace
