#include "a3i/engine/query_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "a3i/aqp/decompose.hpp"
#include "a3i/aqp/eager_materialize.hpp"
#include "a3i/aqp/stratum_cursor.hpp"
#include "a3i/storage/manifest.hpp"
#include "a3i/util/rng.hpp"

namespace a3i {

namespace {

double max_relative_half_width(const std::vector<AggregateEstimate>& est) {
    double worst = 0.0;
    for (const AggregateEstimate& e : est) {
        if (e.exact) continue;
        worst = std::max(worst, e.relative_half_width);
    }
    return worst;
}

}  // namespace

QueryEngine::QueryEngine(const BinaryColumnStore& store, IndexTable& table,
                         AdaptiveAccessPath& access_path, EngineConfig config)
    : store_(store),
      table_(table),
      access_path_(access_path),
      config_(config),
      measure_count_(store.measure_count()),
      estimator_(),
      allocator_(config.allocator) {
    // The largest absolute per-measure global mean scales the magnitude
    // floors of the relative-half-width checks and variance budgets.
    for (MeasureId mid = 0; mid < measure_count_; ++mid) {
        const GlobalMeasureStats& g = store_.global_stats(mid);
        if (g.non_nan_count > 0) {
            const double mu = g.sum / static_cast<double>(g.non_nan_count);
            global_mean_abs_ = std::max(global_mean_abs_, std::abs(mu));
        }
    }
}

void QueryEngine::initialize() {
    if (initialized_) return;
    access_path_.ensure_built();
    // Eager materialization applies only to a summary-keeping behavior over a
    // fully-built, stable substrate; on a cracking substrate the partitions are
    // created by queries, so summaries can only be filled lazily on first touch.
    if (config_.persist_summaries && access_path_.is_fully_built()) {
        materialize_all_summaries(access_path_, table_, store_, state_,
                                  measure_count_);
    }
    initialized_ = true;
}

void QueryEngine::build_residual_partitions(const QueryDecomposition& decomp) {
    residual_.clear();
    ql_.clear();

    std::unordered_map<PartitionId, std::size_t> seen;
    for (const ReusableStratum& s : decomp.reusable_strata) {
        if (seen.count(s.pid)) continue;
        seen.emplace(s.pid, residual_.size());
        ResidualPartition p;
        p.reusable       = true;
        p.write_to_state = config_.persist_summaries;
        p.pid   = s.pid;
        p.begin = s.begin;
        p.size  = static_cast<std::uint32_t>(s.end - s.begin);
        p.N     = s.population_size;
        p.tracker = s.tracker;
        if (!p.write_to_state) {
            // Tracker came from decompose (query-local fresh tracker); park
            // the per-query accumulator alongside the query-local strata.
            QueryLocalState st;
            st.moments.resize(measure_count_);
            st.tracker = s.tracker;
            ql_.emplace(s.pid, std::move(st));
        }
        residual_.push_back(std::move(p));
    }
    for (const QueryLocalStratum& s : decomp.query_local_strata) {
        if (ql_.count(s.pid)) continue;
        QueryLocalState st;
        st.moments.resize(measure_count_);
        st.tracker = std::make_shared<SampleTracker>(s.qualifying->size());
        auto [it, ok] = ql_.emplace(s.pid, std::move(st));
        ResidualPartition p;
        p.reusable       = false;
        p.write_to_state = false;
        p.pid   = s.pid;
        p.begin = s.begin;
        p.size  = static_cast<std::uint32_t>(s.qualifying->size());
        p.qualifying = s.qualifying.get();
        p.N = s.population_size;
        p.tracker = it->second.tracker;
        residual_.push_back(std::move(p));
    }
}

std::uint64_t QueryEngine::sampled_count(const ResidualPartition& p) const {
    if (p.write_to_state) {
        const MeasureSummary* s = state_.find(p.pid, 0);
        return s ? s->sampled_rows : 0;
    }
    return ql_.at(p.pid).sampled;
}

StratumSample QueryEngine::sample_for(const ResidualPartition& p,
                                      MeasureId mid) const {
    StratumSample s;
    s.N = p.N;
    if (p.write_to_state) {
        const MeasureSummary* sum = state_.find(p.pid, mid);
        if (sum) {
            s.m = sum->sampled_rows;
            s.n = sum->non_nan.non_nan_count;
            s.S = sum->non_nan.sum();
            s.Q = sum->non_nan.sum_sq();
        }
    } else {
        const QueryLocalState& q = ql_.at(p.pid);
        s.m = q.sampled;
        s.n = q.moments[mid].non_nan_count;
        s.S = q.moments[mid].sum();
        s.Q = q.moments[mid].sum_sq();
    }
    return s;
}

std::vector<std::vector<StratumSample>> QueryEngine::assemble_estimator_input()
    const {
    std::vector<std::vector<StratumSample>> by_measure(measure_count_);
    for (const ResidualPartition& p : residual_) {
        for (MeasureId mid = 0; mid < measure_count_; ++mid) {
            by_measure[mid].push_back(sample_for(p, mid));
        }
    }
    return by_measure;
}

std::vector<StratumAlloc> QueryEngine::assemble_allocation() const {
    std::vector<StratumAlloc> out;
    out.reserve(residual_.size());
    for (const ResidualPartition& p : residual_) {
        StratumAlloc a;
        a.N = p.N;
        a.sampled = sampled_count(p);
        a.observed.resize(measure_count_);
        for (MeasureId mid = 0; mid < measure_count_; ++mid) {
            a.observed[mid] = sample_for(p, mid);
        }
        out.push_back(std::move(a));
    }
    return out;
}

bool QueryEngine::read_round(const std::vector<std::uint64_t>& targets,
                             std::uint64_t ordinal, std::uint64_t round,
                             QueryMetrics& metrics) {
    // Draw the merged, ascending id/tag stream for a set of per-stratum targets
    // (no I/O), plus the per-stratum new-row counts. Drawing marks the per-
    // stratum sample trackers, so it must run exactly once per round -- the
    // scan-to-exactify decision below is therefore made before drawing.
    struct Batch {
        std::vector<RowId>         ids;
        std::vector<StratumTag>    tags;
        std::vector<std::uint64_t> new_rows;
    };
    const auto build = [&](const std::vector<std::uint64_t>& tg) -> Batch {
        Batch b;
        b.new_rows.assign(residual_.size(), 0);
        std::vector<StratumCursor> cursors;
        cursors.reserve(residual_.size());
        for (std::size_t h = 0; h < residual_.size(); ++h) {
            const std::uint64_t sampled = sampled_count(residual_[h]);
            const std::uint64_t target = h < tg.size() ? tg[h] : 0;
            if (target <= sampled) continue;
            const std::uint64_t delta = target - sampled;
            ResidualPartition& p = residual_[h];
            Rng rng(mix_seed(ordinal, round, h, target));
            StratumCursor c;
            if (p.reusable) {
                c = make_reusable_sampled_cursor(table_, p.begin, p.size, *p.tracker,
                                                 delta, rng,
                                                 static_cast<StratumTag>(h),
                                                 config_.sort_gather_by_row_id);
            } else {
                c = make_query_local_sampled_cursor(table_, p.begin, *p.qualifying,
                                                    *p.tracker, delta, rng,
                                                    static_cast<StratumTag>(h),
                                                    config_.sort_gather_by_row_id);
            }
            b.new_rows[h] = c.owned.size();
            cursors.push_back(std::move(c));
        }
        if (cursors.empty()) return b;
        KWayMerge merge(cursors);
        constexpr std::size_t kChunk = 4096;
        std::vector<RowId>      id_chunk(kChunk);
        std::vector<StratumTag> tag_chunk(kChunk);
        std::size_t got = 0;
        while ((got = merge.next_chunk(id_chunk, tag_chunk)) > 0) {
            b.ids.insert(b.ids.end(), id_chunk.begin(), id_chunk.begin() + got);
            b.tags.insert(b.tags.end(), tag_chunk.begin(), tag_chunk.begin() + got);
        }
        return b;
    };

    // Scan-to-exactify decision, made BEFORE drawing (drawing mutates the
    // trackers, so it runs once). Count the new rows this round would read; if
    // that many rows scattered over the table's row span would take the storage
    // scan path, the read sweeps the whole span regardless -- so read the
    // entire residual in that one pass instead of sampling now and re-scanning
    // the span over later rounds. The span is the whole table because the
    // prepared layout is shuffled, so any sample spreads across ~the whole file
    // (conservative for a clustered layout: a wider span only suppresses
    // escalation). Skipped when the round already targets the full residual
    // (the terminal exactify); a no-op in memory (would_scan is false there).
    std::uint64_t planned_new = 0;
    bool already_full = true;
    for (std::size_t h = 0; h < residual_.size(); ++h) {
        const std::uint64_t t = h < targets.size() ? targets[h] : 0;
        const std::uint64_t sampled = sampled_count(residual_[h]);
        if (t > sampled) planned_new += t - sampled;
        if (t < residual_[h].N) already_full = false;
    }
    const std::vector<std::uint64_t>* eff = &targets;
    std::vector<std::uint64_t> full;
    bool escalated = false;
    if (!already_full && planned_new > 0 && table_.size() > 0 &&
        store_.would_scan(0, static_cast<RowId>(table_.size() - 1), planned_new)) {
        full.assign(residual_.size(), 0);
        for (std::size_t h = 0; h < residual_.size(); ++h) full[h] = residual_[h].N;
        eff = &full;
        escalated = true;
    }

    // Draw exactly once, for the effective targets (the planned sample, or the
    // full residual if the round escalated).
    const Batch b = build(*eff);
    if (b.ids.empty()) return false;
    const std::vector<RowId>&         ids = b.ids;
    const std::vector<StratumTag>&    tags = b.tags;
    const std::vector<std::uint64_t>& new_rows = b.new_rows;

    std::vector<std::vector<MomentStats>> round_moments(
        residual_.size(), std::vector<MomentStats>(measure_count_));
    // One batched read covering every measure: the row ids are identical
    // across columns, so gathering all measures together computes the sort and
    // page-range coalescing plan once and reuses it for every column, rather
    // than rebuilding that O(n) plan per measure. The values land contiguously
    // per measure in vals[mid], which decouples the random read from the
    // streaming fold below (both measured faster than the per-measure
    // alternative at the engine's batch sizes -- decisively so in the mid
    // range where the repeated planning would otherwise dominate).
    std::vector<std::vector<double>> vals;
    // The k-way merge above yields globally ascending ids exactly when
    // gather-sorting is enabled (each merged cursor is then itself sorted);
    // pass that through so the store skips re-checking the order. The store
    // reports how it served this batch (scattered gather vs sequential scan);
    // accumulate the split across rounds for the on-disk access-path metric.
    GatherPathStats path{};
    store_.gather_all(ids, vals, config_.sort_gather_by_row_id, &path);
    metrics.scan_path_rows    += path.scan_rows;
    metrics.gather_path_rows  += path.gather_rows;
    metrics.scan_bytes_read   += path.scan_bytes;
    metrics.gather_bytes_read += path.gather_bytes;
    // Record this round's path so analysis can see multi-round re-scanning. An
    // eager (in-memory) store reports no rows on either path, so nothing is
    // recorded and round_paths stays empty for in-memory cells.
    if (path.scan_rows != 0 || path.gather_rows != 0) {
        metrics.round_paths.push_back(RoundPath{round, path.scan_rows, path.gather_rows,
                                                path.scan_bytes, path.gather_bytes});
    }
    for (MeasureId mid = 0; mid < measure_count_; ++mid) {
        metrics.measure_reads += ids.size();
        for (std::size_t i = 0; i < ids.size(); ++i) {
            round_moments[tags[i]][mid].add_if_present(vals[mid][i]);
        }
    }

    for (std::size_t h = 0; h < residual_.size(); ++h) {
        if (new_rows[h] == 0) continue;
        ResidualPartition& p = residual_[h];
        // Rows that complete a stratum are full-read work, rows that keep it
        // partially sampled are sampling work -- regardless of which round
        // shape requested them (`eff` is the effective target set, which the
        // scan-to-exactify escalation may have raised to the full residual).
        const std::uint64_t eff_target = h < eff->size() ? (*eff)[h] : 0;
        if (eff_target >= p.N) {
            metrics.exactified_rows += new_rows[h];
        } else {
            metrics.sampled_rows += new_rows[h];
        }
        if (p.write_to_state) {
            for (MeasureId mid = 0; mid < measure_count_; ++mid) {
                SampleDelta d;
                d.new_sampled_rows = new_rows[h];
                d.moments = round_moments[h][mid];
                state_.update_sampled(p.pid, mid, d);
            }
        } else {
            QueryLocalState& q = ql_.at(p.pid);
            q.sampled += new_rows[h];
            for (MeasureId mid = 0; mid < measure_count_; ++mid) {
                q.moments[mid].merge(round_moments[h][mid]);
            }
        }
    }
    return escalated;
}

void QueryEngine::exactify_round(std::uint64_t ordinal, std::uint64_t round,
                                 QueryMetrics& metrics) {
    // Target every residual stratum at its whole population: the draw clamps
    // to the rows not yet sampled, so this reads each remainder exactly once.
    std::vector<std::uint64_t> targets(residual_.size());
    for (std::size_t h = 0; h < residual_.size(); ++h) {
        targets[h] = residual_[h].N;
    }
    read_round(targets, ordinal, round, metrics);
}

bool QueryEngine::all_satisfied(const std::vector<AggregateEstimate>& est,
                                double rel) const {
    for (const AggregateEstimate& e : est) {
        if (e.exact) continue;
        if (!(e.relative_half_width <= rel)) return false;  // catches NaN/inf
    }
    return true;
}

QueryResult QueryEngine::execute(const RangeQuery& query,
                                 std::uint64_t query_ordinal) {
    QueryResult result;
    QueryMetrics& m = result.metrics;
    m.query_ordinal = query_ordinal;

    const bool force_exact =
        config_.accuracy_mode == EngineConfig::AccuracyMode::ForceExact;
    const double rel = force_exact ? 0.0 : query.target.relative_error;
    const double conf = force_exact ? 0.95 : query.target.confidence;
    AccuracyTarget eff;
    eff.relative_error = rel;
    eff.confidence = conf;

    initialize();
    // One top-down descent from the substrate roots builds the decomposition:
    // it classifies each node, cracks partial leaves through the substrate's
    // refine() when the substrate adapts, and stops at contained nodes already
    // exact for every measure. The substrate supplies geometry and
    // partitioning; the descent owns the state-store decisions.
    DecompositionResult d = decompose_descent(
        query.predicate, access_path_, state_, table_, measure_count_,
        config_.persist_summaries, access_path_.supports_refine());
    m.partitions_refined = d.partitions_refined;
    m.frontier_partitions = d.frontier_partitions;
    m.exact_contributors = d.exact_contributor_partitions;
    m.reusable_sampled_strata = d.reusable_sampled_partitions;
    m.reusable_absent_strata = d.reusable_absent_partitions;
    m.query_local_strata = d.query_local_partitions;

    build_residual_partitions(d.decomposition);

    auto estimates = estimator_.estimate(d.exact_bucket, d.total_count,
                                         assemble_estimator_input(),
                                         global_mean_abs_, conf);

    if (rel <= 0.0) {
        // Read every residual stratum to completion, then re-estimate: with no
        // residual variance left the answer is exact.
        exactify_round(query_ordinal, /*round=*/0, m);
        estimates = estimator_.estimate(d.exact_bucket, d.total_count,
                                        assemble_estimator_input(),
                                        global_mean_abs_, conf);
        m.status = "exact";
        m.exactify_cause = "none";
        m.target_satisfied = true;
        result.aggregates = std::move(estimates);
        return result;
    }

    std::uint64_t round = 0;
    bool exactified = false;
    std::string cause = "none";
    double pre_err = 0.0;

    // Pilot phase: bring every residual stratum to the pilot sample (small
    // strata are taken whole) so the planner works from observed statistics.
    // Strata with enough persisted samples from earlier queries need no
    // pilot reads.
    if (!all_satisfied(estimates, rel)) {
        ++round;
        const AllocationPlan pilot =
            allocator_.plan_pilot(assemble_allocation());
        if (!pilot.no_target_increase) {
            if (read_round(pilot.target, query_ordinal, round, m)) {
                exactified = true;
                cause = "scan_cheaper_than_gather";
            }
            estimates = estimator_.estimate(d.exact_bucket, d.total_count,
                                            assemble_estimator_input(),
                                            global_mean_abs_, conf);
        }
    }

    // Planned rounds: Neyman re-plans from observed statistics, each one
    // either converging or escalating saturated strata to full reads. The
    // round budget counts the pilot and the terminal full read: with a
    // budget of R, rounds 2..R-1 are planned rounds and round R reads the
    // remainder.
    while (!all_satisfied(estimates, rel)) {
        const bool budget_left =
            round + 1 < config_.allocator.max_sampling_rounds;
        AllocationPlan plan;
        if (budget_left) {
            plan = allocator_.plan(assemble_allocation(), estimates, eff,
                                   global_mean_abs_);
        }
        // Terminal fallback: budget spent, or the planner cannot raise any
        // target while an aggregate still fails. Reading the remaining
        // residual in full always terminates with a correct answer.
        if (!budget_left || plan.no_target_increase) {
            ++round;
            pre_err = max_relative_half_width(estimates);
            exactify_round(query_ordinal, round, m);
            exactified = true;
            cause = "gave_up";
            estimates = estimator_.estimate(d.exact_bucket, d.total_count,
                                            assemble_estimator_input(),
                                            global_mean_abs_, conf);
            break;
        }

        ++round;
        if (read_round(plan.target, query_ordinal, round, m)) {
            exactified = true;
            cause = "scan_cheaper_than_gather";
        }
        estimates = estimator_.estimate(d.exact_bucket, d.total_count,
                                        assemble_estimator_input(),
                                        global_mean_abs_, conf);
    }

    const bool satisfied = all_satisfied(estimates, rel);
    m.adaptive_rounds = round;
    m.exactify_cause = cause;
    m.pre_exactification_error_bound = pre_err;
    m.target_satisfied = satisfied;
    m.status = exactified ? (satisfied ? "exactified" : "exhausted_unconverged")
                          : "converged";
    result.aggregates = std::move(estimates);
    return result;
}

}  // namespace a3i
