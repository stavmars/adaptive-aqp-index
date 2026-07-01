#include "a3i/engine/query_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "a3i/aqp/decompose.hpp"
#include "a3i/aqp/eager_materialize.hpp"
#include "a3i/aqp/outlier_scorer.hpp"
#include "a3i/aqp/stratum_cursor.hpp"
#include "a3i/engine/method.hpp"
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
    // Resolve the behavior-and-substrate derived flags once: whether the
    // descent may crack partitions and whether summaries are precomputed up
    // front. Both follow from the substrate's capabilities and the behavior.
    const ResolvedRunConfig rr = ResolvedRunConfig::resolve(config_, access_path_);
    allow_refine_      = rr.allow_refine;
    eager_materialize_ = rr.eager_materialize;

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
    // The outlier index, when enabled, is built at init from the exact global
    // statistics. When eager materialization runs it feeds the scorer for free
    // off the same column sweep; a lazy substrate has no such pass, so it pays
    // one dedicated streaming sweep. Null scorer => disabled => no flag column.
    std::unique_ptr<OutlierScorer> scorer = make_scorer();
    // Eager materialization applies when summaries persist and the substrate
    // prebuilds its partitions; otherwise summaries are filled lazily on first
    // touch as queries reach each partition.
    if (eager_materialize_) {
        materialize_all_summaries(access_path_, table_, store_, state_,
                                  measure_count_, access_path_.row_owner_map(),
                                  scorer.get());
    } else if (scorer) {
        run_scoring_sweep(*scorer);
    }
    if (scorer) {
        table_.set_flags_by_rowid(scorer->finalize());
    }
    initialized_ = true;
}

std::unique_ptr<OutlierScorer> QueryEngine::make_scorer() const {
    const double frac = config_.outlier_budget_fraction;
    if (!(frac > 0.0) || measure_count_ == 0 || table_.size() == 0) return nullptr;
    const std::uint64_t budget = static_cast<std::uint64_t>(
        frac * static_cast<double>(table_.size()) + 0.5);
    if (budget == 0) return nullptr;
    std::vector<double> mean(measure_count_, 0.0);
    std::vector<double> inv_scale(measure_count_, 0.0);  // 1/mean per measure
    // Supply the per-measure center (mean) and scale (1/mean) that OutlierScorer
    // scores rows against, from the exact global stats. The scoring formula lives in OutlierScorer. A zero mean has no
    // 1/mean scale and is skipped; any nonzero mean (either sign) participates.
    bool any = false;
    for (MeasureId mid = 0; mid < measure_count_; ++mid) {
        const GlobalMeasureStats& g = store_.global_stats(mid);
        if (g.non_nan_count == 0) continue;
        const double mu = g.sum / static_cast<double>(g.non_nan_count);
        mean[mid] = mu;
        if (mu != 0.0) { inv_scale[mid] = 1.0 / mu; any = true; }
    }
    if (!any) return nullptr;  // no nonzero-mean measure to score
    return std::make_unique<OutlierScorer>(std::move(mean), std::move(inv_scale),
                                           budget);
}

void QueryEngine::run_scoring_sweep(OutlierScorer& scorer) const {
    const std::size_t n = table_.size();
    const std::size_t k = measure_count_;
    if (n == 0 || k == 0) return;
    // Mirror the materialize column sweep: read every measure column once in
    // ascending row-id order and feed each row's values to the scorer.
    constexpr std::size_t kBlockRows = std::size_t{1} << 20;
    std::vector<std::vector<double>> scratch(k);
    std::vector<std::span<const double>> cols(k);
    std::vector<double> row_vals(k);
    for (MeasureId mid = 0; mid < k; ++mid) {
        store_.advise_sequential(static_cast<MeasureId>(mid));
    }
    for (std::size_t block = 0; block < n; block += kBlockRows) {
        const std::size_t count = std::min(kBlockRows, n - block);
        for (std::size_t mid = 0; mid < k; ++mid) {
            cols[mid] = store_.read_rows(static_cast<MeasureId>(mid),
                                         static_cast<RowId>(block), count,
                                         scratch[mid]);
        }
        for (std::size_t i = 0; i < count; ++i) {
            for (std::size_t mid = 0; mid < k; ++mid) row_vals[mid] = cols[mid][i];
            scorer.observe(static_cast<RowId>(block + i), row_vals);
        }
    }
}

void QueryEngine::apply_addback(DecompositionResult& d,
                                QueryMetrics& metrics) const {
    if (d.addback_rows.empty()) return;
    // The held-out rows contribute exactly: gather their measure values and
    // fold the non-missing ones into the same exact bucket the contained-exact
    // contributors use, with zero variance.
    std::vector<std::vector<double>> vals;
    GatherPathStats path{};
    // The held-out row ids are collected in descent/position order, not row-id
    // order, so they are not guaranteed ascending; pass already_sorted=false so
    // the store orders them itself (required for the on-disk scan path to read
    // the right rows; a no-op for an in-memory store).
    store_.gather_all(d.addback_rows, vals, /*already_sorted=*/false, &path);
    metrics.scan_path_rows    += path.scan_rows;
    metrics.gather_path_rows  += path.gather_rows;
    metrics.scan_bytes_read   += path.scan_bytes;
    metrics.gather_bytes_read += path.gather_bytes;
    for (MeasureId mid = 0; mid < measure_count_; ++mid) {
        double sum = 0.0;
        std::uint64_t cnt = 0;
        for (std::size_t i = 0; i < d.addback_rows.size(); ++i) {
            const double v = vals[mid][i];
            if (!std::isnan(v)) {
                sum += v;
                ++cnt;
            }
        }
        d.exact_bucket.sum_by_measure[mid] += sum;
        d.exact_bucket.count_by_measure[mid] += cnt;
        metrics.measure_reads += d.addback_rows.size();
    }
    // These held-out rows (query-local boundary outliers, and non-persist
    // reusable) are read per query rather than banked, but they are still
    // outlier reads -- count them with the banked ones so outlier_rows is the
    // complete held-out-read total and the read-work identity holds.
    metrics.outlier_rows += d.addback_rows.size();
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
        p.excluded = s.excluded.get();
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
                             QueryMetrics& metrics, ExactBucket& exact_bucket) {
    // Draw the merged, ascending id/tag stream for a set of per-stratum targets
    // (no I/O), plus the per-stratum new-row counts. Drawing marks the per-
    // stratum sample trackers, so it must run exactly once per round -- the
    // scan-to-exactify decision below is therefore made before drawing.
    struct Batch {
        std::vector<RowId>         ids;
        std::vector<StratumTag>    tags;
        std::vector<char>          is_outlier;  // 1 = held-out row routed to bank
        std::vector<std::uint64_t> new_rows;      // body rows drawn per stratum
        std::vector<std::uint64_t> outlier_new;   // held-out rows appended per stratum
    };
    const auto build = [&](const std::vector<std::uint64_t>& tg) -> Batch {
        Batch b;
        b.new_rows.assign(residual_.size(), 0);
        b.outlier_new.assign(residual_.size(), 0);
        std::vector<StratumCursor> cursors;
        cursors.reserve(residual_.size());
        // A reusable persisted partition whose held-out rows are not yet banked
        // is materialized this round by adding a second cursor of its flagged
        // rows (tag = h, marked is_outlier) to the same merge -- so the held-out
        // rows ride the one gather and the merged stream stays globally sorted.
        // Triggered on first touch regardless of the body draw, so a zero-delta
        // or all-outlier partition still banks.
        for (std::size_t h = 0; h < residual_.size(); ++h) {
            ResidualPartition& p = residual_[h];
            const std::uint64_t sampled = sampled_count(p);
            const std::uint64_t target = h < tg.size() ? tg[h] : 0;
            const std::uint64_t delta = target > sampled ? target - sampled : 0;
            bool need_outlier = false;
            if (p.reusable && p.write_to_state) {
                const MeasureSummary* s0 = state_.find(p.pid, 0);
                if (s0 != nullptr && s0->outlier_rows > 0 &&
                    !s0->outliers_materialized) {
                    need_outlier = true;
                }
            }
            if (delta == 0 && !need_outlier) continue;
            if (delta > 0) {
                Rng rng(mix_seed(ordinal, round, h));
                StratumCursor c;
                if (p.reusable) {
                    c = make_reusable_sampled_cursor(table_, p.begin, p.size, *p.tracker,
                                                     delta, rng,
                                                     static_cast<StratumTag>(h),
                                                     config_.sort_gather_by_row_id,
                                                     p.excluded);
                } else {
                    c = make_query_local_sampled_cursor(table_, p.begin, *p.qualifying,
                                                        *p.tracker, delta, rng,
                                                        static_cast<StratumTag>(h),
                                                        config_.sort_gather_by_row_id);
                }
                b.new_rows[h] = c.owned.size();
                cursors.push_back(std::move(c));
            }
            if (need_outlier) {
                StratumCursor oc =
                    make_outlier_cursor(table_, p.begin, p.size,
                                        static_cast<StratumTag>(h),
                                        config_.sort_gather_by_row_id);
                b.outlier_new[h] = oc.owned.size();
                cursors.push_back(std::move(oc));
            }
        }
        if (cursors.empty()) return b;
        KWayMerge merge(cursors);
        constexpr std::size_t kChunk = 4096;
        std::vector<RowId>      id_chunk(kChunk);
        std::vector<StratumTag> tag_chunk(kChunk);
        std::vector<char>       out_chunk(kChunk);
        std::size_t got = 0;
        while ((got = merge.next_chunk(id_chunk, tag_chunk, out_chunk)) > 0) {
            b.ids.insert(b.ids.end(), id_chunk.begin(), id_chunk.begin() + got);
            b.tags.insert(b.tags.end(), tag_chunk.begin(), tag_chunk.begin() + got);
            b.is_outlier.insert(b.is_outlier.end(),
                                out_chunk.begin(), out_chunk.begin() + got);
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
    const std::vector<char>&          is_outlier = b.is_outlier;
    const std::vector<std::uint64_t>& new_rows = b.new_rows;
    const std::vector<std::uint64_t>& outlier_new = b.outlier_new;

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
    // gather-sorting is enabled (each merged cursor -- body and held-out alike
    // -- is then itself sorted); pass that through so the store skips
    // re-checking the order. The store reports how it served this batch
    // (scattered gather vs sequential scan); accumulate the split across rounds
    // for the on-disk access-path metric.
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
    // Held-out (outlier) rows are routed to a separate per-stratum accumulator
    // and banked exactly; only the body rows feed the sampling moments, so the
    // variance that drives allocation stays outlier-free. That accumulator is
    // allocated and consulted only when this round actually carries held-out
    // rows -- otherwise every row is a body row and the fold skips the per-row
    // branch.
    bool have_outliers = false;
    for (std::size_t h = 0; h < residual_.size(); ++h) {
        if (outlier_new[h] != 0) { have_outliers = true; break; }
    }
    std::vector<std::vector<MomentStats>> outlier_moments;
    if (have_outliers) {
        outlier_moments.assign(residual_.size(),
                               std::vector<MomentStats>(measure_count_));
    }
    for (MeasureId mid = 0; mid < measure_count_; ++mid) {
        metrics.measure_reads += ids.size();
        if (have_outliers) {
            for (std::size_t i = 0; i < ids.size(); ++i) {
                MomentStats& dst = is_outlier[i] ? outlier_moments[tags[i]][mid]
                                                 : round_moments[tags[i]][mid];
                dst.add_if_present(vals[mid][i]);
            }
        } else {
            for (std::size_t i = 0; i < ids.size(); ++i) {
                round_moments[tags[i]][mid].add_if_present(vals[mid][i]);
            }
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

    // Bank each materialized partition's held-out rows: store their exact
    // per-measure contribution in the summary (separate from the body moments)
    // and add it to the exact bucket so every later estimate includes it. Read
    // once here, then reused for free on later queries. Only reusable persisted
    // partitions reach this (need_outlier required write_to_state), so the
    // summary exists. Charged to exactified rows (read exactly, not sampled).
    for (std::size_t h = 0; h < residual_.size(); ++h) {
        if (outlier_new[h] == 0) continue;
        ResidualPartition& p = residual_[h];
        metrics.outlier_rows += outlier_new[h];
        for (MeasureId mid = 0; mid < measure_count_; ++mid) {
            const double os = outlier_moments[h][mid].sum();
            const std::uint64_t oc = outlier_moments[h][mid].non_nan_count;
            state_.bank_outliers(p.pid, mid, os, oc);
            exact_bucket.sum_by_measure[mid] += os;
            exact_bucket.count_by_measure[mid] += oc;
        }
    }
    return escalated;
}

void QueryEngine::exactify_round(std::uint64_t ordinal, std::uint64_t round,
                                 QueryMetrics& metrics, ExactBucket& exact_bucket) {
    // Target every residual stratum at its whole population: the draw clamps
    // to the rows not yet sampled, so this reads each remainder exactly once.
    std::vector<std::uint64_t> targets(residual_.size());
    for (std::size_t h = 0; h < residual_.size(); ++h) {
        targets[h] = residual_[h].N;
    }
    read_round(targets, ordinal, round, metrics, exact_bucket);
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
        config_.persist_summaries, allow_refine_);
    m.partitions_refined = d.partitions_refined;
    m.frontier_partitions = d.frontier_partitions;
    m.exact_contributors = d.exact_contributor_partitions;
    m.reusable_sampled_strata = d.reusable_sampled_partitions;
    m.reusable_absent_strata = d.reusable_absent_partitions;
    m.query_local_strata = d.query_local_partitions;

    build_residual_partitions(d.decomposition);

    // Fold the held-out rows into the exact bucket once, before any estimate;
    // every estimate below then reads the augmented bucket. A no-op when the
    // index is disabled (no rows were held out).
    apply_addback(d, m);

    auto estimates = estimator_.estimate(d.exact_bucket, d.total_count,
                                         assemble_estimator_input(),
                                         global_mean_abs_, conf);

    if (rel <= 0.0) {
        // Read every residual stratum to completion, then re-estimate: with no
        // residual variance left the answer is exact.
        exactify_round(query_ordinal, /*round=*/0, m, d.exact_bucket);
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

    // A reusable persisted partition whose held-out rows are not yet banked
    // must trigger a round so the round-fold materializes them -- otherwise a
    // query containing only such partitions (e.g. an all-outlier partition with
    // no body to sample) could converge on the body estimate before the
    // held-out rows are ever read.
    const auto unmaterialized_outliers = [&]() {
        if (!table_.flags_enabled()) return false;  // no outlier index => nothing to bank
        for (const ResidualPartition& p : residual_) {
            if (!p.reusable || !p.write_to_state) continue;
            const MeasureSummary* s = state_.find(p.pid, 0);
            if (s != nullptr && s->outlier_rows > 0 && !s->outliers_materialized) {
                return true;
            }
        }
        return false;
    };

    // Pilot phase: bring every residual stratum to the pilot sample (small
    // strata are taken whole) so the planner works from observed statistics.
    // Strata with enough persisted samples from earlier queries need no
    // pilot reads.
    if (!all_satisfied(estimates, rel) || unmaterialized_outliers()) {
        ++round;
        const AllocationPlan pilot =
            allocator_.plan_pilot(assemble_allocation());
        if (!pilot.no_target_increase || unmaterialized_outliers()) {
            if (read_round(pilot.target, query_ordinal, round, m, d.exact_bucket)) {
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
            exactify_round(query_ordinal, round, m, d.exact_bucket);
            exactified = true;
            cause = "gave_up";
            estimates = estimator_.estimate(d.exact_bucket, d.total_count,
                                            assemble_estimator_input(),
                                            global_mean_abs_, conf);
            break;
        }

        ++round;
        if (read_round(plan.target, query_ordinal, round, m, d.exact_bucket)) {
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
