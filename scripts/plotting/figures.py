"""Declared figure manifest.

Each figure declares, as data, how to build itself from the loaded matrix.
Unlike the exploratory driver (which *infers* axis roles), a figure here selects
the cells it needs and which axis it varies. A figure builds zero or more
`(suffix, matplotlib figure)` items -- one per facet it applies to; yielding none
means its cells are absent (the driver skips it, or fails under strict mode). The
comparison guards (shared cold/mem, shared-prefix clip) are applied as each
figure selects its cells.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from . import aggregate, render


class MissingCells(Exception):
    """A declared figure whose required cells are absent from the loaded matrix."""


@dataclass(frozen=True)
class Figure:
    id: str
    build: Callable  # (frame, eb) -> list[(suffix, matplotlib figure)]


# Approximate run ids (the ones with error bounds / coverage).
APPROX = ("a3i", "adkd_sampling")

# Default values the comparison figures pin the within-facet, non-method axes to:
# a method comparison must hold every axis but the method fixed, so it picks one
# nm and one eb rather than drawing every (nm x eb) combination. Overridable per
# run from the CLI (--nm, --eb). A pinned nm that is absent in a facet falls back
# to the largest present there. (Selectivity and dataset size are not here: they
# are part of the (dataset, workload) facet identity, not within-facet axes.)
PIN_DEFAULTS = {"eb": 0.01, "nm": 4, "partition_size": 1024, "mem": "INMEM"}

# Optional cap on the per-query figure: show only the first N queries to zoom
# into the convergence region. Default None -> the full run, so late adaptation
# is never hidden; set via the CLI (--per-query-head) when a clean zoom is wanted
# (log-y already keeps the full run legible). The cumulative figure is unaffected.
PER_QUERY_HEAD = None

_SEL_RE = re.compile(r"_sel([0-9.]+)")
_SIZE_RE = re.compile(r"_(\d+)([MB])\b")
_FAMILY_RE = re.compile(r"^synth\d*_\d+[MB]_")


def _selectivity(workload) -> float | None:
    m = _SEL_RE.search(str(workload))
    return float(m.group(1)) if m else None


def _row_count(dataset) -> float | None:
    m = _SIZE_RE.search(str(dataset))
    if not m:
        return None
    return float(m.group(1)) * (1e9 if m.group(2) == "B" else 1e6)


def _family(workload) -> str:
    """A workload's size-independent family (e.g. `clustered`, `random_sel0.1`)."""
    return _FAMILY_RE.sub("", str(workload))


def _facets(frame):
    """Yield (tag, title, subframe) per (dataset, workload). The title is just the
    workload (which already embeds the dataset name, so the dataset prefix would be
    redundant and widen the title); the tag keeps both for the output filename."""
    for (dataset, workload), sub in frame.groupby(["dataset", "workload"], dropna=False):
        yield f"{dataset}__{workload}", str(workload), sub


def _resolve_nm(frame, nm):
    """The nm to pin to in this frame: the requested one if present, else the
    largest available (so a facet without the pinned nm still renders)."""
    present = sorted(pd.to_numeric(frame["nm"], errors="coerce").dropna().unique())
    if not present:
        return None
    if nm is not None and nm in present:
        return nm
    return present[-1]


def _annotate(base, *, nm=None, eb=None):
    """Append the pinned (non-varied) axes to a facet title on a second line, e.g.
    'taxi_random\\n(nm=4, eb=0.01)' -- a new line rather than a suffix so the
    parenthetical does not widen (and overflow) a narrow single-column title. A
    figure passes only the axes it holds fixed; the axis it sweeps is the x-axis
    and is omitted. Honoured only when the title is drawn (see render.TITLES)."""
    bits = []
    if nm is not None:
        bits.append(f"nm={nm:g}")
    if eb is not None:
        bits.append(f"eb={eb:g}")
    return f"{base}\n({', '.join(bits)})" if bits else base


def _method_slice(sub, eb, nm):
    """A clean method comparison within one facet: a single `nm` (the pinned one,
    falling back to the largest present), exact methods plus approximate ones at
    `eb`, with shared cold/mem and the shared-prefix clip. Returns None if
    nothing qualifies. Raises if cold/mem are mixed."""
    nm = _resolve_nm(sub, nm)
    if nm is None:
        return None
    s = sub[sub["nm"] == nm]
    s = s[s["eb"].isna() | (s["eb"] == eb)]
    if s.empty:
        return None
    aggregate.shared_or_raise(s, "cold")
    aggregate.shared_or_raise(s, "mem")
    return aggregate.clip_to_shared_prefix(s)


# --- per-facet method-comparison figures ------------------------------------

def _cumulative_time(frame, eb, nm):
    out = []
    for tag, title, sub in _facets(frame):
        s = _method_slice(sub, eb, nm)
        if s is None:
            continue
        title = _annotate(title, nm=_resolve_nm(sub, nm), eb=eb)
        out.append((tag, render.trajectory(s, metric="latency_ms", cumulative=True,
                                           title=title, linewidth=1.6)))
    return out


def _per_query_latency(frame, eb, nm):
    out = []
    for tag, title, sub in _facets(frame):
        s = _method_slice(sub, eb, nm)
        if s is None:
            continue
        title = _annotate(title, nm=_resolve_nm(sub, nm), eb=eb)
        out.append((tag, render.trajectory(s, metric="latency_ms", cumulative=False,
                                           title=title, head=PER_QUERY_HEAD,
                                           linewidth=0.7)))
    return out


def _measure_reads(frame, eb, nm):
    out = []
    for tag, title, sub in _facets(frame):
        s = _method_slice(sub, eb, nm)
        if s is None:
            continue
        title = _annotate(title, nm=_resolve_nm(sub, nm), eb=eb)
        out.append((tag, render.bar(aggregate.cell_summary(s, eb),
                                    metric="total_reads", title=title, logy=True)))
    return out


def _latency_tail(frame, eb, nm):
    out = []
    for tag, title, sub in _facets(frame):
        s = _method_slice(sub, eb, nm)
        if s is None:
            continue
        title = _annotate(title, nm=_resolve_nm(sub, nm), eb=eb)
        out.append((tag, render.grouped_bar(
            aggregate.cell_summary(s, eb),
            metrics=["lat_p50", "lat_p95", "lat_p99"], title=title,
            ylabel="latency (ms)", logy=True)))
    return out


# --- accuracy figures (approximate methods) ---------------------------------

def _achieved_vs_targeted(frame, eb, nm):
    """Guarantee compliance: per (method, eb), the p50 and p95 of the achieved
    relative error over the genuinely-sampled estimates, against the y = x
    line. The promise is probabilistic -- each estimate is within the target
    with the requested confidence (default 0.95) -- so the p95 sitting on or
    below the diagonal IS the guarantee holding; raw per-estimate scatters
    always show ~5% of points above the line by design and invite misreading.
    Quantiles pool aggregates and measures (each estimate is one trial, as in
    the per-query error distributions AQP evaluations conventionally report);
    the per-measure split, where one heavy-tailed measure can hide, belongs to
    the validator's tables."""
    out = []
    for tag, title, sub in _facets(frame):
        rnm = _resolve_nm(sub, nm)
        sub = sub[sub["nm"] == rnm]
        pts = aggregate.with_error(sub)
        pts = pts[pts["method"].isin(APPROX) & ~pts["exact"]]
        pts = pts.dropna(subset=["error", "eb"])
        if pts.empty:
            continue
        q = (pts.groupby(["method", "eb"])["error"]
                .quantile([0.5, 0.95]).unstack()
                .rename(columns={0.5: "p50", 0.95: "p95"}).reset_index())
        out.append((tag, render.quantiles(
            q, x="eb", bands=("p95", "p50"), by="method", diagonal=True,
            xlabel="targeted relative error (eb)",
            ylabel="achieved relative error",
            title=_annotate(title, nm=rnm))))
    return out


# --- sweeps (a numeric experiment axis on x) --------------------------------

def _effect_of_nm(frame, eb, nm):
    del nm  # nm is the swept axis here, not a pinned one
    out = []
    for tag, title, sub in _facets(frame):
        s = sub[sub["eb"].isna() | (sub["eb"] == eb)]
        if s["nm"].dropna().nunique() < 2:
            continue
        aggregate.shared_or_raise(s, "cold")
        aggregate.shared_or_raise(s, "mem")
        out.append((tag, render.sweep(aggregate.cell_summary(s, eb),
                                      x_axis="nm", metric="cum_ms",
                                      title=_annotate(title, eb=eb), logy=True)))
    return out


def _effect_of_eb(frame, eb, nm):
    out = []
    for tag, title, sub in _facets(frame):
        s = sub[sub["method"].isin(APPROX) & sub["eb"].notna()]
        if s["eb"].dropna().nunique() < 2:
            continue
        rnm = _resolve_nm(s, nm)
        s = s[s["nm"] == rnm]
        aggregate.shared_or_raise(s, "cold")
        aggregate.shared_or_raise(s, "mem")
        out.append((tag, render.sweep(aggregate.cell_summary(s, eb),
                                      x_axis="eb", metric="total_latency_ms",
                                      title=_annotate(title, nm=rnm), logx=True, logy=True)))
    return out


def _speedup_vs_error(frame, eb, nm):
    """Speedup over the exact scan vs the error bound, per approximate method --
    the AQP headline ("N x faster at e% error"). Needs >= 2 eb values (an error
    sweep) to draw a curve; skips otherwise."""
    out = []
    for tag, title, sub in _facets(frame):
        rnm = _resolve_nm(sub, nm)
        s = sub[sub["nm"] == rnm]
        if s[s["method"].isin(APPROX)]["eb"].dropna().nunique() < 2:
            continue
        aggregate.shared_or_raise(s, "cold")
        aggregate.shared_or_raise(s, "mem")
        summ = aggregate.cell_summary(s, eb)
        summ = summ[summ["method"].isin(APPROX)].dropna(subset=["speedup_vs_scan", "eb"])
        if summ.empty:
            continue
        out.append((tag, render.sweep(summ, x_axis="eb", metric="speedup_vs_scan",
                                      title=_annotate(title, nm=rnm), logx=True, logy=True)))
    return out


def _selectivity_sweep(frame, eb, nm):
    out = []
    for dataset, dsub in frame.groupby("dataset", dropna=False):
        s = dsub.copy()
        s["selectivity"] = s["workload"].map(_selectivity)
        s = s[s["selectivity"].notna() & (s["eb"].isna() | (s["eb"] == eb))]
        if s["selectivity"].nunique() < 2:
            continue
        rnm = _resolve_nm(s, nm)
        s = s[s["nm"] == rnm]
        aggregate.shared_or_raise(s, "cold")
        aggregate.shared_or_raise(s, "mem")
        summ = aggregate.cell_summary(s, eb)
        summ["selectivity"] = summ["workload"].map(_selectivity)
        out.append((str(dataset), render.sweep(
            summ, x_axis="selectivity", metric="cum_ms",
            title=_annotate(str(dataset), nm=rnm, eb=eb), logx=True, logy=True)))
    return out


def _scalability_sweep(frame, eb, nm):
    s = frame.copy()
    s["N"] = s["dataset"].map(_row_count)
    s = s[s["N"].notna() & (s["eb"].isna() | (s["eb"] == eb))]
    if s.empty:
        return []
    rnm = _resolve_nm(s, nm)
    s = s[s["nm"] == rnm]
    s["family"] = s["workload"].map(_family)
    out = []
    for family, fsub in s.groupby("family", dropna=False):
        if fsub["N"].nunique() < 2:
            continue
        aggregate.shared_or_raise(fsub, "cold")
        aggregate.shared_or_raise(fsub, "mem")
        summ = aggregate.cell_summary(fsub, eb)
        summ["N"] = summ["dataset"].map(_row_count)
        out.append((str(family), render.sweep(
            summ, x_axis="N", metric="cum_ms",
            title=_annotate(f"scalability ({family})", nm=rnm, eb=eb),
            logx=True, logy=True)))
    return out


FIGURES = (
    Figure("cumulative_time", _cumulative_time),
    Figure("per_query_latency", _per_query_latency),
    Figure("measure_reads", _measure_reads),
    Figure("latency_tail", _latency_tail),
    Figure("achieved_vs_targeted_error", _achieved_vs_targeted),
    Figure("effect_of_nm", _effect_of_nm),
    Figure("effect_of_eb", _effect_of_eb),
    Figure("speedup_vs_error", _speedup_vs_error),
    Figure("selectivity_sweep", _selectivity_sweep),
    Figure("scalability_sweep", _scalability_sweep),
)


def by_id(figure_id: str) -> Figure:
    for fig in FIGURES:
        if fig.id == figure_id:
            return fig
    raise KeyError(figure_id)
