#!/usr/bin/env python3
"""Render a workload's query rectangles over the dataset domain (static 2-D plot).

Standalone and independent of any results: it reads a workload CSV (the generated
rectangles) and the co-located `<name>.metadata.json` (which embeds the dataset
`domain_bounds` and dimension count at generation time), then draws every query
rectangle over the domain with the axes aligned to the dimension domains.

Sequences are long (500 queries), so `--max-queries` caps how many rectangles are
drawn; first/last/waypoint centre markers show sequence order and are off by
default. Output is a vector PDF by default (PNG/SVG by file extension).

For geographic workloads (lon/lat data such as taxi and ebird), `--basemap auto`
(the default) overlays an OpenStreetMap tile background clipped to the dataset
domain, reprojecting to Web Mercator so the tiles align and labelling the ticks
in degrees; non-geographic workloads get a plain plot. `--basemap on/off` forces
it. The OSM background needs the optional `contextily` package; without it the
rectangles still render.

Usage:
    scripts/visualize_workload.py <workload.csv | workload_id> [-o out.pdf]
    scripts/visualize_workload.py taxi_random --markers          # auto OSM basemap
    scripts/visualize_workload.py --all                          # every workload
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import a3i_config
from plotting import style


class WorkloadError(RuntimeError):
    """A workload CSV/metadata pair that cannot be read or is inconsistent."""


def _metadata_path(csv_path: Path) -> Path:
    """The sidecar `<name>.metadata.json` next to a workload CSV."""
    return csv_path.parent / (csv_path.stem + ".metadata.json")


def resolve_workload(arg: str) -> Path:
    """Accept either a path to a workload CSV or a bare id under the workloads dir."""
    p = Path(arg)
    if p.is_file():
        return p
    cand = a3i_config.workloads_dir() / f"{arg}.csv"
    if cand.is_file():
        return cand
    raise WorkloadError(f"workload not found: {arg!r} (tried {p} and {cand})")


def load_workload(csv_path, meta_path=None):
    """Parse a workload into (rects, meta).

    `rects` is a list of `(lows, highs)` per query (each a list of `dimensions`
    floats); `meta` is the parsed sidecar (carrying `domain_bounds`, `dimensions`,
    `name`, ...). Raises `WorkloadError` on a missing file, a missing fingerprint
    line, or a header that does not match the dimension count.
    """
    csv_path = Path(csv_path)
    meta_path = Path(meta_path) if meta_path else _metadata_path(csv_path)
    if not csv_path.is_file():
        raise WorkloadError(f"workload CSV not found: {csv_path}")
    if not meta_path.is_file():
        raise WorkloadError(f"workload metadata not found: {meta_path}")

    meta = json.loads(meta_path.read_text())
    dims = int(meta["dimensions"])
    expected = [f"lower_{i}" for i in range(dims)] + [f"upper_{i}" for i in range(dims)]

    rects: list[tuple[list[float], list[float]]] = []
    with csv_path.open(newline="") as fh:
        first = fh.readline()
        if not first.startswith("fingerprint="):
            raise WorkloadError(f"missing fingerprint line in {csv_path}")
        reader = csv.DictReader(fh)
        if reader.fieldnames != expected:
            raise WorkloadError(
                f"unexpected header {reader.fieldnames} in {csv_path}; "
                f"expected {expected}")
        for row in reader:
            lo = [float(row[f"lower_{i}"]) for i in range(dims)]
            hi = [float(row[f"upper_{i}"]) for i in range(dims)]
            rects.append((lo, hi))
    return rects, meta


def _linspace_int(n: int, k: int) -> list[int]:
    """k distinct, evenly-spaced indices in [0, n-1] (waypoints), ends included."""
    if n <= 0:
        return []
    k = max(1, min(k, n))
    if k == 1:
        return [0]
    idx = {round(i * (n - 1) / (k - 1)) for i in range(k)}
    return sorted(idx)


def is_geographic(domain_low, domain_high, dims=(0, 1)) -> bool:
    """True iff the plotted axes fall within valid lon/lat ranges.

    Used to auto-enable the map for geospatial workloads (taxi/ebird): the x
    axis must lie within [-180, 180] and the y axis within [-90, 90]. Non-geo
    domains (synth's 0..1000, gaia's 0..360 right ascension) fail this and get a
    plain plot.
    """
    ax_i, ax_j = dims
    return (-180.0 <= domain_low[ax_i] and domain_high[ax_i] <= 180.0
            and -90.0 <= domain_low[ax_j] and domain_high[ax_j] <= 90.0)


# Light, low-noise tile layers suit a data overlay better than the busy default
# OSM raster. CartoDB Positron is the de-facto choice; keep a few alternatives.
_PROVIDERS = {
    "positron": ("CartoDB", "Positron"),                  # light, subtle labels
    "positron-no-labels": ("CartoDB", "PositronNoLabels"),  # cleanest
    "voyager": ("CartoDB", "Voyager"),                    # light, more detail
    "osm": ("OpenStreetMap", "Mapnik"),                   # the busy default
}


def _provider(ctx, name):
    group, layer = _PROVIDERS[name]
    return getattr(getattr(ctx.providers, group), layer)


def _label_axes_in_degrees(ax) -> None:
    """Format Web-Mercator (EPSG:3857) tick positions back as lon/lat degrees."""
    from matplotlib.ticker import FuncFormatter
    from pyproj import Transformer
    inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{inv.transform(v, 0.0)[0]:.2f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{inv.transform(0.0, v)[1]:.2f}"))


def build_figure(rects, domain_low, domain_high, *, dims=(0, 1),
                 max_queries=None, markers=False, title=None, width="single",
                 basemap="auto", zoom=None, provider="positron", attribution=False):
    """Draw the (first `max_queries`) rectangles over the domain; return (fig, n).

    Each query is an unfilled rectangle and the axes are clamped to the domain.
    `basemap` is "auto" (a light tile background for geographic workloads, plain
    otherwise), "on" (force it), or "off". `provider` picks the tile layer (a
    low-noise light basemap by default); `attribution` shows the small on-figure
    tile credit (off by default -- credit the tile source in the caption, which
    the tile license still requires). When the map is used the
    geometry is reprojected to Web Mercator so the tiles align, the aspect is
    locked equal so shapes are undistorted, and the ticks are relabelled in
    degrees. With `markers`, the first/last/waypoint query centres are overlaid.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    style.apply()
    ax_i, ax_j = dims
    selected = rects if max_queries is None else rects[:max_queries]
    n = len(selected)

    want_map = basemap == "on" or (
        basemap == "auto" and is_geographic(domain_low, domain_high, dims))
    transformer = ctx = None
    if want_map:
        try:
            import contextily as ctx
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        except ImportError:
            print("warning: basemap needs contextily + pyproj "
                  "(pip install contextily); rendering without a map")
            want_map = False
            transformer = ctx = None

    def project(x, y):
        return transformer.transform(x, y) if transformer else (x, y)

    fig, ax = plt.subplots(figsize=style.figure_size(width, aspect=1.0))
    for lo, hi in selected:
        x0, y0 = project(lo[ax_i], lo[ax_j])
        x1, y1 = project(hi[ax_i], hi[ax_j])
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, linewidth=0.4, edgecolor=style.PALETTE["blue"], alpha=0.35))

    if markers and n:
        cen = [project((lo[ax_i] + hi[ax_i]) / 2.0, (lo[ax_j] + hi[ax_j]) / 2.0)
               for lo, hi in selected]
        mx = [p[0] for p in cen]
        my = [p[1] for p in cen]
        way = _linspace_int(n, 5)
        ax.scatter([mx[w] for w in way], [my[w] for w in way], s=8,
                   color=style.PALETTE["orange"], marker="o", zorder=3, label="waypoint")
        ax.scatter([mx[0]], [my[0]], s=28, color=style.PALETTE["green"],
                   marker="*", zorder=4, label="first")
        ax.scatter([mx[-1]], [my[-1]], s=28, color=style.PALETTE["vermilion"],
                   marker="X", zorder=4, label="last")
        ax.legend(loc="best", framealpha=0.6)

    xmin, ymin = project(domain_low[ax_i], domain_low[ax_j])
    xmax, ymax = project(domain_high[ax_i], domain_high[ax_j])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if want_map:
        ax.set_aspect("equal", adjustable="box")
        try:  # fetches tiles for the extent (cached); never aborts the render
            ctx.add_basemap(ax, source=_provider(ctx, provider),
                            crs="EPSG:3857", attribution_size=4,
                            attribution=("" if not attribution else None),
                            zoom=("auto" if zoom is None else zoom))
        except Exception as exc:
            print(f"warning: basemap tile fetch failed: {exc}")
        ax.set_xlim(xmin, xmax)   # re-clip to the dataset domain after the tiles
        ax.set_ylim(ymin, ymax)
        _label_axes_in_degrees(ax)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    else:
        ax.set_xlabel(f"dim {ax_i}")
        ax.set_ylabel(f"dim {ax_j}")
    if title:
        ax.set_title(title)
    return fig, n


def render_to(csv_path: Path, out: Path, *, dims=(0, 1), max_queries=None,
              markers=False, width="single", basemap="auto", zoom=None,
              provider="positron", attribution=False):
    """Render one workload to `out`; return (n_drawn, total). Raises WorkloadError."""
    import matplotlib.pyplot as plt

    rects, meta = load_workload(csv_path)
    d = int(meta["dimensions"])
    if max(dims) >= d or min(dims) < 0:
        raise WorkloadError(
            f"--dims {dims} out of range for the {d}-dimensional workload {csv_path.name}")
    name = meta.get("name", csv_path.stem)
    sel = meta.get("selectivity")
    title = f"{name}  (n={len(rects)}" + (f", sel={sel}" if sel is not None else "") + ")"
    fig, n = build_figure(
        rects, meta["domain_bounds"]["low"], meta["domain_bounds"]["high"],
        dims=dims, max_queries=max_queries, markers=markers, title=title, width=width,
        basemap=basemap, zoom=zoom, provider=provider, attribution=attribution)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return n, len(rects)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Plot a workload's query rectangles over the dataset domain.")
    ap.add_argument("workload", nargs="?",
                    help="path to a workload CSV, or a workload id under the workloads dir")
    ap.add_argument("--all", action="store_true",
                    help="render every workload under the workloads dir to --out-dir")
    ap.add_argument("-o", "--output", default=None,
                    help="output file for a single workload (.pdf/.png/.svg; "
                         "default <name>.pdf in the CWD)")
    ap.add_argument("--out-dir", default="experiments/plots/workloads",
                    help="output directory for --all (default experiments/plots/workloads)")
    ap.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf",
                    help="file format for --all (default pdf)")
    ap.add_argument("--max-queries", type=int, default=None,
                    help="draw only the first N rectangles (sequences are 500-long)")
    ap.add_argument("--markers", action="store_true",
                    help="overlay first/last/waypoint query centres (off by default)")
    ap.add_argument("--dims", default="0,1",
                    help="the two dimension indices to plot (default 0,1)")
    ap.add_argument("--width", choices=["single", "full"], default="single")
    ap.add_argument("--basemap", choices=["auto", "on", "off"], default="auto",
                    help="OpenStreetMap tile background: auto (on for geographic "
                         "workloads like taxi/ebird, off otherwise), on, or off "
                         "(auto/on need contextily)")
    ap.add_argument("--zoom", type=int, default=None,
                    help="basemap tile zoom level (default: auto by extent)")
    ap.add_argument("--basemap-provider", choices=list(_PROVIDERS), default="positron",
                    help="tile layer (default positron: light, low-noise)")
    ap.add_argument("--attribution", action="store_true",
                    help="show the on-figure tile credit (off by default; credit "
                         "the tile source in the caption, as the license requires)")
    args = ap.parse_args(argv)

    if args.all == bool(args.workload):
        ap.error("provide exactly one of: a workload, or --all")
    try:
        dims = tuple(int(x) for x in args.dims.split(","))
    except ValueError:
        ap.error("--dims must be two comma-separated integers, e.g. 0,1")
    if len(dims) != 2:
        ap.error("--dims must name exactly two dimensions, e.g. 0,1")

    common = dict(dims=dims, max_queries=args.max_queries,
                  markers=args.markers, width=args.width,
                  basemap=args.basemap, zoom=args.zoom,
                  provider=args.basemap_provider, attribution=args.attribution)

    if args.all:
        wdir = a3i_config.workloads_dir()
        csvs = sorted(wdir.glob("*.csv"))
        if not csvs:
            print(f"no workloads found under {wdir}")
            return 1
        out_dir = Path(args.out_dir)
        rendered = skipped = 0
        for csv_path in csvs:
            out = out_dir / f"{csv_path.stem}.{args.format}"
            try:
                n, total = render_to(csv_path, out, **common)
            except WorkloadError as exc:
                print(f"skip {csv_path.name}: {exc}")
                skipped += 1
                continue
            print(f"wrote {out} ({n} of {total} rectangles)")
            rendered += 1
        print(f"rendered {rendered} workload(s) to {out_dir}"
              + (f"; skipped {skipped}" if skipped else ""))
        return 0 if rendered else 1

    csv_path = resolve_workload(args.workload)
    out = Path(args.output) if args.output else Path(f"{csv_path.stem}.pdf")
    try:
        n, total = render_to(csv_path, out, **common)
    except WorkloadError as exc:
        ap.error(str(exc))
    print(f"wrote {out} ({n} of {total} rectangles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
