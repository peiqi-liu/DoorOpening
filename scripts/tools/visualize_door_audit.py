#!/usr/bin/env python3
"""Render a self-contained HTML report of what the failing doors have in common.

    python scripts/rl_games/play.py ... --door-audit-out logs/door_audit_v6.json
    python scripts/tools/visualize_door_audit.py logs/door_audit_v6.json -o logs/door_audit_v6.html

Sections:
  1. headline success rate
  2. failure anatomy -- where in the pipeline each failure died (grasp / swing / traverse)
  3. how the episode ended, and the success rate of each ending
  4. door parameters ranked by how strongly they separate failure from success
  5. fail-vs-success distributions for the top-ranked parameters

Outcome measures (how far the door swung, how far the base got, which gains were live at the
kill) are reported SEPARATELY from door parameters: they describe what happened, so ranking them
against success is circular. Only the geometry the door was BUILT with can explain anything.

No external dependencies -- inline SVG, works offline, light and dark.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_door_audit import META_FIELDS, dig, derived, rank_biserial  # noqa: E402


# Recorded per episode -- these are RESULTS, not door properties. Never ranked as explanations.
OUTCOME_KEYS = {
    "success", "ref_frame_reached", "hinge_max_rad", "latch_max_rad", "dist_past_door_max_m",
    "episode_steps", "killed_robot_drift", "killed_door_drift", "x5_door_collision",
    "franka_box_collision", "is_push", "asset_idx", "env_id",
    # Rewritten every step by edit_door_articulation (the latch relock), so their value at the
    # kill reflects WHEN the episode ended, not the ADR draw it started with.
    "board_stiffness", "board_damping",
}

PAL = {
    "fail_l": "#eb6834", "fail_d": "#d95926",
    "ok_l": "#2a78d6", "ok_d": "#3987e5",
}


def load(audit_path: Path):
    rows = json.loads(audit_path.read_text())["rows"]
    recs = []
    cache = {}
    for row in rows:
        meta_path = Path(row["asset_path"]).parent / "variant_meta.json"
        if meta_path not in cache:
            cache[meta_path] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta = cache[meta_path]
        rec = dict(row)
        rec["door"] = Path(row["asset_path"]).parent.name
        for label, path in META_FIELDS:
            v = dig(meta, path)
            if isinstance(v, (int, float)):
                rec[label] = float(v)
        rec.update(derived(meta))
        for label, path in [("opening_direction", "actual_properties.opening_direction"),
                            ("handle_type", "handle.type"), ("has_bump", "handle.has_bump")]:
            v = dig(meta, path)
            if v is not None:
                rec[label] = v
        recs.append(rec)
    return recs


def esc(s):
    return html.escape(str(s))


def bar_row(label, value, vmax, color, sublabel="", width=520):
    w = 0 if vmax <= 0 else max(2.0, width * value / vmax)
    return (
        f'<div class="row"><div class="rl">{esc(label)}</div>'
        f'<div class="rb"><span class="bar" style="width:{w:.1f}px;background:{color}" '
        f'title="{esc(label)}: {esc(sublabel or value)}"></span>'
        f'<span class="rv">{esc(sublabel or value)}</span></div></div>'
    )


def diverging_bars(scored, width=260):
    out = ['<div class="dv">']
    vmax = max((abs(s[0]) for s in scored), default=1.0) or 1.0
    for rb, key, fmed, omed in scored:
        w = width * abs(rb) / vmax
        side = "r" if rb > 0 else "l"
        color = "var(--fail)" if rb > 0 else "var(--ok)"
        tip = f"{key}: effect {rb:+.3f} | fail median {fmed:.4g} vs success median {omed:.4g}"
        out.append(
            f'<div class="dvrow" title="{esc(tip)}">'
            f'<div class="dvl">{esc(key)}</div>'
            f'<div class="dvtrack"><span class="dvbar {side}" style="width:{w:.1f}px;background:{color}"></span></div>'
            f'<div class="dvv">{rb:+.2f}</div>'
            f'<div class="dvm">{fmed:.4g} <span class="mut">vs</span> {omed:.4g}</div></div>'
        )
    out.append("</div>")
    return "".join(out)


def histogram(recs, key, nbins=18, w=300, h=92):
    vals = [(r[key], r["success"] > 0.5) for r in recs if isinstance(r.get(key), float)]
    if len(vals) < 8:
        return ""
    lo = min(v for v, _ in vals)
    hi = max(v for v, _ in vals)
    if hi - lo < 1e-12:
        return ""
    fb = [0] * nbins
    ob = [0] * nbins
    for v, ok in vals:
        i = min(nbins - 1, int(nbins * (v - lo) / (hi - lo)))
        (ob if ok else fb)[i] += 1
    # Normalize each group to its own share, so the smaller group stays visible.
    nf = max(1, sum(fb))
    no = max(1, sum(ob))
    fr = [x / nf for x in fb]
    orr = [x / no for x in ob]
    peak = max(max(fr), max(orr)) or 1.0
    bw = w / nbins
    marks = []
    for i in range(nbins):
        x = i * bw
        edge_lo = lo + (hi - lo) * i / nbins
        edge_hi = lo + (hi - lo) * (i + 1) / nbins
        for series, color, cls in ((orr, "var(--ok)", "ok"), (fr, "var(--fail)", "fail")):
            bh = (h - 14) * series[i] / peak
            if bh <= 0.4:
                continue
            marks.append(
                f'<rect class="hb {cls}" x="{x + 1:.1f}" y="{h - 14 - bh:.1f}" width="{bw - 2:.1f}" '
                f'height="{bh:.1f}" rx="2" fill="{color}">'
                f'<title>{esc(key)} {edge_lo:.4g}–{edge_hi:.4g}: '
                f'{fb[i]} fail, {ob[i]} success</title></rect>'
            )
    return (
        f'<figure class="hist"><figcaption>{esc(key)}</figcaption>'
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img" '
        f'aria-label="{esc(key)} distribution, failures versus successes">'
        f'<line x1="0" y1="{h-14}" x2="{w}" y2="{h-14}" stroke="var(--grid)" stroke-width="1"/>'
        + "".join(marks) +
        f'<text x="0" y="{h-2}" class="ax">{lo:.3g}</text>'
        f'<text x="{w}" y="{h-2}" class="ax" text-anchor="end">{hi:.3g}</text>'
        f'</svg></figure>'
    )


def median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def build(recs, title, swing_thresh=0.35, latch_thresh=0.3):
    fails = [r for r in recs if r["success"] <= 0.5]
    oks = [r for r in recs if r["success"] > 0.5]
    n = len(recs)
    rate = 100.0 * len(oks) / n if n else 0.0

    # --- failure anatomy: how far down the pipeline each failure got --------------------
    stages = [
        ("Never turned the handle",
         [r for r in fails if r.get("latch_max_rad", 0.0) < latch_thresh],
         "grasp never established"),
        ("Turned handle, door never swung",
         [r for r in fails if r.get("latch_max_rad", 0.0) >= latch_thresh
          and r.get("hinge_max_rad", 0.0) < swing_thresh],
         "latched but could not move the panel"),
        ("Opened the door, never got through",
         [r for r in fails if r.get("latch_max_rad", 0.0) >= latch_thresh
          and r.get("hinge_max_rad", 0.0) >= swing_thresh],
         "sustained-force / traversal failure"),
    ]
    smax = max((len(g) for _, g, _ in stages), default=1)
    anatomy = "".join(
        bar_row(name, len(grp), smax, "var(--fail)",
                f"{len(grp)}  ({100.0*len(grp)/max(1,len(fails)):.0f}% of failures) — {note}")
        for name, grp, note in stages
    )

    # --- termination reason --------------------------------------------------------------
    reasons = {}
    for r in recs:
        key = r.get("term_reason", "unknown")
        d = reasons.setdefault(key, [0, 0])
        d[0 if r["success"] > 0.5 else 1] += 1
    rmax = max((sum(v) for v in reasons.values()), default=1)
    term = "".join(
        bar_row(k, sum(v), rmax, "var(--ok)" if v[0] else "var(--fail)",
                f"{v[0]}/{sum(v)} success ({100.0*v[0]/max(1,sum(v)):.0f}%)")
        for k, v in sorted(reasons.items(), key=lambda kv: -sum(kv[1]))
    )

    # --- door parameters ranked ------------------------------------------------------------
    param_keys = sorted({k for r in recs for k, v in r.items()
                         if isinstance(v, float) and k not in OUTCOME_KEYS})
    scored = []
    for key in param_keys:
        f = [r[key] for r in fails if isinstance(r.get(key), float)]
        o = [r[key] for r in oks if isinstance(r.get(key), float)]
        if len(f) < 4 or len(o) < 4:
            continue
        if max(f + o) - min(f + o) < 1e-9:  # constant -> nothing to separate
            continue
        rb = rank_biserial(f, o)
        if rb is None:
            continue
        scored.append((rb, key, median(f), median(o)))
    scored.sort(key=lambda s: -abs(s[0]))

    hists = "".join(histogram(recs, key) for _, key, _, _ in scored[:6])

    # --- categoricals -----------------------------------------------------------------------
    cats = []
    for label in ("opening_direction", "handle_type", "has_bump"):
        vals = sorted({str(r.get(label)) for r in recs if label in r})
        if len(vals) < 2:
            continue
        rows = []
        for v in vals:
            grp = [r for r in recs if str(r.get(label)) == v]
            ok = sum(1 for r in grp if r["success"] > 0.5)
            rows.append(bar_row(v, 100.0 * ok / len(grp), 100.0, "var(--ok)",
                                f"{ok}/{len(grp)} = {100.0*ok/len(grp):.0f}%"))
        cats.append(f'<h3>{esc(label)}</h3>' + "".join(rows))

    worst = sorted(fails, key=lambda r: r.get("dist_past_door_max_m", 0.0))[:15]
    wcols = ["door", "latch_max_rad", "hinge_max_rad", "dist_past_door_max_m", "term_reason"]
    wrows = "".join(
        "<tr>" + "".join(
            f"<td>{esc(r.get(c) if not isinstance(r.get(c), float) else f'{r[c]:.3f}')}</td>"
            for c in wcols) + "</tr>"
        for r in worst
    )

    return f"""<div class="viz-root">
<h1>{esc(title)}</h1>
<p class="lede">{n} doors, one episode each.
<strong class="ok-t">{len(oks)} traversed</strong> the doorway
(<strong>{rate:.1f}%</strong>), <strong class="fail-t">{len(fails)} failed</strong>.</p>

<section><h2>1 &middot; Where the failures died</h2>
<p class="note">Each failing episode placed at the furthest stage it reached. A door that
turned the latch and swung open but never got the base through is a <em>sustained-force</em>
failure, not a grasp failure.</p>
{anatomy}</section>

<section><h2>2 &middot; How the episode ended</h2>
<p class="note">Drift kills come from the reference-tracking thresholds; a timeout means the
episode simply ran its course. Bar length = number of episodes, label = success rate within it.</p>
{term}</section>

<section><h2>3 &middot; Door parameters ranked by separation</h2>
<p class="note">Rank-biserial correlation. <span class="fail-t">Positive (orange)</span> = higher
values fail more; <span class="ok-t">negative (blue)</span> = higher values succeed more.
&plusmn;1 is perfect separation, 0 is none. Only parameters the door was <em>built</em> with are
ranked &mdash; outcome measures are excluded, since ranking those against success is circular.
Numbers on the right are fail median vs success median.</p>
{diverging_bars(scored[:14])}</section>

<section><h2>4 &middot; Distributions of the strongest parameters</h2>
<p class="note">Each group is normalised to its own share, so the smaller group stays readable.
Overlapping bars mean the parameter does not separate the two groups.</p>
<div class="hists">{hists}</div>
<div class="legend"><span class="sw" style="background:var(--ok)"></span>traversed
<span class="sw" style="background:var(--fail)"></span>failed</div></section>

<section><h2>5 &middot; Categorical splits</h2><p class="note">Success rate within each group.</p>
{"".join(cats)}</section>

<section><h2>6 &middot; Furthest-from-success doors</h2>
<table><thead><tr>{"".join(f"<th>{esc(c)}</th>" for c in wcols)}</tr></thead>
<tbody>{wrows}</tbody></table></section>
</div>"""


CSS = """
.viz-root{--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--muted:#78776f;
--grid:#e2e1dc;--fail:#eb6834;--ok:#2a78d6;color-scheme:light;background:var(--surface-1);
color:var(--text-primary);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
padding:28px 22px;max-width:1000px;margin:0 auto}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])) .viz-root{
--surface-1:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;--muted:#9a998f;--grid:#34332f;
--fail:#d95926;--ok:#3987e5;color-scheme:dark}}
:root[data-theme=dark] .viz-root{--surface-1:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;
--muted:#9a998f;--grid:#34332f;--fail:#d95926;--ok:#3987e5;color-scheme:dark}
.viz-root h1{font-size:1.5rem;margin:0 0 .3em;letter-spacing:-.01em}
.viz-root h2{font-size:1.02rem;margin:0 0 .5em;letter-spacing:.01em}
.viz-root h3{font-size:.86rem;margin:1.1em 0 .4em;color:var(--text-secondary);font-weight:600}
.lede{color:var(--text-secondary);margin:0 0 1.6em}
.note{color:var(--muted);font-size:.83rem;margin:0 0 1em;max-width:74ch}
section{margin:0 0 2.3em;padding-top:1.1em;border-top:1px solid var(--grid)}
.ok-t{color:var(--ok)}.fail-t{color:var(--fail)}
.row{display:flex;align-items:center;gap:12px;margin:5px 0}
.rl{flex:0 0 240px;font-size:.83rem;color:var(--text-secondary);text-align:right}
.rb{flex:1;display:flex;align-items:center;gap:9px;min-width:0}
.bar{height:13px;border-radius:0 4px 4px 0;flex:none}
.rv{font-size:.78rem;color:var(--muted);white-space:nowrap}
.dv{display:flex;flex-direction:column;gap:3px}
.dvrow{display:flex;align-items:center;gap:10px}
.dvl{flex:0 0 190px;font-size:.8rem;text-align:right;color:var(--text-secondary)}
.dvtrack{flex:0 0 540px;height:14px;position:relative;border-left:2px solid var(--grid);
display:flex;justify-content:center}
.dvbar{position:absolute;height:13px;top:0}
.dvbar.r{left:50%;border-radius:0 4px 4px 0}.dvbar.l{right:50%;border-radius:4px 0 0 4px}
.dvv{flex:0 0 52px;font-size:.78rem;font-variant-numeric:tabular-nums;color:var(--text-primary)}
.dvm{font-size:.76rem;color:var(--muted);font-variant-numeric:tabular-nums}
.mut{color:var(--grid)}
.hists{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}
.hist{margin:0}.hist figcaption{font-size:.8rem;color:var(--text-secondary);margin-bottom:3px}
.hb{fill-opacity:.85}.hb.fail{fill-opacity:.75}
.ax{font-size:9px;fill:var(--muted)}
.legend{margin-top:12px;font-size:.8rem;color:var(--text-secondary);display:flex;gap:8px;align-items:center}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block;margin-left:10px}
table{border-collapse:collapse;font-size:.79rem;width:100%;display:block;overflow-x:auto}
th,td{text-align:left;padding:5px 11px 5px 0;border-bottom:1px solid var(--grid);white-space:nowrap}
th{color:var(--muted);font-weight:600}
td{font-variant-numeric:tabular-nums}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audit_json", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    recs = load(args.audit_json)
    title = args.title or f"Door failure audit — {args.audit_json.stem}"
    body = build(recs, title)
    out = args.out or args.audit_json.with_suffix(".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body>{body}</body></html>",
        encoding="utf-8",
    )
    print(f"Wrote {out}  ({len(recs)} doors)")


if __name__ == "__main__":
    main()
