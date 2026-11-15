"""
Interactive (Plotly) charts for the app: hover for exact values, drag to
zoom, double-click to reset, click a legend entry to hide a series.

Colour is assigned by role and stays fixed everywhere:

  healthy / clog / leak   aqua / orange / blue   (identity)
  below / above baseline  blue / red around grey (polarity)
  affected vessel         red, always with a text label as well
"""

import numpy as np
import plotly.graph_objects as go

LIGHT = dict(
    blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", critical="#d03b3b",
    neutral="#8a8984", ink="#52514e", grid="rgba(0,0,0,0.08)",
    diverging=[[0.0, "#1c5cab"], [0.5, "#f0efec"], [1.0, "#d03b3b"]],
    surface="#ffffff",
)
DARK = dict(
    blue="#3987e5", orange="#d95926", aqua="#199e70", critical="#e66767",
    neutral="#8a8984", ink="#c3c2b7", grid="rgba(255,255,255,0.10)",
    diverging=[[0.0, "#3987e5"], [0.5, "#383835"], [1.0, "#e66767"]],
    surface="#0e1117",
)


def palette(dark=False):
    return DARK if dark else LIGHT


# fixed colour slot per anomaly type, wherever it appears
LABEL_KEYS = {"healthy": "aqua", "clog": "orange", "leak": "blue"}


def label_colors(pal):
    return {label: pal[key] for label, key in LABEL_KEYS.items()}


def _layout(fig, pal, title=None, xtitle=None, ytitle=None, height=340, legend=True):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15), x=0, xanchor="left") if title else None,
        height=height, margin=dict(l=10, r=10, t=44 if title else 16, b=10),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        hoverlabel=dict(font_size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(title=xtitle, gridcolor=pal["grid"], zeroline=False, automargin=True)
    fig.update_yaxes(title=ytitle, gridcolor=pal["grid"], zeroline=False, automargin=True,
                     exponentformat="e")
    return fig


# ---------------------------------------------------------- vessel tree --

def vessel_tree(net, positions, pressures, flows, healthy_flows, mmhg_to_pa,
                highlight_vessel_id=None, scenario="healthy", pal=LIGHT):
    """The vessel network. Line width follows vessel radius; hover a vessel
    for its flow and the change from the healthy baseline, hover a junction
    for its pressure."""
    fig = go.Figure()
    max_r = max((v.radius for v in net.vessels if v.radius), default=1e-3)

    mid_x, mid_y, mid_text = [], [], []
    for ves in net.vessels:
        if ves.v == net.GROUND:
            continue
        x0, y0 = positions[ves.u]
        x1, y1 = positions[ves.v]
        hit = ves.id == highlight_vessel_id
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(width=1.5 + 7 * (ves.radius / max_r if ves.radius else 0),
                      color=pal["critical"] if hit else pal["blue"]),
            hoverinfo="skip", showlegend=False,
        ))
        q, q0 = flows[ves.id], healthy_flows[ves.id]
        change = (q - q0) / q0 * 100 if q0 else 0.0
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        mid_text.append(
            f"<b>Vessel {ves.id}</b>{' (affected)' if hit else ''}<br>"
            f"radius {ves.radius * 1e3:.2f} mm<br>"
            f"flow {q * 1e6:.3f} mL/s<br>"
            f"vs healthy {change:+.1f}%"
        )
    # invisible, generous hover targets at each vessel's midpoint
    fig.add_trace(go.Scatter(
        x=mid_x, y=mid_y, mode="markers", marker=dict(size=18, opacity=0),
        text=mid_text, hovertemplate="%{text}<extra></extra>", showlegend=False,
    ))

    nodes = [n for n in positions if n != net.GROUND]
    fig.add_trace(go.Scatter(
        x=[positions[n][0] for n in nodes], y=[positions[n][1] for n in nodes],
        mode="markers",
        marker=dict(size=9, color=pal["ink"], line=dict(width=2, color=pal["surface"])),
        text=[f"<b>Junction {n}</b>{' (sensor)' if n in net.leaves else ''}<br>"
              f"pressure {pressures[n] / mmhg_to_pa:.1f} mmHg" for n in nodes],
        hovertemplate="%{text}<extra></extra>", showlegend=False,
    ))

    if highlight_vessel_id is not None:
        ves = net.vessels[highlight_vessel_id]
        x0, y0 = positions[ves.u]
        x1, y1 = positions[ves.v]
        mx = (x0 + x1) / 2
        fig.add_annotation(
            x=mx, y=(y0 + y1) / 2, text=f"{scenario} · vessel {ves.id}",
            showarrow=True, arrowhead=0, arrowcolor=pal["neutral"],
            # push the label outward, away from the neighbouring branches
            ax=-90 if mx < 0.5 else 90, ay=-34,
            font=dict(size=12), bgcolor=pal["surface"], borderpad=3,
        )

    _layout(fig, pal, height=480, legend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ------------------------------------------------------------ bar / line --

def delta_bars(labels, deltas, pal, title, ytitle):
    """Signed change per sensor: blue below the baseline, red above."""
    deltas = np.asarray(deltas, dtype=float)
    fig = go.Figure(go.Bar(
        x=[str(l) for l in labels], y=deltas,
        marker=dict(color=[pal["critical"] if d > 0 else pal["blue"] for d in deltas],
                    cornerradius=4),
        hovertemplate="sensor %{x}<br>%{y:+.2f} mmHg<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color=pal["neutral"])
    _layout(fig, pal, title, "Sensor (leaf junction)", ytitle, legend=False)
    fig.update_xaxes(type="category")
    return fig


def horizontal_bars(names, values, pal, title, xtitle):
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=pal["blue"], cornerradius=4),
        hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
    ))
    _layout(fig, pal, title, xtitle, None, height=max(300, 24 * len(names) + 90), legend=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def line_chart(x, series, pal, title, xtitle, ytitle, yrange=None, markers=True, height=340):
    """series: list of (name, y values, colour key). One y axis only."""
    fig = go.Figure()
    for name, y, key in series:
        fig.add_trace(go.Scatter(
            x=x, y=y, name=name, mode="lines+markers" if markers else "lines",
            line=dict(width=2, color=pal[key]), marker=dict(size=7),
            hovertemplate=f"{name}<br>{xtitle}: %{{x:.3g}}<br>%{{y:.4g}}<extra></extra>",
        ))
    _layout(fig, pal, title, xtitle, ytitle, height=height, legend=len(series) > 1)
    if yrange:
        fig.update_yaxes(range=yrange)
    return fig


def roc_curve(fpr, tpr, auc_value, pal):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                             line=dict(width=1, dash="dash", color=pal["neutral"]),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name="detector", line=dict(width=2, color=pal["blue"]),
        hovertemplate="false positive rate %{x:.3f}<br>true positive rate %{y:.3f}<extra></extra>"))
    _layout(fig, pal, f"ROC curve · AUC {auc_value:.3f}", "False positive rate",
            "True positive rate", legend=False)
    return fig


def pr_curve(recall, precision, auc_value, pal):
    fig = go.Figure(go.Scatter(
        x=recall, y=precision, mode="lines", line=dict(width=2, color=pal["blue"]),
        hovertemplate="recall %{x:.3f}<br>precision %{y:.3f}<extra></extra>"))
    _layout(fig, pal, f"Precision-recall · AUC {auc_value:.3f}", "Recall", "Precision",
            legend=False)
    return fig


def threshold_sweep(thresholds, sensitivity, specificity, best, pal):
    fig = line_chart(thresholds, [("sensitivity", sensitivity, "blue"),
                                  ("specificity", specificity, "orange")],
                     pal, "Threshold sweep",
                     "Probability threshold", "Rate", markers=False)
    fig.add_vline(x=best, line_width=1, line_dash="dash", line_color=pal["neutral"],
                  annotation_text=f"chosen {best:.2f}", annotation_position="bottom right")
    return fig


def heatmap(z, x_labels, y_labels, pal, xtitle, ytitle, unit):
    """Signed values around zero: blue negative, grey zero, red positive."""
    z = np.asarray(z)
    lim = float(np.max(np.abs(z))) or 1.0
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(v) for v in x_labels], y=[str(v) for v in y_labels],
        colorscale=pal["diverging"], zmin=-lim, zmax=lim, xgap=1, ygap=1,
        colorbar=dict(title=unit, thickness=12),
        hovertemplate=f"{ytitle} %{{y}}<br>{xtitle} %{{x}}<br>%{{z:.3f}} {unit}<extra></extra>",
    ))
    _layout(fig, pal, None, xtitle, ytitle, height=420, legend=False)
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category", autorange="reversed")
    return fig


# ------------------------------------------------------- distributions --

def histogram(groups, pal, title, xtitle, vline=None, nbins=30):
    """groups: list of (name, values, colour key).

    One group is drawn as bars. Several groups are drawn as step outlines on
    shared bins, each as a share of its own total, so groups of very
    different size can be compared without one hiding the other."""
    fig = go.Figure()
    if len(groups) == 1:
        name, values, key = groups[0]
        fig.add_trace(go.Histogram(
            x=values, name=name, nbinsx=nbins,
            marker=dict(color=pal[key], line=dict(width=1, color=pal["surface"])),
            hovertemplate=f"{xtitle}: %{{x}}<br>%{{y}} patients<extra></extra>",
        ))
        ytitle = "Patients"
    else:
        everything = np.concatenate([np.asarray(v, dtype=float) for _, v, _ in groups])
        edges = np.histogram_bin_edges(everything, bins=min(nbins, 24))
        for name, values, key in groups:
            counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
            share = counts / max(counts.sum(), 1) * 100
            fig.add_trace(go.Scatter(
                x=edges, y=np.append(share, share[-1]), name=name, mode="lines",
                line=dict(width=2, color=pal[key], shape="hv"),
                customdata=np.append(edges[1:], edges[-1]),
                hovertemplate=f"{name}<br>{xtitle}: %{{x:.0f}} to %{{customdata:.0f}}<br>"
                              "%{y:.1f}% of group<extra></extra>",
            ))
        ytitle = "Share of group (%)"
    if vline is not None:
        fig.add_vline(x=vline[0], line_width=1, line_dash="dash", line_color=pal["ink"],
                      annotation_text=vline[1], annotation_position="top")
    _layout(fig, pal, title, xtitle, ytitle, legend=len(groups) > 1)
    fig.update_yaxes(rangemode="tozero")
    return fig


def scatter(x, y, pal, title, xtitle, ytitle, color_by=None, color_map=None,
            identity_line=False, height=380):
    """Scatter, optionally split into at most three coloured groups."""
    fig = go.Figure()
    x = np.asarray(x)
    y = np.asarray(y)
    keys = ["blue", "orange", "aqua"]
    if color_by is None:
        fig.add_trace(go.Scattergl(
            x=x, y=y, mode="markers", marker=dict(size=6, color=pal["blue"], opacity=0.55),
            hovertemplate=f"{xtitle}: %{{x:.4g}}<br>{ytitle}: %{{y:.4g}}<extra></extra>",
            showlegend=False))
    else:
        color_by = np.asarray(color_by).astype(str)
        for i, group in enumerate(sorted(set(color_by))):
            mask = color_by == group
            colour = (color_map or {}).get(group, pal[keys[i % 3]])
            fig.add_trace(go.Scattergl(
                x=x[mask], y=y[mask], mode="markers", name=group,
                marker=dict(size=6, color=colour, opacity=0.55),
                hovertemplate=f"{group}<br>{xtitle}: %{{x:.4g}}<br>{ytitle}: %{{y:.4g}}<extra></extra>"))
    if identity_line:
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", hoverinfo="skip",
                                 line=dict(width=1, dash="dash", color=pal["neutral"]),
                                 showlegend=False))
    _layout(fig, pal, title, xtitle, ytitle, height=height, legend=color_by is not None)
    return fig


def box(values, groups, pal, title, xtitle, ytitle, color_map=None):
    fig = go.Figure()
    groups = np.asarray(groups).astype(str)
    values = np.asarray(values)
    keys = ["blue", "orange", "aqua"]
    names = sorted(set(groups))
    for i, group in enumerate(names):
        colour = (color_map or {}).get(group, pal[keys[i % 3]] if len(names) <= 3 else pal["blue"])
        fig.add_trace(go.Box(y=values[groups == group], name=group, marker_color=colour,
                             boxpoints="outliers", line=dict(width=2)))
    _layout(fig, pal, title, xtitle, ytitle, height=380, legend=False)
    return fig


def count_bars(categories, pal, title, xtitle):
    values, counts = np.unique(np.asarray(categories).astype(str), return_counts=True)
    try:  # numeric-looking categories in numeric order
        order = np.argsort(values.astype(float))
    except ValueError:
        order = np.argsort(-counts)
    fig = go.Figure(go.Bar(
        x=values[order], y=counts[order], marker=dict(color=pal["blue"], cornerradius=4),
        hovertemplate=f"{xtitle}: %{{x}}<br>%{{y}} patients<extra></extra>"))
    _layout(fig, pal, title, xtitle, "Patients", height=380, legend=False)
    fig.update_xaxes(type="category")
    return fig
