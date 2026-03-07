"""
main.py

Entry point for the Novellia Account Volume & Revenue Forecast dashboard.
Loads account_trend.csv (bundled), then launches the Dash app
with auto browser open at localhost:8050.

To build the Mac executable:
    pyinstaller --onefile --windowed --name NovelliaDashboard \
        --add-data "account_trend.csv:." main.py
"""

import os
import sys
import threading
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc

# ---------------------------------------------------------------------------
# PyInstaller resource path helper
# ---------------------------------------------------------------------------

def _resource_path(relative: str) -> str:
    """
    Return the absolute path to a bundled resource file.

    When frozen by PyInstaller, data files are extracted to sys._MEIPASS.
    When running from source, resolves relative to this file's directory.

    Parameters
    ----------
    relative : str
        Filename relative to the project root (e.g. 'randomized_accounts.csv').

    Returns
    -------
    str : Absolute path safe to pass to pd.read_csv() or open().
    """
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative)


# ---------------------------------------------------------------------------
# Load source data
# ---------------------------------------------------------------------------

randomized_accounts_df = pd.read_csv(_resource_path("account_trend.csv"))
randomized_accounts_df["Account Name"] = randomized_accounts_df["Account Name"].astype(str)

# ---------------------------------------------------------------------------
# Default Parameters
# ---------------------------------------------------------------------------

DEFAULT_BASELINE_TIER_BREAKS = {1: 9, 2: 19, 3: 39, 4: 74, 5: 99,
                                 6: 199, 7: 466}

DEFAULT_TIER_DISCOUNTS = {
    1: 0.05, 2: 0.07, 3: 0.09, 4: 0.11, 5: 0.13,
    6: 0.15, 7: 0.17
}

DEFAULT_RATE_OFFSET           =  0.00   # added to each account's own trend
DEFAULT_UPWARD_RATE           =  0.05
DEFAULT_MIN_VOLUME            =  1.0
DEFAULT_FLOOR_PROXIMITY_PCT   =  0.15
DEFAULT_CEILING_PROXIMITY_PCT =  0.15
DEFAULT_DOWNWARD_HOLD_PROB    =  0.30
DEFAULT_UPWARD_BOOST_PROB     =  0.30
DEFAULT_RANDOM_SEED           =  42

DEFAULT_WAC_2026 = 4500.00
DEFAULT_WAC_2027 = 4675.00

QUARTERS = ["Q1_2026", "Q2_2026", "Q3_2026", "Q4_2026",
            "Q1_2027", "Q2_2027", "Q3_2027", "Q4_2027"]

QUARTER_KEYS   = ["Q4_2025"] + QUARTERS
QUARTER_LABELS = [q.replace("_", " ") for q in QUARTER_KEYS]
QUARTER_ORDER  = {k: i for i, k in enumerate(QUARTER_KEYS)}

PORT = int(os.environ.get("PORT", 8050))

CATEGORY_COLORS = {
    "DownwardMotivated": "#EF5350",
    "UpwardMotivated": "#42A5F5",
    "DoesNotCare":       "#78909C",
}
TIER_LINE_COLOR  = "rgba(255, 213, 79, 0.55)"
TIER_LABEL_COLOR = "#FFD54F"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_year(quarter_key: str) -> int:
    return int(quarter_key.split("_")[1])


# ---------------------------------------------------------------------------
# Forecast Engine
# ---------------------------------------------------------------------------

def build_tier_boundaries(tier_breaks: dict) -> list:
    sorted_tiers = sorted(tier_breaks.items())
    boundaries = []
    for i, (tier, ceiling) in enumerate(sorted_tiers):
        floor = float(sorted_tiers[i - 1][1]) if i > 0 else 0.0
        boundaries.append((tier, float(floor), float(ceiling)))
    #last_tier, last_floor, _ = boundaries[-1]
    #boundaries[-1] = (last_tier, last_floor, float("inf"))
    last_tier, last_floor, last_ceiling = boundaries[-1]
    boundaries.append((last_tier + 1, last_ceiling, float("inf")))
    return boundaries


def assign_tier(volume: float, boundaries: list) -> tuple:
    for tier, floor, ceiling in boundaries:
        if floor <= volume < ceiling:
            return tier, floor, ceiling
    return boundaries[-1]


def resolve_rate(volume, category, boundaries, rng,
                 baseline_rate, upward_rate,
                 floor_proximity_pct, ceiling_proximity_pct,
                 downward_hold_prob, upward_boost_prob):
    if category == "DoesNotCare":
        return baseline_rate
    tier, floor, ceiling = assign_tier(volume, boundaries)
    if category == "DownwardMotivated":
        if volume <= floor * (1.0 + floor_proximity_pct):
            return 0.0 if rng.random() < downward_hold_prob else baseline_rate
        return baseline_rate
    if category == "UpwardMotivated":
        in_zone = (ceiling == float("inf")) or \
                  (volume >= ceiling * (1.0 - ceiling_proximity_pct))
        if in_zone:
            return upward_rate if rng.random() < upward_boost_prob else baseline_rate
        return baseline_rate
    return baseline_rate


def run_forecast(df, tier_breaks, tier_discounts,
                 baseline_rate, upward_rate, min_volume,  # baseline_rate = rate offset
                 floor_proximity_pct, ceiling_proximity_pct,
                 downward_hold_prob, upward_boost_prob,
                 random_seed, wac_2026, wac_2027):
    """
    Forecast volumes, tiers, gross, and net revenue for all accounts.

    Net is calculated as Gross x (1 - tier_discount) for the tier the account
    occupies in that quarter.

    Parameters
    ----------
    df             : Source DataFrame (Account Name, Baseline Volume, Category, Trend)
    tier_breaks    : dict of tier -> upper boundary
    tier_discounts : dict of tier -> discount fraction
    baseline_rate  : Default quarterly rate
    upward_rate    : UpwardlyMotivated boost rate
    min_volume     : Volume floor
    floor_proximity_pct   : DownwardMotivated trigger
    ceiling_proximity_pct : UpwardlyMotivated trigger
    downward_hold_prob    : DownwardMotivated MC probability
    upward_boost_prob     : UpwardlyMotivated MC probability
    random_seed    : RNG seed
    wac_2026       : WAC for 2026 quarters
    wac_2027       : WAC for 2027 quarters

    Returns
    -------
    pd.DataFrame : Wide-format with Volume, Tier, Gross, Net per quarter
    """
    wac_by_year = {2026: wac_2026, 2027: wac_2027}
    boundaries  = build_tier_boundaries(tier_breaks)
    rng         = np.random.default_rng(random_seed)
    records     = []

    for _, row in df.iterrows():
        aggregator    = str(row["Account Name"])
        category      = row["Category"]
        start_vol     = float(row["Baseline Volume"])
        account_trend = float(row["Trend"]) / 100.0   # % -> fraction
        base_tier, _, _ = assign_tier(start_vol, boundaries)
        base_gross = start_vol * wac_by_year[2026]
        base_disc  = tier_discounts.get(base_tier, 0.0)

        record = {
            "Aggregator":      aggregator,
            "Baseline_Volume": start_vol,
            "Baseline_Tier":   base_tier,
            "Category": category,
            "Baseline_Gross":  round(base_gross, 2),
            "Baseline_Net":    round(base_gross * (1 - base_disc), 2),
        }

        current_vol = start_vol
        for quarter in QUARTERS:
            effective_baseline = account_trend + baseline_rate   # trend + offset
            rate    = resolve_rate(current_vol, category, boundaries, rng,
                                   effective_baseline, upward_rate,
                                   floor_proximity_pct, ceiling_proximity_pct,
                                   downward_hold_prob, upward_boost_prob)
            new_vol  = max(current_vol * (1.0 + rate), min_volume)
            new_tier, _, _ = assign_tier(new_vol, boundaries)
            wac      = wac_by_year[get_year(quarter)]
            gross    = new_vol * wac
            discount = tier_discounts.get(new_tier, 0.0)

            record[f"{quarter}_Volume"] = round(new_vol, 4)
            record[f"{quarter}_Tier"]   = new_tier
            record[f"{quarter}_Gross"]  = round(gross, 2)
            record[f"{quarter}_Net"]    = round(gross * (1 - discount), 2)
            current_vol = new_vol

        records.append(record)

    return pd.DataFrame(records)


def build_plot_data(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide forecast_df to long format for plotting."""
    rows = []
    for _, r in forecast_df.iterrows():
        rows.append({
            "Aggregator":      r["Aggregator"],
            "Category": r["Category"],
            "Quarter":         "Q4_2025",
            "QuarterLabel":    "Q4 2025",
            "QuarterOrder":    0,
            "Volume":          float(r["Baseline_Volume"]),
            "Tier":            int(r["Baseline_Tier"]),
            "Gross":           float(r["Baseline_Gross"]),
            "Net":             float(r["Baseline_Net"]),
        })
        for i, qkey in enumerate(QUARTERS, start=1):
            rows.append({
                "Aggregator":      r["Aggregator"],
                "Category": r["Category"],
                "Quarter":         qkey,
                "QuarterLabel":    qkey.replace("_", " "),
                "QuarterOrder":    i,
                "Volume":          float(r[f"{qkey}_Volume"]),
                "Tier":            int(r[f"{qkey}_Tier"]),
                "Gross":           float(r[f"{qkey}_Gross"]),
                "Net":             float(r[f"{qkey}_Net"]),
            })
    return pd.DataFrame(rows).sort_values(
        ["Aggregator", "QuarterOrder"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_volume_figure(plot_data, selected, tier_breaks):
    fig = go.Figure()
    for tier_num, tier_val in sorted(tier_breaks.items()):
        fig.add_hline(
            y=tier_val, line_dash="dot",
            line_color=TIER_LINE_COLOR, line_width=1,
            annotation_text=f"  T{tier_num}->T{tier_num+1}  {tier_val}",
            annotation_position="right",
            annotation_font=dict(size=9, color=TIER_LABEL_COLOR,
                                 family="IBM Plex Mono"),
        )
    if selected:
        subset = plot_data[plot_data["Aggregator"].isin(selected)]
        for aggregator, grp in subset.groupby("Aggregator", sort=False):
            grp      = grp.sort_values("QuarterOrder")
            category = grp["Category"].iloc[0]
            color    = CATEGORY_COLORS.get(category, "#888888")
            fig.add_trace(go.Scatter(
                x=grp["QuarterLabel"].tolist(),
                y=grp["Volume"].tolist(),
                mode="lines+markers", name=aggregator,
                line=dict(color=color, width=1.8),
                marker=dict(size=5, color=color),
                customdata=list(zip(grp["Tier"], grp["Category"])),
                hovertemplate=(
                    "<b>%{meta}</b><br>Quarter: %{x}<br>"
                    "Volume: %{y:.2f}<br>Tier: %{customdata[0]}<br>"
                    "Category: %{customdata[1]}<extra></extra>"
                ),
                meta=aggregator, legendgroup=category,
            ))
    fig.update_layout(
        paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        font=dict(family="IBM Plex Mono", color="#C9D1D9", size=11),
        title=dict(text="Volume Forecast",
                   font=dict(size=12, color="#58A6FF"), x=0),
        xaxis=dict(title="Quarter", categoryorder="array",
                   categoryarray=QUARTER_LABELS,
                   gridcolor="#21262D", linecolor="#30363D",
                   tickfont=dict(size=10)),
        yaxis=dict(title="Volume", type="log", range=[0, 3.3],
                   gridcolor="#21262D", linecolor="#30363D",
                   tickfont=dict(size=10)),
        legend=dict(bgcolor="#161B22", bordercolor="#30363D",
                    borderwidth=1, font=dict(size=10)),
        hovermode="closest",
        margin=dict(l=60, r=160, t=40, b=40),
    )
    return fig


def build_revenue_figure(plot_data, selected, metric: str):
    fig = go.Figure()
    label = metric

    if selected:
        subset = plot_data[plot_data["Aggregator"].isin(selected)]
        for aggregator, grp in subset.groupby("Aggregator", sort=False):
            grp      = grp.sort_values("QuarterOrder")
            category = grp["Category"].iloc[0]
            color    = CATEGORY_COLORS.get(category, "#888888")
            fig.add_trace(go.Scatter(
                x=grp["QuarterLabel"].tolist(),
                y=grp[metric].tolist(),
                mode="lines+markers", name=aggregator,
                line=dict(color=color, width=1.8),
                marker=dict(size=5, color=color),
                customdata=grp["Category"].tolist(),
                hovertemplate=(
                    f"<b>%{{meta}}</b><br>Quarter: %{{x}}<br>"
                    f"{label}: $%{{y:,.2f}}<br>"
                    f"Category: %{{customdata}}<extra></extra>"
                ),
                meta=aggregator, legendgroup=category, showlegend=False,
            ))

    fig.update_layout(
        paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
        font=dict(family="IBM Plex Mono", color="#C9D1D9", size=11),
        title=dict(
            text=f"{'Gross' if metric == 'Gross' else 'Net'} Revenue Forecast"
                 f"{'  (Volume x WAC)' if metric == 'Gross' else '  (Gross x (1 - Tier Discount))'}",
            font=dict(size=12, color="#58A6FF"), x=0,
        ),
        xaxis=dict(title="Quarter", categoryorder="array",
                   categoryarray=QUARTER_LABELS,
                   gridcolor="#21262D", linecolor="#30363D",
                   tickfont=dict(size=10)),
        yaxis=dict(title=f"{label} Revenue ($)", tickformat="$,.0f",
                   gridcolor="#21262D", linecolor="#30363D",
                   tickfont=dict(size=10)),
        hovermode="closest",
        margin=dict(l=80, r=160, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _label(text):
    return html.P(text, style={
        "fontSize": "10px", "color": "#8B949E",
        "letterSpacing": "0.08em", "marginBottom": "3px", "marginTop": "10px",
    })

def _slider(id_, min_val, max_val, step, default):
    marks = {v: {"label": f"{v:.0%}" if max_val <= 1.0 else str(v),
                 "style": {"fontSize": "9px", "color": "#8B949E"}}
             for v in [min_val, (min_val + max_val) / 2, max_val]}
    return dcc.Slider(id=id_, min=min_val, max=max_val, step=step,
                      value=default, marks=marks,
                      tooltip={"placement": "top", "always_visible": False})

def _num_input(id_, value, min_val=None, step=1, width="75px"):
    return dcc.Input(id=id_, type="number", value=value,
                     min=min_val, step=step, debounce=True,
                     style={"width": width, "backgroundColor": "#21262D",
                            "color": "#C9D1D9", "border": "1px solid #30363D",
                            "borderRadius": "3px", "padding": "2px 6px",
                            "fontSize": "11px", "fontFamily": "IBM Plex Mono"})

def _ctrl_btn_style(bg="#21262D"):
    return {"backgroundColor": bg, "color": "#C9D1D9",
            "border": "1px solid #30363D", "borderRadius": "4px",
            "padding": "4px 10px", "cursor": "pointer",
            "fontSize": "10px", "fontFamily": "IBM Plex Mono, monospace",
            "marginRight": "4px"}

def _section(title):
    return html.P(title, style={
        "fontSize": "10px", "color": "#58A6FF",
        "letterSpacing": "0.12em", "marginTop": "16px",
        "marginBottom": "4px", "borderBottom": "1px solid #21262D",
        "paddingBottom": "4px",
    })


def build_layout(aggregators):
    cat_buttons = [
        html.Button(cat, id=f"cat-btn-{cat}", n_clicks=0,
                    style=_ctrl_btn_style(CATEGORY_COLORS[cat]))
        for cat in ["DownwardMotivated", "UpwardMotivated", "DoesNotCare"]
    ]

    tier_inputs = [
        html.Div([
            html.Span(f"T{t}", style={"fontSize": "10px", "color": "#8B949E",
                                       "width": "24px", "display": "inline-block"}),
            _num_input(f"tier-{t}", DEFAULT_BASELINE_TIER_BREAKS[t], min_val=1),
        ], style={"display": "flex", "alignItems": "center",
                  "gap": "8px", "marginBottom": "4px"})
        for t in sorted(DEFAULT_BASELINE_TIER_BREAKS.keys())
    ]

    return html.Div(
        style={"backgroundColor": "#0D1117", "minHeight": "100vh",
               "fontFamily": "IBM Plex Mono, monospace",
               "color": "#C9D1D9", "padding": "24px"},
        children=[
            html.H2("NOVELLIA · CARDIOVEX ACCOUNTS · VOLUME & REVENUE FORECAST",
                    style={"color": "#58A6FF", "letterSpacing": "0.15em",
                           "fontSize": "18px", "marginBottom": "4px"}),
            html.P("Q4 2025 baseline -> Q4 2027  |  Tier breaks overlaid  |  Live parameter tuning",
                   style={"color": "#8B949E", "fontSize": "12px",
                          "marginBottom": "20px"}),

            html.Div(style={"display": "flex", "gap": "16px",
                            "alignItems": "flex-start"},
                children=[

                    # Left: Aggregator selector
                    html.Div(style={
                        "width": "190px", "flexShrink": "0",
                        "backgroundColor": "#161B22",
                        "border": "1px solid #30363D",
                        "borderRadius": "8px", "padding": "14px",
                    }, children=[
                        _section("CATEGORIES"),
                        html.Div(cat_buttons,
                                 style={"display": "flex", "flexWrap": "wrap",
                                        "gap": "4px", "marginBottom": "10px"}),
                        html.Div([
                            html.Button("Select All", id="btn-select-all",
                                        n_clicks=0, style=_ctrl_btn_style()),
                            html.Button("Clear All", id="btn-clear-all",
                                        n_clicks=0, style=_ctrl_btn_style()),
                        ], style={"marginBottom": "10px"}),
                        _section("ACCOUNTS"),
                        dcc.Checklist(
                            id="aggregator-checklist",
                            options=[{"label": a, "value": a} for a in aggregators],
                            value=aggregators[:10],
                            labelStyle={"display": "block", "fontSize": "11px",
                                        "marginBottom": "4px", "cursor": "pointer",
                                        "color": "#C9D1D9"},
                            inputStyle={"marginRight": "6px",
                                        "accentColor": "#58A6FF"},
                            style={"maxHeight": "500px", "overflowY": "auto",
                                   "paddingRight": "4px"},
                        ),
                    ]),

                    # Center: Charts
                    html.Div(style={"flex": "1"}, children=[
                        dcc.Graph(id="forecast-chart",
                                  config={"displayModeBar": False},
                                  style={"height": "680px",
                                         "marginBottom": "12px"}),
                        html.Div([
                            html.Span("Revenue View: ",
                                      style={"fontSize": "11px",
                                             "color": "#8B949E",
                                             "marginRight": "10px"}),
                            dcc.RadioItems(
                                id="revenue-toggle",
                                options=[
                                    {"label": " Gross", "value": "Gross"},
                                    {"label": " Net",   "value": "Net"},
                                ],
                                value="Gross",
                                inline=True,
                                inputStyle={"marginRight": "4px",
                                            "accentColor": "#58A6FF"},
                                labelStyle={"marginRight": "16px",
                                            "fontSize": "11px",
                                            "color": "#C9D1D9",
                                            "cursor": "pointer"},
                            ),
                        ], style={"marginBottom": "8px",
                                  "display": "flex",
                                  "alignItems": "center"}),
                        dcc.Graph(id="revenue-chart",
                                  config={"displayModeBar": False},
                                  style={"height": "380px"}),
                    ]),

                    # Right: Parameters
                    html.Div(style={
                        "width": "210px", "flexShrink": "0",
                        "backgroundColor": "#161B22",
                        "border": "1px solid #30363D",
                        "borderRadius": "8px", "padding": "14px",
                        "maxHeight": "960px", "overflowY": "auto",
                    }, children=[
                        _section("WAC"),
                        _label("2026 WAC"),
                        _num_input("param-wac-2026", DEFAULT_WAC_2026,
                                   min_val=0, step=0.01, width="100px"),
                        _label("2027 WAC"),
                        _num_input("param-wac-2027", DEFAULT_WAC_2027,
                                   min_val=0, step=0.01, width="100px"),

                        _section("VOLUME RATES"),
                        _label("Rate Offset (all accounts)"),
                        _slider("param-baseline-rate", -0.10, 0.10, 0.005,
                                DEFAULT_RATE_OFFSET),
                        _label("Upward Rate (triggered)"),
                        _slider("param-upward-rate", 0.0, 0.20, 0.01,
                                DEFAULT_UPWARD_RATE),
                        _label("Min Volume Floor"),
                        _num_input("param-min-volume", DEFAULT_MIN_VOLUME,
                                   min_val=0.1, step=0.1),

                        _section("DOWNWARD MOTIVATED"),
                        _label("Floor Proximity Trigger"),
                        _slider("param-floor-prox", 0.05, 0.40, 0.01,
                                DEFAULT_FLOOR_PROXIMITY_PCT),
                        _label("Hold-at-Zero Probability"),
                        _slider("param-downward-hold", 0.0, 1.0, 0.05,
                                DEFAULT_DOWNWARD_HOLD_PROB),

                        _section("UPWARDLY MOTIVATED"),
                        _label("Ceiling Proximity Trigger"),
                        _slider("param-ceil-prox", 0.05, 0.40, 0.01,
                                DEFAULT_CEILING_PROXIMITY_PCT),
                        _label("Boost Probability"),
                        _slider("param-upward-boost", 0.0, 1.0, 0.05,
                                DEFAULT_UPWARD_BOOST_PROB),

                        _section("RANDOMNESS"),
                        _label("Random Seed"),
                        _num_input("param-seed", DEFAULT_RANDOM_SEED,
                                   min_val=0, step=1),

                        _section("TIER BREAKS (upper boundary)"),
                        html.Div(tier_inputs),

                        html.Button("↺  Re-run Forecast", id="btn-run",
                                    n_clicks=0, style={
                                        **_ctrl_btn_style("#1F6FEB"),
                                        "width": "100%", "marginTop": "16px",
                                        "padding": "8px", "fontSize": "11px",
                                        "fontWeight": "600",
                                        "letterSpacing": "0.05em",
                                    }),
                        html.Div(id="run-status",
                                 style={"fontSize": "10px", "color": "#3FB950",
                                        "marginTop": "6px",
                                        "textAlign": "center"}),
                    ]),
                ]),
        ]
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(source_df: pd.DataFrame) -> Dash:
    aggregators = sorted(source_df["Account Name"].astype(str).unique().tolist())
    tier_ids    = [f"tier-{t}" for t in sorted(DEFAULT_BASELINE_TIER_BREAKS.keys())]

    app = Dash(__name__, external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap",
    ])
    app.layout = build_layout(aggregators)
    app.layout.children.append(dcc.Store(id="plot-data-store"))

    @app.callback(
        Output("forecast-chart",  "figure"),
        Output("revenue-chart",   "figure"),
        Output("run-status",      "children"),
        Output("plot-data-store", "data"),
        Input("btn-run",              "n_clicks"),
        Input("aggregator-checklist", "value"),
        Input("revenue-toggle",       "value"),
        State("param-wac-2026",       "value"),
        State("param-wac-2027",       "value"),
        State("param-baseline-rate",  "value"),
        State("param-upward-rate",    "value"),
        State("param-min-volume",     "value"),
        State("param-floor-prox",     "value"),
        State("param-ceil-prox",      "value"),
        State("param-downward-hold",  "value"),
        State("param-upward-boost",   "value"),
        State("param-seed",           "value"),
        State("plot-data-store",      "data"),
        *[State(tid, "value") for tid in tier_ids],
    )
    def update(n_clicks, selected, metric,
               wac_2026, wac_2027,
               baseline_rate, upward_rate, min_volume,
               floor_prox, ceil_prox,
               downward_hold, upward_boost,
               seed, stored_data, *tier_vals):

        trigger = ctx.triggered_id

        if trigger == "revenue-toggle" and stored_data is not None:
            plot_data   = pd.DataFrame(stored_data)
            tier_breaks = DEFAULT_BASELINE_TIER_BREAKS
            vol_fig     = build_volume_figure(plot_data, selected or [], tier_breaks)
            rev_fig     = build_revenue_figure(plot_data, selected or [], metric)
            return vol_fig, rev_fig, "", stored_data

        tier_breaks = {
            t: int(v) if v else DEFAULT_BASELINE_TIER_BREAKS[t]
            for t, v in zip(sorted(DEFAULT_BASELINE_TIER_BREAKS.keys()), tier_vals)
        }

        forecast_df = run_forecast(
            df=source_df,
            tier_breaks=tier_breaks,
            tier_discounts=DEFAULT_TIER_DISCOUNTS,
            baseline_rate=float(baseline_rate  if baseline_rate is not None else DEFAULT_RATE_OFFSET),
            upward_rate=float(upward_rate       or DEFAULT_UPWARD_RATE),
            min_volume=float(min_volume         or DEFAULT_MIN_VOLUME),
            floor_proximity_pct=float(floor_prox or DEFAULT_FLOOR_PROXIMITY_PCT),
            ceiling_proximity_pct=float(ceil_prox or DEFAULT_CEILING_PROXIMITY_PCT),
            downward_hold_prob=float(downward_hold or DEFAULT_DOWNWARD_HOLD_PROB),
            upward_boost_prob=float(upward_boost   or DEFAULT_UPWARD_BOOST_PROB),
            random_seed=int(seed) if seed is not None else DEFAULT_RANDOM_SEED,
            wac_2026=float(wac_2026 or DEFAULT_WAC_2026),
            wac_2027=float(wac_2027 or DEFAULT_WAC_2027),
        )

        plot_data  = build_plot_data(forecast_df)
        vol_fig    = build_volume_figure(plot_data, selected or [], tier_breaks)
        rev_fig    = build_revenue_figure(plot_data, selected or [], metric)
        status     = "✓ forecast updated" if trigger == "btn-run" else ""
        return vol_fig, rev_fig, status, plot_data.to_dict("records")

    @app.callback(
        Output("aggregator-checklist", "value"),
        Input("btn-select-all",            "n_clicks"),
        Input("btn-clear-all",             "n_clicks"),
        Input("cat-btn-DownwardMotivated", "n_clicks"),
        Input("cat-btn-UpwardMotivated", "n_clicks"),
        Input("cat-btn-DoesNotCare",       "n_clicks"),
        prevent_initial_call=True,
    )
    def control_selection(*_, _agg=aggregators):
        trigger = ctx.triggered_id
        if trigger == "btn-select-all":
            return _agg
        if trigger == "btn-clear-all":
            return []
        cat_map = {
            "cat-btn-DownwardMotivated": "DownwardMotivated",
            "cat-btn-UpwardMotivated": "UpwardMotivated",
            "cat-btn-DoesNotCare":       "DoesNotCare",
        }
        if trigger in cat_map:
            return source_df[
                source_df["Category"] == cat_map[trigger]
            ]["Account Name"].astype(str).tolist()
        return _agg[:10]

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

app = create_app(randomized_accounts_df, title="Tier Dashboard")
server = app.server  # required for gunicorn

if __name__ == "__main__":
    def open_browser():
        webbrowser.open(f"http://localhost:{PORT}")

    #print(f"Novellia Dashboard running at http://localhost:{PORT}")
    threading.Timer(1.5, open_browser).start()
    app.run(debug=False, port=PORT)
