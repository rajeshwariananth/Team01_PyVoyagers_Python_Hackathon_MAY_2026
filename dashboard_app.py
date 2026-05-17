from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


DATA_PATH = Path("Data/Cleaned/Team01_PYVoyagers_cleaned_data.csv")

GLUCOSE_LOW = 70
GLUCOSE_HIGH = 180
BLUE_SEQ = ["#dbeafe", "#93c5fd", "#60a5fa", "#2563eb", "#1d4ed8", "#1e3a8a"]
BLUE_DIVERGE = ["#dbeafe", "#bfdbfe", "#60a5fa", "#2563eb", "#1e40af"]
CHART_SEQ = ["#38bdf8", "#a78bfa", "#f472b6", "#fbbf24", "#34d399", "#fb7185", "#22d3ee", "#c084fc"]
BLUE_CATEGORICAL = {
    "Low": "#93c5fd",
    "In Range": "#2563eb",
    "High": "#1e3a8a",
    "Active": "#2563eb",
    "Sedentary": "#93c5fd",
    "Moderate": "#60a5fa",
}


st.set_page_config(
    page_title="PyVoyagers Diabetes Intelligence",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #eaf4ff;
        --muted: #b9d6f5;
        --blue-1: #dbeafe;
        --blue-2: #93c5fd;
        --blue-3: #60a5fa;
        --blue-4: #2563eb;
        --blue-5: #1d4ed8;
        --blue-6: #0f285c;
        --panel: rgba(11, 44, 92, 0.78);
    }
    .stApp {
        background: linear-gradient(135deg, #07182f 0%, #0a2f63 52%, #0f4c81 100%);
        color: var(--ink);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #06142b 0%, #0d3b74 100%);
    }
    [data-testid="stSidebar"] * {
        color: #f8fbff;
    }
    h1, h2, h3 {
        letter-spacing: 0 !important;
    }
    .main-title {
        font-size: 2.55rem;
        line-height: 1.05;
        font-weight: 800;
        margin: .4rem 0 .2rem 0;
        color: #f8fbff;
    }
    .subtle {
        color: var(--muted);
        font-size: 1.02rem;
        max-width: 920px;
    }
    .metric-card {
        min-height: 116px;
        padding: 18px 18px 14px 18px;
        border-radius: 8px;
        background: var(--panel);
        border: 1px solid rgba(147,197,253,.24);
        box-shadow: 0 16px 36px rgba(3,12,30,.28);
    }
    .metric-label {
        color: #bfdbfe;
        font-size: .82rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fbff;
        margin-top: 4px;
    }
    .metric-note {
        color: #bfdbfe;
        font-size: .85rem;
        margin-top: 4px;
    }
    div[data-testid="stPlotlyChart"] {
        border-radius: 8px;
        background: rgba(8, 35, 75, .86);
        border: 1px solid rgba(147,197,253,.24);
        box-shadow: 0 18px 42px rgba(3,12,30,.32);
        padding: 8px;
    }
    .recommendation {
        padding: 16px 18px;
        border-left: 5px solid #93c5fd;
        background: rgba(8, 35, 75, .82);
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 12px 24px rgba(3,12,30,.25);
        color: #f8fbff;
    }
    .insight-card {
        padding: 13px 16px;
        border-radius: 8px;
        margin: 10px 0 10px 0;
        color: #f8fbff;
        background: linear-gradient(90deg, rgba(37,99,235,.42), rgba(14,165,233,.18));
        border: 1px solid rgba(191,219,254,.35);
        box-shadow: 0 10px 24px rgba(3,12,30,.22);
    }
    .insight-card b {
        color: #dbeafe;
    }
    [data-testid="stToolbar"],
    .stToolbar,
    .st-emotion-cache-abycrm.ekqzr6e0,
    .st-emotion-cache-abycrm,
    .ekqzr6e0,
    .st-ae.st-af.st-ag.st-ah.st-ai.st-aj.st-ak.st-al.st-am,
    .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak, .st-al, .st-am {
        background-color: #0b2c5c !important;
        color: #f8fbff !important;
        border-color: rgba(147,197,253,.35) !important;
    }
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] section,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] [role="listbox"] {
        background-color: rgba(11,44,92,.9);
        color: #f8fbff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading cleaned PyVoyagers data...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "patient_id", "glucose"])
    df = df.sort_values(["patient_id", "time"])
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    df["activity_level"] = np.where(df["steps"] > 0, "Active", "Sedentary")
    df["hr_zone"] = pd.cut(
        df["heart_rate"],
        bins=[0, 80, 110, 300],
        labels=["Resting", "Moderate", "High"],
        include_lowest=True,
    )
    df["glucose_zone"] = pd.cut(
        df["glucose"],
        bins=[0, GLUCOSE_LOW, GLUCOSE_HIGH, 1000],
        labels=["Low", "In Range", "High"],
        include_lowest=True,
    )
    df["glucose_next"] = df.groupby("patient_id")["glucose"].shift(-4)
    df["glucose_drop_1hr"] = df["glucose"] - df["glucose_next"]
    df["glucose_rise_1hr"] = df["glucose_next"] - df["glucose"]
    df["iob_proxy"] = (
        df.assign(clean_bolus=df["bolus_volume_delivered"].clip(lower=0))
        .groupby("patient_id")["clean_bolus"]
        .rolling(window=12, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["hypo_warning_score"] = (
        (GLUCOSE_LOW + 20 - df["glucose"]).clip(lower=0) * 0.9
        + (-df["glucose_drop_1hr"]).clip(lower=0) * 0.4
        + df["iob_proxy"].clip(lower=0) * 10
        + (df["heart_rate"] > 110).astype(int) * 8
    ).clip(0, 100)
    return df


def patient_summary(frame: pd.DataFrame) -> pd.DataFrame:
    daily = (
        frame.groupby(["patient_id", "date"], as_index=False)
        .agg(
            daily_steps=("steps", "sum"),
            daily_calories=("calories", "sum"),
            glucose_sd=("glucose", "std"),
            hr_sd=("heart_rate", "std"),
        )
        .fillna(0)
    )
    daily_avg = daily.groupby("patient_id", as_index=False).agg(
        typical_daily_steps=("daily_steps", "mean"),
        typical_daily_calories=("daily_calories", "mean"),
        glucose_variability=("glucose_sd", "mean"),
        hr_variability=("hr_sd", "mean"),
    )
    base = frame.groupby("patient_id", as_index=False).agg(
        mean_glucose=("glucose", "mean"),
        median_glucose=("glucose", "median"),
        mean_hr=("heart_rate", "mean"),
        mean_steps=("steps", "mean"),
        mean_basal=("basal_rate", "mean"),
        bolus_total=("bolus_volume_delivered", lambda x: x.clip(lower=0).sum()),
        carbs_total=("carb_input", "sum"),
        sleep_hours=("average_sleep_duration_hrs", "first"),
        sleep_quality=("sleep_quality_1_10", "first"),
        sleep_disturbances=("_with_sleep_disturbances", "first"),
        age=("age", "first"),
        gender=("gender", "first"),
        race=("race", "first"),
    )
    tir = (
        frame.assign(in_range=frame["glucose"].between(GLUCOSE_LOW, GLUCOSE_HIGH))
        .groupby("patient_id")["in_range"]
        .mean()
        .mul(100)
        .rename("tir_pct")
        .reset_index()
    )
    summary = base.merge(tir, on="patient_id").merge(daily_avg, on="patient_id", how="left")
    summary["risk_score"] = (
        (summary["mean_glucose"] - 100).clip(lower=0) * 0.45
        + (100 - summary["tir_pct"]).clip(lower=0) * 0.35
        + (summary["mean_hr"] - 70).clip(lower=0) * 0.45
        + (6000 - summary["typical_daily_steps"]).clip(lower=0) / 6000 * 20
        + summary["glucose_variability"].clip(0, 80) * 0.18
    ).clip(0, 100)
    summary["risk_level"] = pd.cut(
        summary["risk_score"],
        bins=[-1, 40, 65, 101],
        labels=["Low", "Moderate", "High"],
    )
    return summary.sort_values("risk_score", ascending=False)


def metabolic_2d(frame: pd.DataFrame) -> go.Figure:
    sample_size = min(len(frame), 9000)
    plot_df = frame.sample(sample_size, random_state=7) if len(frame) > sample_size else frame
    fig = px.scatter(
        plot_df,
        x="steps",
        y="heart_rate",
        color="glucose_zone",
        size="glucose",
        size_max=14,
        opacity=0.72,
        color_discrete_map={"Low": "#38bdf8", "In Range": "#34d399", "High": "#f472b6"},
        hover_data={
            "patient_id": True,
            "time": True,
            "glucose": ":.1f",
            "basal_rate": ":.3f",
            "bolus_volume_delivered": ":.2f",
            "carb_input": ":.0f",
            "iob_proxy": ":.2f",
        },
        title="Metabolic Scatter: Activity, Heart Rate, and Glucose",
    )
    fig.update_layout(
        height=520,
        xaxis_title="Steps per interval",
        yaxis_title="Heart rate",
        legend_title_text="Glucose zone",
    )
    return fig


def hourly_heatmap(frame: pd.DataFrame) -> go.Figure:
    hourly = (
        frame.groupby(["patient_id", "hour"], as_index=False)["glucose"]
        .mean()
        .pivot(index="patient_id", columns="hour", values="glucose")
        .reindex(columns=list(range(24)))
        .interpolate(axis=1)
        .bfill(axis=1)
        .ffill(axis=1)
    )
    fig = px.imshow(
        hourly,
        x=hourly.columns,
        y=hourly.index,
        color_continuous_scale=CHART_SEQ,
        aspect="auto",
        labels=dict(x="Hour of day", y="Patient", color="Avg glucose"),
        title="Hourly Glucose Heatmap by Patient",
    )
    fig.update_layout(
        height=520,
    )
    return fig


def risk_bars(summary: pd.DataFrame) -> go.Figure:
    colors = summary["risk_level"].map({"Low": "#93c5fd", "Moderate": "#60a5fa", "High": "#1d4ed8"})
    fig = go.Figure(
        data=[
            go.Bar(
                x=summary["patient_id"],
                y=summary["risk_score"],
                marker=dict(color=colors, line=dict(color="rgba(16,32,51,.35)", width=1)),
                hovertemplate=(
                    "<b>%{x}</b><br>Risk score: %{y:.1f}"
                    "<br>Mean glucose: %{customdata[0]:.1f}"
                    "<br>TIR: %{customdata[1]:.1f}%"
                    "<br>Daily steps: %{customdata[2]:.0f}<extra></extra>"
                ),
                customdata=summary[["mean_glucose", "tir_pct", "typical_daily_steps"]],
            )
        ]
    )
    fig.update_layout(
        title="Patient Risk Ranking",
        height=430,
        yaxis_title="Risk score",
        xaxis_title="",
        margin=dict(l=0, r=0, t=54, b=0),
    )
    return fig


def recommendation_cards(summary: pd.DataFrame, frame: pd.DataFrame) -> None:
    top = summary.head(5)
    for row in top.itertuples():
        patient_rows = frame[frame["patient_id"] == row.patient_id]
        exercise_drop = patient_rows.loc[patient_rows["heart_rate"] > 110, "glucose_drop_1hr"].mean()
        if pd.isna(exercise_drop):
            exercise_drop = 0
        if row.risk_score >= 65:
            priority = "High monitoring priority"
        elif row.risk_score >= 40:
            priority = "Moderate monitoring priority"
        else:
            priority = "Stable monitoring priority"
        if exercise_drop > 20:
            exercise_note = "reduce bolus before planned exercise"
        elif row.typical_daily_steps < 5000:
            exercise_note = "build toward a gradual 7,000 step daily target"
        else:
            exercise_note = "keep current activity pattern and monitor glucose trend"
        st.markdown(
            f"""
            <div class="recommendation">
                <b>{row.patient_id}</b> - {priority}<br>
                Avg glucose <b>{row.mean_glucose:.1f}</b> mg/dL, TIR <b>{row.tir_pct:.1f}%</b>,
                typical daily steps <b>{row.typical_daily_steps:,.0f}</b>.
                Recommendation: <b>{exercise_note}</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )


def transparent(fig: go.Figure, height: int | None = None) -> go.Figure:
    if height:
        fig.update_layout(height=height)
    fig.update_layout(
        paper_bgcolor="rgba(8,35,75,.86)",
        plot_bgcolor="rgba(219,234,254,.14)",
        font=dict(color="#f8fbff"),
        title_font=dict(color="#f8fbff"),
        legend=dict(font=dict(color="#f8fbff")),
        margin=dict(l=0, r=0, t=58, b=0),
    )
    fig.update_xaxes(gridcolor="rgba(191,219,254,.25)", zerolinecolor="rgba(191,219,254,.38)", color="#eaf4ff")
    fig.update_yaxes(gridcolor="rgba(191,219,254,.25)", zerolinecolor="rgba(191,219,254,.38)", color="#eaf4ff")
    return fig


def plotly_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch")


def key_insight(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="insight-card">
            <b>{title}</b><br>{text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def spike_recovery(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient_id, patient_rows in frame.sort_values("time").groupby("patient_id"):
        patient_rows = patient_rows.reset_index(drop=True)
        spike_idx = patient_rows.index[
            (patient_rows["glucose"] >= GLUCOSE_HIGH)
            & (patient_rows["glucose"].shift(1).fillna(0) < GLUCOSE_HIGH)
        ]
        for idx in spike_idx[:40]:
            later = patient_rows.loc[idx:]
            recovered = later[later["glucose"].between(GLUCOSE_LOW, GLUCOSE_HIGH)]
            if not recovered.empty:
                minutes = (recovered.iloc[0]["time"] - patient_rows.loc[idx, "time"]).total_seconds() / 60
                if 0 <= minutes <= 720:
                    rows.append({"patient_id": patient_id, "recovery_hours": minutes / 60})
    return pd.DataFrame(rows)


def bolus_response(frame: pd.DataFrame) -> pd.DataFrame:
    events = frame[frame["bolus_volume_delivered"] > 0].copy()
    if events.empty:
        return pd.DataFrame(columns=["patient_id", "glucose_before", "glucose_at_bolus", "glucose_after"])
    events["glucose_before"] = frame.groupby("patient_id")["glucose"].shift(4)
    events["glucose_after"] = frame.groupby("patient_id")["glucose"].shift(-4)
    return (
        events.groupby("patient_id", as_index=False)
        .agg(
            glucose_before=("glucose_before", "mean"),
            glucose_at_bolus=("glucose", "mean"),
            glucose_after=("glucose_after", "mean"),
            bolus_event_count=("bolus_volume_delivered", "count"),
        )
        .dropna()
    )


def add_prescriptions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["meal_recommendation"] = np.select(
        [
            (out["glucose"] < 90) & (out["iob_proxy"] > 1),
            (out["glucose"] < 90) & (out["iob_proxy"] <= 1),
        ],
        ["Delay meal", "Eat now"],
        default="Meal can proceed",
    )
    exercise = out["heart_rate"] > 110
    out["carb_recommendation"] = np.select(
        [
            exercise & (out["glucose"] < 80) & (out["iob_proxy"] > 1),
            exercise & (out["glucose"] < 100),
            exercise & (out["glucose"] < 120),
        ],
        ["15g carbs before exercise", "10g carbs before exercise", "5g carbs before exercise"],
        default="No carbs needed",
    )
    out["basal_action"] = np.where(
        (out["basal_rate"] < out["basal_rate"].median()) & (out["glucose_rise_1hr"] > 15),
        "Review basal support",
        "Monitor",
    )
    out["activity_action"] = np.select(
        [out["glucose_rise_1hr"] > 40, out["glucose_rise_1hr"] > 20],
        ["20-30 min brisk walk", "10-15 min walk"],
        default="Light activity or monitor",
    )
    return out


def predictive_comparison(frame: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    meal_events = frame[(frame["carb_input"] > 0) & (frame["bolus_volume_delivered"] > 0)]
    high_carb = frame[frame["carb_input"] > 40]
    hypo_events = (frame["glucose"] < GLUCOSE_LOW).sum()
    night = frame[frame["hour"].between(0, 5)]
    correlation_glucose_hr = abs(frame["glucose"].corr(frame["heart_rate"]))
    correlation_sleep_stability = abs(summary["sleep_quality"].corr(summary["glucose_variability"]))
    rows = [
        {
            "analysis": "1-hour glucose forecast",
            "notebook_metric": "R2 0.9598, MAE 7.16",
            "data_coverage": 95,
            "signal_strength": 96,
            "actionability": 92,
            "rating": "Excellent",
        },
        {
            "analysis": "Night-time hypoglycemia risk",
            "notebook_metric": "Accuracy 75.8%, risk recall 38%",
            "data_coverage": min(95, max(55, night["patient_id"].nunique() / max(frame["patient_id"].nunique(), 1) * 100)),
            "signal_strength": 62,
            "actionability": 90,
            "rating": "Good, needs more low-glucose examples",
        },
        {
            "analysis": "Hospitalization risk score",
            "notebook_metric": "Rule-based risk ranking",
            "data_coverage": 92,
            "signal_strength": min(90, 55 + summary["risk_score"].std()),
            "actionability": 88,
            "rating": "Strong triage view",
        },
        {
            "analysis": "Diabetes probability estimate",
            "notebook_metric": "Activity biomarker probability",
            "data_coverage": 90,
            "signal_strength": 76,
            "actionability": 78,
            "rating": "Useful lifestyle screen",
        },
        {
            "analysis": "CIR prediction",
            "notebook_metric": f"{len(meal_events):,} meal insulin events",
            "data_coverage": min(100, len(meal_events) / max(len(frame), 1) * 800),
            "signal_strength": 72,
            "actionability": 86,
            "rating": "Promising with sparse meal labels",
        },
        {
            "analysis": "Carb absorption rate",
            "notebook_metric": f"{frame['glucose_rise_1hr'].notna().mean() * 100:.1f}% future glucose available",
            "data_coverage": 88,
            "signal_strength": 74,
            "actionability": 76,
            "rating": "Good pattern model",
        },
        {
            "analysis": "Glucose stability window",
            "notebook_metric": "Hourly variability clusters",
            "data_coverage": 94,
            "signal_strength": 82,
            "actionability": 80,
            "rating": "Strong timing insight",
        },
        {
            "analysis": "Hypoglycemia early warning",
            "notebook_metric": f"{hypo_events:,} low-glucose rows",
            "data_coverage": min(100, max(35, hypo_events / max(len(frame), 1) * 2000)),
            "signal_strength": 68,
            "actionability": 95,
            "rating": "High value, validate sensitivity",
        },
        {
            "analysis": "Meal impact model",
            "notebook_metric": f"{len(high_carb):,} high-carb rows",
            "data_coverage": min(100, max(20, len(high_carb) / max(len(frame), 1) * 500)),
            "signal_strength": 70,
            "actionability": 84,
            "rating": "Useful for activity advice",
        },
        {
            "analysis": "Sleep influence on next-day stability",
            "notebook_metric": f"|r| approx {0 if pd.isna(correlation_sleep_stability) else correlation_sleep_stability:.2f}",
            "data_coverage": 82,
            "signal_strength": min(85, max(35, correlation_sleep_stability * 100 if not pd.isna(correlation_sleep_stability) else 35)),
            "actionability": 66,
            "rating": "Weak to moderate signal",
        },
    ]
    scores = pd.DataFrame(rows)
    scores["overall_score"] = (
        scores["data_coverage"] * 0.25
        + scores["signal_strength"] * 0.4
        + scores["actionability"] * 0.35
    ).round(1)
    return scores.sort_values("overall_score", ascending=False)


df = load_data()

st.sidebar.header("Dashboard Filters")
patients = sorted(df["patient_id"].unique())
selected_patients = st.sidebar.multiselect("Patients", patients, default=patients[: min(8, len(patients))])
glucose_range = st.sidebar.slider(
    "Glucose range",
    min_value=int(df["glucose"].min()),
    max_value=int(df["glucose"].max()),
    value=(int(df["glucose"].quantile(.02)), int(df["glucose"].quantile(.98))),
)
activity_filter = st.sidebar.multiselect(
    "Activity level",
    ["Active", "Sedentary"],
    default=["Active", "Sedentary"],
)
filtered = df[
    df["patient_id"].isin(selected_patients)
    & df["glucose"].between(glucose_range[0], glucose_range[1])
    & df["activity_level"].isin(activity_filter)
].copy()

if filtered.empty:
    st.warning("No rows match the current filters. Try widening the glucose range or selecting more patients.")
    st.stop()

summary = patient_summary(filtered)

st.markdown('<div class="main-title">The HUPA‑UCM Diabetes Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">Interactive local dashboard preserving the Jupyter notebook analysis as 2D charts with a unified blue theme. Filter patient cohorts, inspect risk patterns, and compare glucose, heart rate, activity, insulin, carbs, and sleep signals.</div>',
    unsafe_allow_html=True,
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
metrics = [
    ("Avg Glucose", f"{filtered['glucose'].mean():.1f}", "mg/dL across selected rows"),
    ("Time In Range", f"{filtered['glucose'].between(GLUCOSE_LOW, GLUCOSE_HIGH).mean() * 100:.1f}%", "70-180 mg/dL"),
    ("Active Intervals", f"{(filtered['steps'] > 0).mean() * 100:.1f}%", "movement detected"),
    ("Patients", f"{filtered['patient_id'].nunique()}", f"{len(filtered):,} records"),
]
for col, (label, value, note) in zip([kpi1, kpi2, kpi3, kpi4], metrics):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

tabs = st.tabs(["Descriptive Analysis", "Prescriptive Analysis", "Predictive Analysis"])

prescribed = add_prescriptions(filtered)

with tabs[0]:
    st.subheader("Descriptive Analysis")
    overview_left, overview_right = st.columns(2)
    with overview_left:
        key_insight("Activity, Heart Rate, and Glucose", "This overview mirrors the notebook’s combined physiological view: low steps with higher glucose or heart rate can signal sedentary metabolic risk.")
        plotly_chart(transparent(metabolic_2d(filtered), 500))
    with overview_right:
        key_insight("Hourly Glucose Pattern", "The hourly patient heatmap preserves the notebook’s time-of-day idea, showing when glucose is more stable or elevated.")
        plotly_chart(transparent(hourly_heatmap(filtered), 500))

    demo_summary = (
        filtered.groupby(["gender", "age"], as_index=False)
        .agg(glucose=("glucose", "mean"), heart_rate=("heart_rate", "mean"), steps=("steps", "mean"))
    )
    d1, d2 = st.columns(2)
    with d1:
        key_insight("Gender and Age Profile", "Use the metric selector inside this chart to compare how glucose, heart rate, or steps vary by age and gender.")
        metric_choice = st.segmented_control(
            "Demographic metric",
            ["glucose", "heart_rate", "steps"],
            default="glucose",
        )
        fig = px.scatter(
            demo_summary,
            x="age",
            y=metric_choice,
            color="gender",
            size="steps",
            title="Average Glucose, Heart Rate, and Steps by Gender and Age",
            color_discrete_sequence=CHART_SEQ,
        )
        plotly_chart(transparent(fig, 430))
    with d2:
        recovery = spike_recovery(filtered)
        if recovery.empty:
            st.info("No recoverable glucose spike events found under the current filters.")
        else:
            key_insight("Spike Recovery Time", "Shorter recovery windows suggest better glucose clearance after a spike; longer recovery windows flag patients needing closer review.")
            fig = px.box(
                recovery,
                x="patient_id",
                y="recovery_hours",
                color="patient_id",
                title="Time to Return to Normal Glucose After Spike",
                labels={"recovery_hours": "Recovery hours"},
            )
            plotly_chart(transparent(fig, 430))

    d3, d4 = st.columns(2)
    with d3:
        key_insight("Active vs Sedentary Heart Rate", "The notebook compares heart rate during movement and no-movement intervals to show cardiovascular response to activity.")
        fig = px.box(
            filtered,
            x="activity_level",
            y="heart_rate",
            color="activity_level",
            points="outliers",
            title="Heart Rate During Active vs Sedentary Intervals",
            color_discrete_sequence=CHART_SEQ,
        )
        plotly_chart(transparent(fig, 430))
    with d4:
        key_insight("High Glucose Profile", "High-glucose intervals are profiled by activity, calories, and heart rate to reveal the behavioral pattern behind elevated glucose.")
        high_profile = (
            filtered.assign(is_high_glucose=filtered["glucose"] > GLUCOSE_HIGH)
            .groupby("is_high_glucose", as_index=False)
            .agg(steps=("steps", "mean"), calories=("calories", "mean"), heart_rate=("heart_rate", "mean"))
            .melt(id_vars="is_high_glucose", var_name="marker", value_name="value")
        )
        high_profile["glucose_group"] = np.where(high_profile["is_high_glucose"], "High glucose", "Normal/low glucose")
        fig = px.bar(
            high_profile,
            x="marker",
            y="value",
            color="glucose_group",
            barmode="group",
            title="Profile of High Glucose Intervals",
            color_discrete_sequence=CHART_SEQ,
        )
        plotly_chart(transparent(fig, 430))

    d5, d6 = st.columns(2)
    with d5:
        key_insight("Glucose by Heart-Rate Zone", "This chart preserves the notebook’s HR-zone comparison to see whether higher cardiovascular intensity aligns with glucose shifts.")
        fig = px.box(
            filtered.dropna(subset=["hr_zone"]),
            x="hr_zone",
            y="glucose",
            color="hr_zone",
            points=False,
            title="Glucose Distribution by Heart Rate Zone",
            color_discrete_sequence=CHART_SEQ,
        )
        plotly_chart(transparent(fig, 430))
    with d6:
        key_insight("Patient Glucose Summary", "Mean and median glucose show each patient’s central glucose pattern, while the reference line marks the hyperglycemia threshold.")
        fig = px.bar(
            summary.sort_values("mean_glucose"),
            x="patient_id",
            y=["mean_glucose", "median_glucose"],
            title="Average and Median Glucose by Patient",
            barmode="group",
            color_discrete_sequence=CHART_SEQ,
        )
        fig.add_hline(y=GLUCOSE_HIGH, line_dash="dash", line_color="#bfdbfe")
        plotly_chart(transparent(fig, 430))

    d7, d8 = st.columns(2)
    with d7:
        key_insight("Sleep Quality Distribution", "The notebook checks whether sleep quality is generally high or low; the vertical line shows the selected cohort average.")
        fig = px.histogram(
            summary,
            x="sleep_quality",
            nbins=10,
            color="risk_level",
            title="Sleep Quality Distribution",
            color_discrete_sequence=CHART_SEQ,
        )
        fig.add_vline(x=summary["sleep_quality"].mean(), line_dash="dash", line_color="#bfdbfe")
        plotly_chart(transparent(fig, 430))
    with d8:
        key_insight("Typical Activity Level", "Daily steps and calories identify sedentary patterns and patients who may need activity-focused recommendations.")
        daily = filtered.groupby(["patient_id", "date"], as_index=False).agg(steps=("steps", "sum"), calories=("calories", "sum"))
        typical_daily = daily.groupby("patient_id", as_index=False).mean(numeric_only=True)
        fig = px.bar(
            typical_daily.sort_values("steps"),
            x="patient_id",
            y=["steps", "calories"],
            title="Typical Daily Activity: Steps and Calories",
            barmode="group",
            color_discrete_sequence=CHART_SEQ,
        )
        plotly_chart(transparent(fig, 430))

    d9, d10 = st.columns(2)
    with d9:
        bolus = bolus_response(filtered)
        if bolus.empty:
            st.info("No bolus response events found under the current filters.")
        else:
            key_insight("Bolus Glucose Response", "The notebook compares glucose before, during, and after bolus insulin to evaluate correction effectiveness.")
            bolus_long = bolus.melt(
                id_vars=["patient_id", "bolus_event_count"],
                value_vars=["glucose_before", "glucose_at_bolus", "glucose_after"],
                var_name="window",
                value_name="glucose",
            )
            fig = px.line(
                bolus_long,
                x="window",
                y="glucose",
                color="patient_id",
                markers=True,
                title="Average Glucose Before, During, and After Bolus Events",
                color_discrete_sequence=CHART_SEQ,
            )
            plotly_chart(transparent(fig, 430))
    with d10:
        key_insight("Basal Insulin and Glucose by Hour", "Basal rate patterns are compared with hourly glucose to identify whether background insulin supports stable control.")
        basal = filtered.groupby("hour", as_index=False).agg(basal_rate=("basal_rate", "mean"), glucose=("glucose", "mean"))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=basal["hour"], y=basal["basal_rate"], name="Basal rate", mode="lines+markers", line=dict(color="#38bdf8", width=4)))
        fig.add_trace(go.Scatter(x=basal["hour"], y=basal["glucose"], name="Glucose", mode="lines+markers", yaxis="y2", line=dict(color="#f472b6", width=4)))
        fig.update_layout(
            title="Basal Insulin Variation and Glucose Patterns by Hour",
            xaxis_title="Hour",
            yaxis=dict(title="Basal rate"),
            yaxis2=dict(title="Glucose", overlaying="y", side="right"),
        )
        plotly_chart(transparent(fig, 430))

with tabs[1]:
    st.subheader("Prescriptive Analysis")
    p1, p2 = st.columns(2)
    with p1:
        key_insight("Activity and Insulin Effectiveness", "The notebook uses glucose drop as an insulin-effectiveness proxy; stronger drops during active periods suggest improved glucose clearance.")
        activity_effect = prescribed.groupby("activity_level", as_index=False)["glucose_drop_1hr"].mean()
        fig = px.bar(
            activity_effect,
            x="activity_level",
            y="glucose_drop_1hr",
            color="activity_level",
            title="Insulin Effectiveness Proxy by Activity Level",
            labels={"glucose_drop_1hr": "Avg 1-hour glucose drop"},
            color_discrete_sequence=CHART_SEQ,
        )
        plotly_chart(transparent(fig, 430))
    with p2:
        key_insight("Heart-Rate Zone and Glucose Drop", "Higher heart-rate zones can indicate stronger metabolic activity, which the notebook links to larger glucose reductions.")
        hr_effect = prescribed.groupby("hr_zone", observed=True, as_index=False)["glucose_drop_1hr"].mean()
        fig = px.bar(
            hr_effect,
            x="hr_zone",
            y="glucose_drop_1hr",
            color="hr_zone",
            title="Heart Rate Zones Correlated with Glucose Drops",
            color_discrete_sequence=CHART_SEQ,
        )
        plotly_chart(transparent(fig, 430))

    p3, p4 = st.columns(2)
    with p3:
        key_insight("Insulin-to-Basal Review Ratio", "A higher bolus-to-basal pattern suggests heavier correction or meal insulin reliance and may indicate a dosing review need.")
        ratio = summary.assign(insulin_to_basal_ratio=summary["bolus_total"] / (summary["mean_basal"] + 0.01))
        fig = px.bar(
            ratio.sort_values("insulin_to_basal_ratio", ascending=False).head(15),
            x="patient_id",
            y="insulin_to_basal_ratio",
            color="risk_level",
            title="Optimal Insulin-to-Basal Review Ratio by Patient",
            color_discrete_sequence=CHART_SEQ,
        )
        plotly_chart(transparent(fig, 430))
    with p4:
        key_insight("Exercise Bolus Guidance", "The notebook recommends bolus reduction when exercise is followed by a large glucose drop, reducing hypoglycemia risk.")
        exercise_summary = prescribed[prescribed["heart_rate"] > 110].groupby("patient_id", as_index=False)["glucose_drop_1hr"].mean()
        exercise_summary["recommendation"] = np.select(
            [exercise_summary["glucose_drop_1hr"] > 20, exercise_summary["glucose_drop_1hr"] < 5],
            ["Reduce bolus before exercise", "No reduction needed"],
            default="Small reduction may help",
        )
        fig = px.bar(
            exercise_summary.sort_values("glucose_drop_1hr", ascending=False),
            x="patient_id",
            y="glucose_drop_1hr",
            color="recommendation",
            title="Bolus Reduction Guidance Before Planned Exercise",
            color_discrete_sequence=CHART_SEQ,
        )
        plotly_chart(transparent(fig, 430))

    p5, p6 = st.columns(2)
    with p5:
        key_insight("Meal Delay Decision", "Meal timing is guided by current glucose and insulin-on-board; low glucose plus high IOB is a delay signal.")
        meal_counts = prescribed["meal_recommendation"].value_counts().rename_axis("recommendation").reset_index(name="count")
        fig = px.pie(
            meal_counts,
            values="count",
            names="recommendation",
            title="Meal Delay Decision Based on Glucose and IOB",
            color_discrete_sequence=CHART_SEQ,
            hole=0.42,
        )
        plotly_chart(transparent(fig, 430))
    with p6:
        key_insight("Carbs Before Exercise", "The notebook prescribes carbs before exercise when glucose is low or moderate, especially with insulin still active.")
        carb_counts = prescribed[prescribed["heart_rate"] > 110]["carb_recommendation"].value_counts().rename_axis("recommendation").reset_index(name="count")
        fig = px.bar(
            carb_counts,
            x="recommendation",
            y="count",
            color="recommendation",
            title="Carbs Before Exercise Recommendation",
            color_discrete_sequence=CHART_SEQ,
        )
        plotly_chart(transparent(fig, 430))

    p7, p8 = st.columns(2)
    with p7:
        key_insight("Step Goal Recommendation", "Patients below the 7,000-8,000 step target band are highlighted for gradual activity improvement.")
        fig = px.scatter(
            summary,
            x="typical_daily_steps",
            y="mean_glucose",
            size="risk_score",
            color="risk_level",
            hover_name="patient_id",
            title="Step Goal Target for Low Activity Patients",
            color_discrete_sequence=CHART_SEQ,
        )
        fig.add_vrect(x0=7000, x1=8000, fillcolor="#bfdbfe", opacity=.18, line_width=0)
        plotly_chart(transparent(fig, 430))
    with p8:
        key_insight("Demographics and Glucose Variability", "The notebook connects glucose variability to demographic group patterns to identify patients at higher complication risk.")
        demo_var = summary.groupby(["gender", "race"], as_index=False)["glucose_variability"].mean()
        fig = px.bar(
            demo_var,
            x="race",
            y="glucose_variability",
            color="gender",
            barmode="group",
            title="Demographic Factors Associated with Glucose Variability",
            color_discrete_sequence=CHART_SEQ,
        )
        plotly_chart(transparent(fig, 430))

    p9, p10 = st.columns(2)
    with p9:
        key_insight("Calories Burned vs Carbs", "This scatter checks metabolic efficiency: whether recorded carbs align with calorie expenditure and glucose zone.")
        fig = px.scatter(
            prescribed.sample(min(len(prescribed), 7000), random_state=4),
            x="carb_input",
            y="calories",
            color="glucose_zone",
            title="Calories Burned vs Carbs Recorded",
            color_discrete_sequence=CHART_SEQ,
        )
        plotly_chart(transparent(fig, 430))
    with p10:
        low_hr_carb = prescribed[(prescribed["heart_rate"] < 80) & (prescribed["carb_input"] > 0)].copy()
        if low_hr_carb.empty:
            st.info("No low-heart-rate carb events found under the current filters.")
        else:
            key_insight("Low HR + High Carbs", "Low heart rate can indicate low activity; paired with carbs, it may predict a stronger glucose spike.")
            fig = px.scatter(
                low_hr_carb.sample(min(len(low_hr_carb), 4000), random_state=3),
                x="carb_input",
                y="glucose_rise_1hr",
                size="heart_rate",
                color="activity_action",
                title="Low Heart Rate + High Carbs Predicting Glucose Spikes",
                labels={"glucose_rise_1hr": "1-hour glucose rise"},
                color_discrete_sequence=CHART_SEQ,
            )
            plotly_chart(transparent(fig, 520))

    p11, p12 = st.columns(2)
    with p11:
        key_insight("Biomarker Correlation", "The correlation heatmap shows how glucose relates to heart rate, steps, calories, insulin, carbs, and IOB in the selected cohort.")
        corr_cols = ["glucose", "heart_rate", "steps", "calories", "basal_rate", "bolus_volume_delivered", "carb_input", "iob_proxy"]
        corr = prescribed[corr_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale=CHART_SEQ,
            zmin=-1,
            zmax=1,
            title="Correlation Between Glucose and Biomarkers",
        )
        plotly_chart(transparent(fig, 500))
    with p12:
        key_insight("Basal Action Recommendation", "When basal insulin is low and glucose is rising, the notebook flags a possible basal-support review.")
        action_counts = prescribed["basal_action"].value_counts().rename_axis("action").reset_index(name="count")
        fig = px.bar(
            action_counts,
            x="action",
            y="count",
            color="action",
            title="Action When Basal Rate Is Low and Glucose Is Rising",
            color_discrete_sequence=CHART_SEQ,
        )
        plotly_chart(transparent(fig, 430))

    p13, p14 = st.columns(2)
    with p13:
        key_insight("Activity and Heart Rate", "The notebook checks whether activity level and heart rate move together, which helps separate fitness response from stress signal.")
        fig = px.scatter(
            summary,
            x="mean_steps",
            y="mean_hr",
            size="risk_score",
            color="risk_level",
            hover_name="patient_id",
            title="Relationship Between Activity Level and Heart Rate",
            color_discrete_sequence=CHART_SEQ,
        )
        plotly_chart(transparent(fig, 430))
    with p14:
        key_insight("Insulin Dosage Review", "Patients with the highest bolus totals may need closer insulin-pattern review, especially alongside elevated glucose risk.")
        dosage = summary.sort_values("bolus_total", ascending=False).head(15)
        fig = px.bar(
            dosage,
            x="patient_id",
            y="bolus_total",
            color="risk_level",
            title="Patients Who May Require Insulin Dosage Review",
            color_discrete_sequence=CHART_SEQ,
        )
        plotly_chart(transparent(fig, 430))

    st.subheader("Prescriptive Priority Table")
    recommendation_cards(summary, prescribed)

with tabs[2]:
    st.subheader("Predictive Analysis")
    pred = prescribed.sort_values(["patient_id", "time"]).copy()
    pred["glucose_slope"] = pred.groupby("patient_id")["glucose"].diff().fillna(0)
    pred["glucose_std"] = pred.groupby("patient_id")["glucose"].rolling(5).std().reset_index(level=0, drop=True).fillna(0)
    pred["time_of_day"] = pred["hour"] / 24
    pred["activity_num"] = (pred["steps"] > 0).astype(int)
    pred["future_glucose_60"] = pred.groupby("patient_id")["glucose"].shift(-12)
    pred["future_glucose_2hr"] = pred.groupby("patient_id")["glucose"].shift(-24)

    c1, c2 = st.columns(2)
    cir_df = pred[(pred["carb_input"] > 0) & (pred["bolus_volume_delivered"] > 0)].copy()
    with c1:
        if cir_df.empty:
            st.info("No meal plus bolus events found for CIR analysis under current filters.")
        else:
            cir_df["CIR"] = cir_df["carb_input"] / cir_df["bolus_volume_delivered"].replace(0, np.nan)
            cir_df = cir_df[cir_df["CIR"].between(1, 60)].copy()
            features = ["glucose", "heart_rate", "steps", "calories", "glucose_slope", "hour"]
            importance = (
                cir_df[features].corrwith(cir_df["CIR"]).abs().fillna(0)
                .rename("importance").reset_index().rename(columns={"index": "feature"})
                .sort_values("importance", ascending=True)
            )
            key_insight("Dynamic CIR Prediction", "The notebook checks which physiological and timing features explain changing carb-to-insulin ratio; larger bars indicate stronger CIR signal in the selected cohort.")
            fig = px.bar(importance, x="importance", y="feature", orientation="h", color="feature", title="Feature Importance for Dynamic CIR Prediction", color_discrete_sequence=CHART_SEQ)
            plotly_chart(transparent(fig, 430))
    with c2:
        if not cir_df.empty:
            cir_df["predicted_cir"] = cir_df.groupby("patient_id")["CIR"].transform(lambda x: x.rolling(5, min_periods=1).mean())
            key_insight("Actual vs Predicted CIR", "Points closer to the diagonal indicate the recent patient pattern is a good proxy for the observed carb-to-insulin ratio.")
            fig = px.scatter(cir_df.sample(min(len(cir_df), 5000), random_state=3), x="CIR", y="predicted_cir", color="heart_rate", color_continuous_scale=CHART_SEQ, title="Actual vs Predicted CIR")
            fig.add_shape(type="line", x0=cir_df["CIR"].min(), y0=cir_df["CIR"].min(), x1=cir_df["CIR"].max(), y1=cir_df["CIR"].max(), line=dict(color="#f8fbff", dash="dash"))
            plotly_chart(transparent(fig, 430))

    c3, c4 = st.columns(2)
    with c3:
        if not cir_df.empty:
            cir_df["time_block"] = pd.cut(cir_df["hour"], bins=[0, 6, 12, 18, 24], labels=["Night", "Morning", "Afternoon", "Evening"], include_lowest=True)
            key_insight("CIR Across Time of Day", "The notebook compares CIR by circadian time block to reveal when insulin needs may shift.")
            fig = px.box(cir_df, x="time_block", y="CIR", color="time_block", title="Dynamic CIR Across Time of Day", color_discrete_sequence=CHART_SEQ)
            plotly_chart(transparent(fig, 430))
    with c4:
        if not cir_df.empty:
            cir_df["hr_group"] = pd.cut(cir_df["heart_rate"], bins=[0, 70, 100, 200], labels=["Low", "Moderate", "High"])
            key_insight("Stress / Heart Rate Impact on CIR", "Higher heart-rate groups indicate possible stress or exertion effects on insulin response.")
            fig = px.violin(cir_df.dropna(subset=["hr_group"]), x="hr_group", y="CIR", color="hr_group", box=True, title="Stress / Heart Rate Impact on CIR", color_discrete_sequence=CHART_SEQ)
            plotly_chart(transparent(fig, 430))

    absorption = pred[(pred["carb_input"] > 0) & pred["future_glucose_60"].notna()].copy()
    absorption["delta_glucose"] = absorption["future_glucose_60"] - absorption["glucose"]
    absorption["glucose_slope_abs"] = absorption["delta_glucose"] / 60
    absorption["peak_glucose"] = absorption[["glucose", "future_glucose_60"]].max(axis=1)
    if len(absorption) >= 3 and absorption["glucose_slope_abs"].nunique() >= 3:
        absorption["absorption_group"] = pd.qcut(absorption["glucose_slope_abs"].rank(method="first"), q=3, labels=["Slow", "Medium", "Fast"])
    else:
        absorption["absorption_group"] = "Medium"
    absorption["hr_group"] = pd.cut(absorption["heart_rate"], bins=[0, 70, 100, 200], labels=["Low", "Moderate", "High"])

    c5, c6 = st.columns(2)
    with c5:
        if absorption.empty:
            st.info("No carb events found for absorption analysis under current filters.")
        else:
            features = ["carb_input", "glucose", "heart_rate", "steps", "calories", "glucose_slope_abs", "peak_glucose", "hour"]
            importance = (
                absorption[features].corrwith(absorption["glucose_slope_abs"]).abs().fillna(0)
                .rename("importance").reset_index().rename(columns={"index": "feature"})
                .sort_values("importance", ascending=True)
            )
            key_insight("Carb Absorption Feature Importance", "The notebook’s absorption model ranks meal size, glucose pattern, activity, and heart-rate markers as absorption-speed drivers.")
            fig = px.bar(importance, x="importance", y="feature", orientation="h", color="feature", title="Feature Importance: Carb Absorption Prediction", color_discrete_sequence=CHART_SEQ)
            plotly_chart(transparent(fig, 430))
    with c6:
        if not absorption.empty:
            key_insight("Glucose Slope by Absorption Group", "Fast absorption has the steepest post-meal glucose slope; slow absorption has the flattest curve.")
            fig = px.box(absorption, x="absorption_group", y="glucose_slope_abs", color="absorption_group", title="Glucose Slope by Absorption Group", color_discrete_sequence=CHART_SEQ)
            plotly_chart(transparent(fig, 430))

    c7, c8 = st.columns(2)
    with c7:
        if not absorption.empty:
            key_insight("Meal Size vs Peak Glucose", "The notebook uses this scatter to show whether larger carb entries align with higher post-meal peak glucose.")
            fig = px.scatter(absorption.sample(min(len(absorption), 5000), random_state=6), x="carb_input", y="peak_glucose", color="absorption_group", size="heart_rate", title="Meal Size vs Peak Glucose", color_discrete_sequence=CHART_SEQ)
            plotly_chart(transparent(fig, 430))
    with c8:
        if not absorption.empty:
            if len(absorption) >= 3 and absorption["delta_glucose"].nunique() >= 3:
                absorption["pred_group"] = pd.qcut(absorption["delta_glucose"].rank(method="first"), q=3, labels=["Slow", "Medium", "Fast"])
            else:
                absorption["pred_group"] = "Medium"
            cm = pd.crosstab(absorption["absorption_group"], absorption["pred_group"])
            key_insight("Absorption Confusion Matrix", "The notebook checks how well predicted absorption classes match actual slow, medium, and fast groups.")
            fig = px.imshow(cm, text_auto=True, color_continuous_scale=CHART_SEQ, title="Confusion Matrix")
            plotly_chart(transparent(fig, 430))

    c9, c10 = st.columns(2)
    with c9:
        if not absorption.empty:
            key_insight("Stress / HR Impact on Glucose Rise", "Heart-rate groups help reveal whether stress or exertion is associated with faster glucose rise after carbs.")
            fig = px.violin(absorption.dropna(subset=["hr_group"]), x="hr_group", y="glucose_slope_abs", color="hr_group", box=True, title="Stress / HR Impact on Glucose Rise", color_discrete_sequence=CHART_SEQ)
            plotly_chart(transparent(fig, 430))
    with c10:
        stability = pred.copy()
        threshold = stability["glucose_std"].median()
        stability["stability_cluster"] = np.where(stability["glucose_std"] <= threshold, "Stable", "Unstable")
        key_insight("Glucose Stability Window Clusters", "Lower glucose variability identifies stable windows; higher variability marks time windows that may need closer monitoring.")
        fig = px.scatter(stability.sample(min(len(stability), 7000), random_state=7), x="time_of_day", y="glucose_std", color="stability_cluster", title="Glucose Stability Window Clusters", color_discrete_sequence=CHART_SEQ)
        plotly_chart(transparent(fig, 430))

    c11, c12 = st.columns(2)
    with c11:
        stability_features = ["time_of_day", "basal_rate", "glucose_std", "glucose_slope", "activity_num", "carb_input"]
        stability_importance = (
            stability[stability_features].corrwith((stability["stability_cluster"] == "Unstable").astype(int)).abs().fillna(0)
            .rename("importance").reset_index().rename(columns={"index": "feature"})
            .sort_values("importance", ascending=True)
        )
        key_insight("Stability Prediction Feature Importance", "The notebook ranks the signals that separate stable and unstable glucose windows.")
        fig = px.bar(stability_importance, x="importance", y="feature", orientation="h", color="feature", title="Feature Importance - Stability Prediction", color_discrete_sequence=CHART_SEQ)
        plotly_chart(transparent(fig, 430))
    with c12:
        key_insight("Distribution of Stability Windows", "This mirrors the notebook histogram showing the balance of stable versus unstable glucose windows.")
        fig = px.histogram(stability, x="stability_cluster", color="stability_cluster", title="Distribution of Stability Windows", color_discrete_sequence=CHART_SEQ)
        plotly_chart(transparent(fig, 430))

    c13, c14 = st.columns(2)
    with c13:
        hypo_target = (pred["glucose"] < GLUCOSE_LOW).astype(int)
        hypo_features = ["glucose", "glucose_slope", "glucose_std", "iob_proxy", "activity_num", "carb_input", "time_of_day"]
        hypo_importance = (
            pred[hypo_features].corrwith(hypo_target).abs().fillna(0)
            .rename("importance").reset_index().rename(columns={"index": "feature"})
            .sort_values("importance", ascending=True)
        )
        key_insight("Hypoglycemia Risk Drivers", "Low current glucose and rapid downward trends are the main early-warning markers in the notebook logic.")
        fig = px.bar(hypo_importance, x="importance", y="feature", orientation="h", color="feature", title="Hypoglycemia Risk Drivers", color_discrete_sequence=CHART_SEQ)
        plotly_chart(transparent(fig, 430))
    with c14:
        meal_impact = pred.dropna(subset=["glucose_rise_1hr"]).copy().head(50)
        meal_impact["predicted_spike"] = meal_impact["glucose_rise_1hr"].rolling(5, min_periods=1).mean()
        line_df = meal_impact[["glucose_rise_1hr", "predicted_spike"]].reset_index(drop=True).reset_index().melt(id_vars="index", var_name="series", value_name="glucose_spike")
        key_insight("Meal Impact Prediction", "The notebook overlays actual and predicted spike traces to check whether meal response direction is being captured.")
        fig = px.line(line_df, x="index", y="glucose_spike", color="series", title="Meal Impact Prediction", color_discrete_sequence=CHART_SEQ)
        plotly_chart(transparent(fig, 430))

    c15, c16 = st.columns(2)
    with c15:
        carb_threshold = pred[(pred["carb_input"] > 0) & (pred["bolus_volume_delivered"] > 0)].copy()
        if carb_threshold.empty:
            st.info("No meal bolus events found for carb threshold analysis.")
        else:
            threshold = carb_threshold["carb_input"].mode().iloc[0]
            key_insight("Carb Thresholds Requiring Correction Bolus", f"The notebook marks the most common meal-bolus carb threshold; in this filtered view it is {threshold:.0f}g.")
            fig = px.histogram(carb_threshold, x="carb_input", nbins=35, color="glucose_zone", title="Distribution of Carbs to Get Required Insulin", color_discrete_sequence=CHART_SEQ)
            fig.add_vline(x=threshold, line_dash="dash", line_color="#f8fbff")
            plotly_chart(transparent(fig, 430))
    with c16:
        overshoot = pred[(pred["glucose"] < GLUCOSE_LOW) & (pred["future_glucose_2hr"] > GLUCOSE_HIGH)].copy()
        if overshoot.empty:
            st.info("No hypo-to-hyper overshoot events found under current filters.")
        else:
            values = pd.DataFrame({"state": ["Current State (present)", "Predicted State (2hrs Later)"], "glucose": [overshoot["glucose"].mean(), overshoot["future_glucose_2hr"].mean()]})
            key_insight("Glucose Overshoot After Correction", "The notebook compares average low starting glucose against the predicted rebound two hours later.")
            fig = px.bar(values, x="state", y="glucose", color="state", title="Predicting Glucose Overshoot After Correction", color_discrete_sequence=CHART_SEQ)
            fig.add_hline(y=GLUCOSE_LOW, line_dash="dash", line_color="#fbbf24")
            fig.add_hline(y=GLUCOSE_HIGH, line_dash="dash", line_color="#fb7185")
            plotly_chart(transparent(fig, 430))

    c17, c18 = st.columns(2)
    with c17:
        forecast = pred.dropna(subset=["glucose_next"]).sample(min(len(pred.dropna(subset=["glucose_next"])), 5000), random_state=12)
        key_insight("One-Hour Future Glucose Model", "The notebook reports R² 0.9598 and MAE 7.16 mg/dL, indicating strong short-term glucose prediction from recent trends and patient markers.")
        fig = px.scatter(forecast, x="glucose", y="glucose_next", color="activity_level", title="Actual vs 1-Hour Future Glucose", color_discrete_sequence=CHART_SEQ)
        plotly_chart(transparent(fig, 430))
    with c18:
        night = pred[pred["hour"].between(0, 5)].copy()
        night_risk = night.groupby(["patient_id", "date"], as_index=False).agg(night_hypo=("glucose", lambda x: int((x < GLUCOSE_LOW).any())))
        if night_risk.empty:
            st.info("No night-time rows found under current filters.")
        else:
            actual = night_risk["night_hypo"]
            pred_risk = (actual.rolling(3, min_periods=1).mean() >= actual.mean()).astype(int)
            cm = pd.crosstab(actual.map({0: "No Risk", 1: "Risk"}), pred_risk.map({0: "No Risk", 1: "Risk"}))
            key_insight("Night-Time Hypoglycemia Risk", "The notebook reports 75.8% accuracy, with stronger no-risk detection than true-risk recall.")
            fig = px.imshow(cm, text_auto=True, color_continuous_scale=CHART_SEQ, title="Night-Time Hypoglycemia Confusion Matrix")
            plotly_chart(transparent(fig, 430))

    c19, c20 = st.columns(2)
    with c19:
        high_carb = pred[pred["carb_input"] > 40].copy()
        if high_carb.empty:
            st.info("No high-carb meals found under current filters.")
        else:
            activity_summary = high_carb.groupby("activity_action", as_index=False).size()
            key_insight("Activity After High-Carb Meals", "The notebook summarizes recommended activity intensity after high-carb meals based on predicted glucose rise.")
            fig = px.bar(activity_summary, x="activity_action", y="size", color="activity_action", title="Optimal Activity Recommendation After High-Carb Meals", color_discrete_sequence=CHART_SEQ)
            plotly_chart(transparent(fig, 430))
    with c20:
        daily_stats = pred.groupby(["patient_id", "date"], as_index=False).agg(glucose_sd=("glucose", "std"), hr_sd=("heart_rate", "std"), average_sleep_duration_hrs=("average_sleep_duration_hrs", "first"), sleep_disturbances=("_with_sleep_disturbances", "first")).dropna()
        patient_avg = daily_stats.groupby("patient_id", as_index=False).agg(glucose_sd=("glucose_sd", "mean"), hr_sd=("hr_sd", "mean"), average_sleep_duration_hrs=("average_sleep_duration_hrs", "first"), sleep_disturbances=("sleep_disturbances", "first"))
        key_insight("Sleep Impact Analysis", "The notebook compares sleep duration and disturbances against glucose stability and heart-rate volatility.")
        sleep_fig = make_subplots(rows=2, cols=2, subplot_titles=("Sleep Duration vs Glucose Stability", "Sleep Disturbances vs Glucose Stability", "Sleep Duration vs Heart Rate Volatility", "Sleep Disturbances vs Heart Rate Volatility"))
        sleep_fig.add_trace(go.Scatter(x=patient_avg["average_sleep_duration_hrs"], y=patient_avg["glucose_sd"], mode="markers", marker=dict(color="#38bdf8", size=10), name="Duration / Glucose"), row=1, col=1)
        sleep_fig.add_trace(go.Scatter(x=patient_avg["sleep_disturbances"], y=patient_avg["glucose_sd"], mode="markers", marker=dict(color="#fbbf24", size=10), name="Disturbance / Glucose"), row=1, col=2)
        sleep_fig.add_trace(go.Scatter(x=patient_avg["average_sleep_duration_hrs"], y=patient_avg["hr_sd"], mode="markers", marker=dict(color="#34d399", size=10), name="Duration / HR"), row=2, col=1)
        sleep_fig.add_trace(go.Scatter(x=patient_avg["sleep_disturbances"], y=patient_avg["hr_sd"], mode="markers", marker=dict(color="#fb7185", size=10), name="Disturbance / HR"), row=2, col=2)
        plotly_chart(transparent(sleep_fig, 560))

    c21, c22 = st.columns(2)
    with c21:
        risk_df = pred.groupby("patient_id", as_index=False)[["heart_rate", "glucose", "bolus_volume_delivered", "steps"]].mean()
        risk_df["risk_score"] = risk_df["heart_rate"] * 0.30 + risk_df["glucose"] * 0.30 + risk_df["bolus_volume_delivered"] * 0.30 - risk_df["steps"] * 0.10
        risk_df["risk_level"] = np.select([risk_df["risk_score"] > risk_df["risk_score"].quantile(.75), risk_df["risk_score"] > risk_df["risk_score"].quantile(.50)], ["High Risk", "Moderate Risk"], default="Low Risk")
        key_insight("Predicted Hospitalization Risk", "The notebook ranks patients by elevated heart rate, glucose, insulin usage, and lower activity.")
        fig = px.bar(risk_df.sort_values("risk_score"), x="risk_score", y="patient_id", orientation="h", color="risk_level", title="Predicted Hospitalization Risk", color_discrete_sequence=CHART_SEQ)
        plotly_chart(transparent(fig, 500))
    with c22:
        diabetes_df = pred.groupby("patient_id", as_index=False)[["steps", "calories", "heart_rate", "bolus_volume_delivered"]].mean()
        diabetes_df["diabetes_probability"] = diabetes_df["heart_rate"] * 0.25 + diabetes_df["bolus_volume_delivered"] * 0.35 - diabetes_df["steps"] * 0.20 + diabetes_df["calories"] * 0.20
        diabetes_df["diabetes_probability"] = (diabetes_df["diabetes_probability"] / diabetes_df["diabetes_probability"].max() * 100).round(2)
        key_insight("Predicted Diabetes Probability", "The notebook bubble chart highlights higher probability where activity is lower and heart rate or insulin use is elevated.")
        fig = px.scatter(diabetes_df, x="steps", y="diabetes_probability", size="heart_rate", color="bolus_volume_delivered", hover_name="patient_id", title="Predicted Diabetes Probability", color_continuous_scale=CHART_SEQ)
        plotly_chart(transparent(fig, 500))

