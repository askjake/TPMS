"""
Observation Patterns page — time-of-day and frequency analysis.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tpms_dashboard.config import EXPORT_DIR
from tpms_dashboard.components.freshness_banner import render_freshness_banner
from tpms_dashboard.utils.data_loader import exports_fingerprint, build_sensor_timelines, build_observation_heatmap


def render():
    """Render the Observation Patterns page."""
    st.header("\U0001f50d Observation Patterns")
    st.caption(
        "When are sensors captured? Reveals time-of-day and day-of-week rhythms \u2014 "
        "useful for understanding receiver location and vehicle traffic patterns."
    )
    render_freshness_banner()

    fp = exports_fingerprint(EXPORT_DIR)
    heatmap_df, hourly_df, weekly_df = build_observation_heatmap(fp)
    tl = build_sensor_timelines(fp)

    if tl.empty:
        st.warning("No export data found.")
        st.stop()

    total_captures = len(tl)
    unique_sensors = tl["sensor_id"].nunique()
    total_sessions = tl["export_file"].nunique()
    span_days = (tl["export_dt"].max() - tl["export_dt"].min()).days

    k = st.columns(4)
    k[0].metric("Total Captures", f"{total_captures:,}")
    k[1].metric("Unique Sensors", f"{unique_sensors:,}")
    k[2].metric("Sessions", total_sessions)
    k[3].metric("Observation Span", f"{span_days} days")

    st.markdown("---")

    # 1. Weekday x Hour heatmap
    st.markdown("### Capture Density: Day of Week vs Hour of Day")
    st.caption("Colour = unique sensors observed in that weekday/hour bucket, summed across all sessions.")
    if not heatmap_df.empty:
        fig_heat = px.imshow(
            heatmap_df,
            color_continuous_scale="Blues",
            aspect="auto",
            title="When are sensors observed? (Day \u00d7 Hour)",
            labels={"color": "Unique Sensors", "x": "Hour of Day", "y": "Weekday"},
        )
        fig_heat.update_xaxes(
            dtick=1, tickvals=list(range(24)),
            ticktext=[f"{h:02d}:00" for h in range(24)]
        )
        fig_heat.update_layout(height=320)
        st.plotly_chart(fig_heat, use_container_width=True)

    # 2. Hour-of-day bar
    st.markdown("### Volume by Hour of Day")
    if not hourly_df.empty:
        hourly_df["hour_label"] = hourly_df["hour"].apply(lambda h: f"{int(h):02d}:00")
        fig_hour = px.bar(
            hourly_df, x="hour_label", y="unique_sensors",
            title="Unique Sensors Observed per Hour of Day (all sessions)",
            labels={"hour_label": "Hour", "unique_sensors": "Unique Sensors"},
            color="unique_sensors", color_continuous_scale="Blues",
        )
        fig_hour.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_hour, use_container_width=True)

    # 3. Day-of-week bar
    st.markdown("### Volume by Day of Week")
    DAYS_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_agg = (
        tl.groupby("weekday")["sensor_id"]
        .nunique()
        .reindex(DAYS_ORDER, fill_value=0)
        .reset_index()
    )
    day_agg.columns = ["Weekday", "Unique Sensors"]
    fig_day = px.bar(
        day_agg, x="Weekday", y="Unique Sensors",
        title="Unique Sensors Observed per Day of Week",
        color="Unique Sensors", color_continuous_scale="Greens",
    )
    fig_day.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_day, use_container_width=True)

    # 4. Weekly cadence
    if not weekly_df.empty and len(weekly_df) > 1:
        st.markdown("### Capture Cadence Over Time")
        fig_wk = go.Figure()
        fig_wk.add_trace(go.Bar(
            x=weekly_df["date"].astype(str), y=weekly_df["unique_sensors"],
            name="Unique Sensors", marker_color="#1f77b4",
        ))
        fig_wk.add_trace(go.Scatter(
            x=weekly_df["date"].astype(str), y=weekly_df["total_captures"],
            name="Total Captures", mode="lines+markers",
            line=dict(color="orange", width=2), yaxis="y2",
        ))
        fig_wk.update_layout(
            title="Observation Cadence Over Calendar Time",
            xaxis_title="Date",
            yaxis=dict(title="Unique Sensors"),
            yaxis2=dict(title="Total Captures", overlaying="y", side="right"),
            hovermode="x unified",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_wk, use_container_width=True)

    # 5. Recurring vs new per session
    st.markdown("### Recurring vs. New Sensors per Session")
    st.caption("Are the same vehicles being seen repeatedly, or mostly new ones each session?")
    if "is_recurring" in tl.columns:
        recur_by_session = (
            tl.groupby(["session_label", "export_dt", "is_recurring"])["sensor_id"]
            .nunique()
            .reset_index(name="count")
            .sort_values("export_dt")
        )
        recur_by_session["type"] = recur_by_session["is_recurring"].map(
            {True: "Recurring", False: "First-time"}
        )
        fig_stack = px.bar(
            recur_by_session, x="session_label", y="count", color="type",
            title="Recurring vs. First-time Sensors per Session",
            labels={"session_label": "Session", "count": "Unique Sensors", "type": ""},
            color_discrete_map={"Recurring": "#1f77b4", "First-time": "#aec7e8"},
            barmode="stack",
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    # 6. Sensor Capture Schedule
    st.markdown("---")
    st.markdown("### Sensor Capture Schedule")
    st.caption(
        "Which hours and day-types does each recurring sensor appear? "
        "Use this to predict when you can next expect to capture a specific vehicle."
    )

    if "last_seen" in tl.columns and tl["last_seen"].notna().any():
        _cap_ts = pd.to_datetime(tl["last_seen"], errors="coerce")
    else:
        _cap_ts = tl["export_dt"]
    tl["capture_hour"] = _cap_ts.dt.hour
    tl["capture_weekday_num"] = _cap_ts.dt.weekday
    tl["capture_weekday"] = _cap_ts.dt.strftime("%a")
    tl["is_weekend"] = tl["capture_weekday_num"] >= 5

    recurring_sched = tl[tl.get("session_count", tl.groupby("sensor_id")["export_file"].transform("nunique")) >= 2].copy()

    if not recurring_sched.empty:
        sched_table = (
            recurring_sched.groupby("sensor_id")
            .agg(
                sessions=("export_file", "nunique"),
                weekday_caps=("is_weekend", lambda x: int((~x).sum())),
                weekend_caps=("is_weekend", lambda x: int(x.sum())),
                typical_hours=("capture_hour", lambda x: sorted(x.dropna().unique().astype(int).tolist())),
                avg_hour=("capture_hour", "mean"),
            )
            .sort_values("sessions", ascending=False)
        )
        sched_table["day_pattern"] = sched_table.apply(
            lambda r: "Both" if r["weekday_caps"] > 0 and r["weekend_caps"] > 0
                      else ("Weekday only" if r["weekday_caps"] > 0 else "Weekend only"),
            axis=1
        )
        sched_table["typical_hours"] = sched_table["typical_hours"].apply(
            lambda hrs: ", ".join(f"{h:02d}:00" for h in hrs[:6]) + (" ..." if len(hrs) > 6 else "")
        )
        sched_table["avg_hour"] = sched_table["avg_hour"].round(1)

        sched_disp = sched_table[["sessions", "day_pattern", "weekday_caps",
                                   "weekend_caps", "typical_hours", "avg_hour"]].copy()
        sched_disp.columns = ["Sessions", "Day Pattern", "Weekday Caps",
                              "Weekend Caps", "Typical Hours", "Avg Hour"]
        sched_disp = sched_disp.reset_index().rename(columns={"sensor_id": "Sensor ID"})
        st.dataframe(sched_disp, use_container_width=True, hide_index=True, height=320)

        # Heatmaps: Sensor x Hour
        top_n = min(25, recurring_sched["sensor_id"].nunique())
        top_sensors = recurring_sched["sensor_id"].value_counts().head(top_n).index.tolist()
        heat_data = recurring_sched[recurring_sched["sensor_id"].isin(top_sensors)]

        col_wd, col_we = st.columns(2)

        wd_data = heat_data[~heat_data["is_weekend"]]
        if not wd_data.empty:
            wd_pivot = wd_data.pivot_table(
                index="sensor_id", columns="capture_hour",
                values="export_file", aggfunc="count", fill_value=0
            ).reindex(columns=range(24), fill_value=0)
            with col_wd:
                st.markdown("**Weekday (Mon\u2013Fri)**")
                fig_wd = px.imshow(
                    wd_pivot, color_continuous_scale="Blues", aspect="auto",
                    labels={"color": "Captures", "x": "Hour", "y": "Sensor"},
                )
                fig_wd.update_xaxes(dtick=2, tickvals=list(range(0, 24, 2)),
                                     ticktext=[f"{h:02d}" for h in range(0, 24, 2)])
                fig_wd.update_layout(height=max(300, top_n * 22), margin=dict(l=10, r=10))
                st.plotly_chart(fig_wd, use_container_width=True)

        we_data = heat_data[heat_data["is_weekend"]]
        if not we_data.empty:
            we_pivot = we_data.pivot_table(
                index="sensor_id", columns="capture_hour",
                values="export_file", aggfunc="count", fill_value=0
            ).reindex(columns=range(24), fill_value=0)
            with col_we:
                st.markdown("**Weekend (Sat\u2013Sun)**")
                fig_we = px.imshow(
                    we_pivot, color_continuous_scale="Oranges", aspect="auto",
                    labels={"color": "Captures", "x": "Hour", "y": "Sensor"},
                )
                fig_we.update_xaxes(dtick=2, tickvals=list(range(0, 24, 2)),
                                     ticktext=[f"{h:02d}" for h in range(0, 24, 2)])
                fig_we.update_layout(height=max(300, top_n * 22), margin=dict(l=10, r=10))
                st.plotly_chart(fig_we, use_container_width=True)

        # Combined bar
        st.markdown("#### Capture probability by hour of day")
        st.caption("All recurring sensors combined \u2014 weekday vs weekend distribution.")
        hour_dist = (
            recurring_sched.groupby(["capture_hour", "is_weekend"])["sensor_id"]
            .count()
            .reset_index(name="captures")
        )
        hour_dist["day_type"] = hour_dist["is_weekend"].map({False: "Weekday", True: "Weekend"})
        fig_prob = px.bar(
            hour_dist, x="capture_hour", y="captures", color="day_type",
            barmode="group",
            labels={"capture_hour": "Hour of Day", "captures": "Captures", "day_type": ""},
            color_discrete_map={"Weekday": "#1f77b4", "Weekend": "#ff7f0e"},
        )
        fig_prob.update_xaxes(dtick=1, tickvals=list(range(24)),
                               ticktext=[f"{h:02d}" for h in range(24)])
        fig_prob.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_prob, use_container_width=True)
    else:
        st.info("Need at least 2 sessions with recurring sensors to build a capture schedule.")

    # 7. Interpretation callout
    st.markdown("---")
    st.markdown("### Interpretation")
    DAYS_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    peak_hour = int(hourly_df.loc[hourly_df["unique_sensors"].idxmax(), "hour"]) if not hourly_df.empty else 0
    day_agg2 = (
        tl.groupby("weekday")["sensor_id"]
        .nunique()
        .reindex(DAYS_ORDER, fill_value=0)
        .reset_index()
    )
    day_agg2.columns = ["Weekday", "Unique Sensors"]
    active_days = day_agg2[day_agg2["Unique Sensors"] > 0]["Weekday"].tolist()
    zero_days = day_agg2[day_agg2["Unique Sensors"] == 0]["Weekday"].tolist()
    recurring_pct = float(tl["is_recurring"].mean() * 100) if "is_recurring" in tl.columns else 0.0

    col1, col2 = st.columns(2)
    with col1:
        st.info(
            f"**Peak capture hour:** {peak_hour:02d}:00\n\n"
            f"**Active days:** {', '.join(active_days)}\n\n"
            f"**No-capture days:** {', '.join(zero_days) if zero_days else 'None'}"
        )
    with col2:
        if zero_days:
            st.warning(
                f"No captures on {', '.join(zero_days)} \u2014 suggests the receiver "
                f"is at a location that is unoccupied on those days "
                f"(e.g. an office parking lot or workplace)."
            )
        st.success(
            f"**{recurring_pct:.0f}%** of all sensor observations are from recurring vehicles "
            f"(seen in more than one session)."
        )
