from __future__ import annotations

from datetime import date
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.data_service import load_all_data, load_split, category_options, mode_value, MONTH_ORDER
from src.model_service import load_artifacts, load_metadata, predict_cancellation, predict_adr, predict_cluster

# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hotel Booking Analytics | Fase 4",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_BG = "#070B14"
CARD_BG = "#0F172A"
CARD_BG_2 = "#111827"
TEXT = "#E5E7EB"
MUTED = "#94A3B8"
TEAL = "#14B8A6"
CYAN = "#22D3EE"
AMBER = "#F59E0B"
RED = "#EF4444"
VIOLET = "#8B5CF6"
GRID = "rgba(148,163,184,0.15)"

st.markdown(
    f"""
<style>
    .stApp {{ background: radial-gradient(circle at 20% 0%, #111827 0%, {DARK_BG} 38%, #05070d 100%); }}
    [data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0B1220 0%, #070B14 100%); border-right: 1px solid #1F2937; }}
    [data-testid="stHeader"] {{ background: rgba(7,11,20,0.75); }}
    h1, h2, h3 {{ letter-spacing: -0.02em; }}
    .hero {{
        padding: 1.35rem 1.55rem;
        border: 1px solid #1F2937;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(20,184,166,.12), rgba(15,23,42,.92) 42%, rgba(139,92,246,.08));
        box-shadow: 0 18px 45px rgba(0,0,0,.25);
        margin-bottom: 1rem;
    }}
    .hero-title {{ font-size: 2.0rem; font-weight: 800; margin: 0; color: {TEXT}; }}
    .hero-sub {{ color: {MUTED}; margin-top: .35rem; font-size: .98rem; }}
    .badge {{ display:inline-block; padding:.25rem .6rem; border-radius:999px; border:1px solid rgba(20,184,166,.35); color:#99F6E4; background:rgba(20,184,166,.08); font-size:.78rem; margin-right:.35rem; }}
    .kpi-card {{
        min-height: 118px;
        padding: 1rem 1.05rem;
        border: 1px solid #1F2937;
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(17,24,39,.95), rgba(15,23,42,.96));
        box-shadow: 0 10px 28px rgba(0,0,0,.18);
    }}
    .kpi-label {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; color:{MUTED}; }}
    .kpi-value {{ font-size:1.7rem; font-weight:800; color:{TEXT}; margin-top:.3rem; }}
    .kpi-note {{ font-size:.78rem; color:{MUTED}; margin-top:.2rem; }}
    .section-card {{
        border: 1px solid #1F2937;
        border-radius: 16px;
        background: rgba(15,23,42,.72);
        padding: 1rem 1.1rem;
        margin-bottom: .8rem;
    }}
    .risk-high {{ color:#FCA5A5; font-weight:800; }}
    .risk-mid {{ color:#FCD34D; font-weight:800; }}
    .risk-low {{ color:#5EEAD4; font-weight:800; }}
    .tiny {{ color:{MUTED}; font-size:.78rem; }}
    div[data-testid="stMetric"] {{ background:rgba(15,23,42,.72); border:1px solid #1F2937; padding:.65rem .8rem; border-radius:14px; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:.4rem; }}
    .stTabs [data-baseweb="tab"] {{ border-radius:10px; padding:.45rem .8rem; background:#0F172A; border:1px solid #1F2937; }}
    .stTabs [aria-selected="true"] {{ background:rgba(20,184,166,.14) !important; border-color:rgba(20,184,166,.45) !important; }}
</style>
""",
    unsafe_allow_html=True,
)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT),
    margin=dict(l=20, r=20, t=55, b=20),
    hoverlabel=dict(bgcolor="#111827", font_color=TEXT),
)


def style_figure(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(**PLOT_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, xanchor="left"))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def kpi_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>""",
        unsafe_allow_html=True,
    )


def fmt_pct(x: float) -> str:
    return f"{x*100:,.1f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_num(x: float, decimals: int = 0) -> str:
    return f"{x:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_eur(x: float) -> str:
    return f"€ {fmt_num(x, 2)}"


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_all_data()


@st.cache_data(show_spinner=False)
def get_train() -> pd.DataFrame:
    return load_split("train")


@st.cache_resource(show_spinner="Cargando modelos de la Fase 3...")
def get_models():
    return load_artifacts()


@st.cache_data(show_spinner=False)
def get_metadata():
    return load_metadata()


@st.cache_data(show_spinner=False)
def get_metrics():
    c = pd.read_csv(ROOT / "models" / "metricas_test_clasificacion.csv")
    r = pd.read_csv(ROOT / "models" / "metricas_test_regresion.csv")
    return c, r


data = get_data()
train = get_train()
metadata = get_metadata()
metrics_class, metrics_reg = get_metrics()

# -----------------------------------------------------------------------------
# Encabezado
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <div>
    <span class="badge">FASE 4</span><span class="badge">CIENCIA DE DATOS</span><span class="badge">UNIVERSIDAD DE LA SALLE</span>
  </div>
  <p class="hero-title">Hotel Booking Analytics</p>
  <p class="hero-sub">Dashboard interactivo para riesgo de cancelación, estimación de ADR y segmentación de reservas.</p>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar / filtros globales
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏨 Hotel Booking")
    st.caption("Proyecto Integrador · Fase 4")
    st.divider()
    st.markdown("#### Filtros del resumen")

    min_date = data["arrival_date"].min().date()
    max_date = data["arrival_date"].max().date()
    date_range = st.date_input(
        "Rango de llegada",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    hotels = st.multiselect("Hotel", category_options(data, "hotel"), default=category_options(data, "hotel"))
    segments_all = category_options(data, "market_segment")
    segments = st.multiselect("Segmento de mercado", segments_all, default=segments_all)

    adr_min, adr_max = float(data["adr"].min()), float(data["adr"].quantile(0.995))
    adr_range = st.slider("ADR (€)", min_value=0.0, max_value=float(round(adr_max, 0)), value=(0.0, float(round(adr_max, 0))), step=5.0)

    lt_min, lt_max = int(data["lead_time"].min()), int(data["lead_time"].max())
    lead_range = st.slider("Anticipación (días)", min_value=lt_min, max_value=lt_max, value=(lt_min, lt_max), step=5)

    st.divider()
    st.caption("Modelos: XGBoost + K-Means · artefactos reales de Fase 3")

# Aplicación de filtros
filtered = data.copy()
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered = filtered[(filtered["arrival_date"] >= start_date) & (filtered["arrival_date"] <= end_date)]
if hotels:
    filtered = filtered[filtered["hotel"].isin(hotels)]
if segments:
    filtered = filtered[filtered["market_segment"].isin(segments)]
filtered = filtered[filtered["adr"].between(adr_range[0], adr_range[1])]
filtered = filtered[filtered["lead_time"].between(lead_range[0], lead_range[1])]

# -----------------------------------------------------------------------------
# Helpers para formularios de predicción
# -----------------------------------------------------------------------------
def options(col: str) -> list[str]:
    return category_options(train, col)


def default_index(col: str, values: list[str]) -> int:
    default = str(mode_value(train, col, values[0] if values else ""))
    return values.index(default) if default in values else 0


def build_reservation_form(prefix: str, include_adr: bool = True) -> pd.DataFrame | None:
    hotel_opts = options("hotel")
    meal_opts = options("meal")
    country_opts = options("country")
    market_opts = options("market_segment")
    channel_opts = options("distribution_channel")
    room_opts = options("reserved_room_type")
    deposit_opts = options("deposit_type")
    agent_opts = options("agent")
    customer_opts = options("customer_type")

    with st.form(f"{prefix}_reservation_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            hotel = st.selectbox("Tipo de hotel", hotel_opts, index=default_index("hotel", hotel_opts), key=f"{prefix}_hotel")
            arrival = st.date_input(
                "Fecha de llegada",
                value=date(2017, 7, 15),
                min_value=date(2015, 7, 1),
                max_value=date(2017, 8, 31),
                key=f"{prefix}_date",
            )
            lead_time = st.number_input("Anticipación (días)", min_value=0, max_value=737, value=int(train["lead_time"].median()), key=f"{prefix}_lead")
            weekend = st.number_input("Noches fin de semana", min_value=0, max_value=19, value=1, key=f"{prefix}_weekend")
            weekday = st.number_input("Noches entre semana", min_value=0, max_value=50, value=2, key=f"{prefix}_weekday")
            adults = st.number_input("Adultos", min_value=1, max_value=10, value=2, key=f"{prefix}_adults")
            children = st.number_input("Niños", min_value=0, max_value=10, value=0, key=f"{prefix}_children")
            babies = st.number_input("Bebés", min_value=0, max_value=10, value=0, key=f"{prefix}_babies")

        with c2:
            meal = st.selectbox("Plan de comidas", meal_opts, index=default_index("meal", meal_opts), key=f"{prefix}_meal")
            country = st.selectbox("País de origen", country_opts, index=default_index("country", country_opts), key=f"{prefix}_country")
            market_segment = st.selectbox("Segmento de mercado", market_opts, index=default_index("market_segment", market_opts), key=f"{prefix}_market")
            distribution_channel = st.selectbox("Canal de distribución", channel_opts, index=default_index("distribution_channel", channel_opts), key=f"{prefix}_channel")
            reserved_room_type = st.selectbox("Habitación reservada", room_opts, index=default_index("reserved_room_type", room_opts), key=f"{prefix}_room")
            deposit_type = st.selectbox("Tipo de depósito", deposit_opts, index=default_index("deposit_type", deposit_opts), key=f"{prefix}_deposit")
            customer_type = st.selectbox("Tipo de cliente", customer_opts, index=default_index("customer_type", customer_opts), key=f"{prefix}_customer")
            agent = st.selectbox("Agencia (código; 0 = sin agencia)", agent_opts, index=default_index("agent", agent_opts), key=f"{prefix}_agent")

        with c3:
            repeated = st.selectbox("Cliente repetido", [0, 1], index=0, format_func=lambda x: "Sí" if x else "No", key=f"{prefix}_repeat")
            prev_cancel = st.number_input("Cancelaciones previas", min_value=0, max_value=30, value=0, key=f"{prefix}_pc")
            prev_ok = st.number_input("Reservas previas no canceladas", min_value=0, max_value=80, value=0, key=f"{prefix}_pok")
            waiting = st.number_input("Días en lista de espera", min_value=0, max_value=400, value=0, key=f"{prefix}_wait")
            parking = st.number_input("Espacios de parqueo requeridos", min_value=0, max_value=8, value=0, key=f"{prefix}_park")
            special = st.number_input("Solicitudes especiales", min_value=0, max_value=8, value=1, key=f"{prefix}_special")
            has_company = st.selectbox("Reserva asociada a empresa", [0, 1], index=0, format_func=lambda x: "Sí" if x else "No", key=f"{prefix}_company")
            adr_value = None
            if include_adr:
                adr_value = st.number_input("ADR conocido (€)", min_value=0.0, max_value=1000.0, value=float(round(train.loc[train["adr"] > 0, "adr"].median(), 2)), step=1.0, key=f"{prefix}_adr")

        submitted = st.form_submit_button("Ejecutar predicción", type="primary", use_container_width=True)

    if not submitted:
        return None

    total_nights = int(weekend + weekday)
    total_guests = int(adults + children + babies)
    month_name = MONTH_ORDER[arrival.month - 1]
    week_number = int(arrival.isocalendar().week)
    prop_weekend = float(weekend / total_nights) if total_nights > 0 else 0.0
    has_agent = int(str(agent) != "0")

    row = {
        "hotel": str(hotel),
        "lead_time": int(lead_time),
        "arrival_date_year": int(arrival.year),
        "arrival_date_month": month_name,
        "arrival_date_week_number": week_number,
        "arrival_date_day_of_month": int(arrival.day),
        "stays_in_weekend_nights": int(weekend),
        "stays_in_week_nights": int(weekday),
        "adults": int(adults),
        "children": int(children),
        "babies": int(babies),
        "meal": str(meal),
        "country": str(country),
        "market_segment": str(market_segment),
        "distribution_channel": str(distribution_channel),
        "is_repeated_guest": int(repeated),
        "previous_cancellations": int(prev_cancel),
        "previous_bookings_not_canceled": int(prev_ok),
        "reserved_room_type": str(reserved_room_type),
        "deposit_type": str(deposit_type),
        "agent": str(agent),
        "days_in_waiting_list": int(waiting),
        "customer_type": str(customer_type),
        "required_car_parking_spaces": int(parking),
        "total_of_special_requests": int(special),
        "tiene_empresa": int(has_company),
        "tiene_agente": int(has_agent),
        "total_huespedes": total_guests,
        "con_ninos": int(children + babies > 0),
        "total_noches": total_nights,
        "prop_fin_semana": prop_weekend,
    }
    if include_adr:
        row["adr"] = float(adr_value)
    return pd.DataFrame([row])


# -----------------------------------------------------------------------------
# Tabs principales
# -----------------------------------------------------------------------------
tab_overview, tab_predict, tab_cluster, tab_models, tab_project = st.tabs(
    ["📊 Resumen ejecutivo", "🔮 Predicción en vivo", "🧩 Segmentación", "🧠 Modelos", "📁 Proyecto"]
)

# -----------------------------------------------------------------------------
# TAB 1: Resumen ejecutivo
# -----------------------------------------------------------------------------
with tab_overview:
    if filtered.empty:
        st.warning("Los filtros actuales no devuelven reservas. Amplía alguno de los rangos.")
    else:
        total_res = len(filtered)
        cancel_rate = float(filtered["is_canceled"].mean())
        avg_adr = float(filtered.loc[filtered["adr"] > 0, "adr"].mean())
        avg_nights = float(filtered["total_noches"].mean())
        booked_value = float(filtered["valor_estancia"].sum())
        canceled_value = float(filtered.loc[filtered["is_canceled"] == 1, "valor_estancia"].sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            kpi_card("Reservas", fmt_num(total_res), "Registros tras filtros")
        with c2:
            kpi_card("Cancelación", fmt_pct(cancel_rate), "Tasa observada")
        with c3:
            kpi_card("ADR promedio", fmt_eur(avg_adr), "Tarifa diaria")
        with c4:
            kpi_card("Estancia media", f"{fmt_num(avg_nights, 1)} noches", "Duración reservada")
        with c5:
            kpi_card("Valor cancelado", fmt_eur(canceled_value), f"de {fmt_eur(booked_value)} reservados")

        st.write("")
        left, right = st.columns([1, 1])
        with left:
            hotel_summary = (
                filtered.groupby("hotel", as_index=False)
                .agg(reservas=("is_canceled", "size"), cancelacion=("is_canceled", "mean"), adr=("adr", "mean"))
            )
            fig = px.bar(
                hotel_summary,
                x="hotel",
                y="cancelacion",
                text=hotel_summary["cancelacion"].map(lambda x: f"{x:.1%}"),
                hover_data={"reservas": True, "adr": ":.2f", "cancelacion": ":.2%"},
                color="hotel",
                color_discrete_sequence=[TEAL, VIOLET],
            )
            fig.update_yaxes(tickformat=".0%")
            fig.update_traces(textposition="outside")
            st.plotly_chart(style_figure(fig, "Tasa de cancelación por hotel"), use_container_width=True)

        with right:
            monthly = (
                filtered.groupby(["arrival_date_year", "arrival_month_num", "arrival_date_month"], as_index=False)
                .agg(cancelacion=("is_canceled", "mean"), reservas=("is_canceled", "size"))
                .sort_values(["arrival_date_year", "arrival_month_num"])
            )
            monthly["periodo"] = monthly["arrival_date_year"].astype(str) + " · " + monthly["arrival_date_month"].str[:3]
            fig = px.line(
                monthly,
                x="periodo",
                y="cancelacion",
                markers=True,
                hover_data={"reservas": True, "cancelacion": ":.2%"},
            )
            fig.update_traces(line=dict(color=CYAN, width=3), marker=dict(size=7))
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(style_figure(fig, "Evolución temporal de la cancelación"), use_container_width=True)

        left, right = st.columns([1, 1])
        with left:
            hist_df = filtered[(filtered["adr"] >= 0) & (filtered["adr"] <= filtered["adr"].quantile(0.995))]
            fig = px.histogram(
                hist_df,
                x="adr",
                color="is_canceled",
                nbins=55,
                barmode="overlay",
                opacity=0.72,
                color_discrete_map={0: TEAL, 1: RED},
                labels={"is_canceled": "Cancelada", "adr": "ADR (€)"},
            )
            st.plotly_chart(style_figure(fig, "Distribución de ADR por estado"), use_container_width=True)

        with right:
            sample = filtered.sample(min(len(filtered), 6000), random_state=42)
            sample["Estado"] = np.where(sample["is_canceled"].eq(1), "Cancelada", "No cancelada")
            fig = px.scatter(
                sample,
                x="lead_time",
                y="adr",
                color="Estado",
                opacity=0.55,
                hover_data=["hotel", "market_segment", "total_noches"],
                color_discrete_map={"Cancelada": RED, "No cancelada": TEAL},
            )
            fig.update_yaxes(range=[0, min(float(sample["adr"].quantile(0.995)), 350)])
            st.plotly_chart(style_figure(fig, "Anticipación vs. ADR"), use_container_width=True)

        st.markdown("#### Lectura ejecutiva")
        top_segment = (
            filtered.groupby("market_segment")["is_canceled"].agg(["mean", "size"]).query("size >= 50").sort_values("mean", ascending=False)
        )
        if not top_segment.empty:
            seg_name = top_segment.index[0]
            seg_rate = float(top_segment.iloc[0]["mean"])
            st.info(
                f"Con los filtros actuales, el segmento **{seg_name}** presenta la mayor tasa de cancelación entre segmentos con al menos 50 reservas: **{seg_rate:.1%}**. "
                f"El valor bruto asociado a reservas canceladas es **{fmt_eur(canceled_value)}**; es una aproximación del ingreso reservado en riesgo, no una pérdida contable definitiva."
            )

        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar datos filtrados (CSV)",
            data=csv_bytes,
            file_name="hotel_booking_filtrado.csv",
            mime="text/csv",
        )

# -----------------------------------------------------------------------------
# TAB 2: Predicción en vivo
# -----------------------------------------------------------------------------
with tab_predict:
    st.markdown("### Predicción supervisada en tiempo real")
    st.caption("Los dos módulos utilizan los pipelines XGBoost persistidos al final de la Fase 3.")

    ptab1, ptab2 = st.tabs(["Riesgo de cancelación", "Estimación de ADR"])

    with ptab1:
        st.markdown("#### Simulador de cancelación")
        st.write("Completa una reserva. El modelo devuelve probabilidad de cancelación y la decisión usando el umbral fijado en validación.")
        row_class = build_reservation_form("class", include_adr=True)
        if row_class is not None:
            try:
                artifacts = get_models()
                result = predict_cancellation(row_class, artifacts, metadata)
                p = result["probability"]
                th = result["threshold"]
                pred = result["prediction"]

                c1, c2 = st.columns([1, 1])
                with c1:
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=p * 100,
                            number={"suffix": "%", "font": {"color": TEXT, "size": 42}},
                            title={"text": "Probabilidad de cancelación", "font": {"color": MUTED}},
                            gauge={
                                "axis": {"range": [0, 100], "tickcolor": MUTED},
                                "bar": {"color": RED if pred else TEAL},
                                "bgcolor": CARD_BG,
                                "bordercolor": "#1F2937",
                                "steps": [
                                    {"range": [0, th * 100], "color": "rgba(20,184,166,.16)"},
                                    {"range": [th * 100, 100], "color": "rgba(239,68,68,.14)"},
                                ],
                                "threshold": {"line": {"color": AMBER, "width": 4}, "thickness": 0.85, "value": th * 100},
                            },
                        )
                    )
                    fig.update_layout(**PLOT_LAYOUT, height=330)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                    if pred:
                        st.markdown(f"### 🔴 Alta alerta del modelo")
                        st.markdown(f"La reserva se clasifica como **cancelable** con una probabilidad de **{p:.1%}**.")
                    else:
                        st.markdown(f"### 🟢 Riesgo por debajo del umbral")
                        st.markdown(f"La reserva se clasifica como **no cancelable** con una probabilidad de cancelación de **{p:.1%}**.")
                    st.write(f"**Umbral aprendido en validación:** {th:.3f}")
                    st.write(f"**ROC-AUC en prueba:** {metadata['clasificacion']['metricas_test']['ROC-AUC']:.3f}")
                    st.write(f"**F1 en prueba:** {metadata['clasificacion']['metricas_test']['F1']:.3f}")
                    st.caption("La salida es una estimación estadística para apoyo a decisiones; no garantiza el comportamiento de una reserva individual.")
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"No fue posible cargar o ejecutar el modelo de clasificación: {exc}")
                st.info("Ejecuta el proyecto con Python 3.12 y las versiones fijadas en requirements.txt.")

    with ptab2:
        st.markdown("#### Estimador de tarifa diaria (ADR)")
        st.write("Introduce las características conocidas de la reserva; el modelo estima el ADR esperado en euros.")
        row_reg = build_reservation_form("reg", include_adr=False)
        if row_reg is not None:
            try:
                artifacts = get_models()
                adr_pred = max(0.0, predict_adr(row_reg, artifacts, metadata))
                total_nights = float(row_reg["total_noches"].iloc[0])
                estimated_stay = adr_pred * total_nights
                c1, c2, c3 = st.columns(3)
                with c1:
                    kpi_card("ADR estimado", fmt_eur(adr_pred), "Predicción XGBoost")
                with c2:
                    kpi_card("Noches", fmt_num(total_nights), "Duración ingresada")
                with c3:
                    kpi_card("Valor estimado estancia", fmt_eur(estimated_stay), "ADR × noches")
                st.success(
                    f"El modelo estima un **ADR de {fmt_eur(adr_pred)}**. En prueba obtuvo **R² = {metadata['regresion']['metricas_test']['R2']:.3f}** y **RMSE = {fmt_eur(metadata['regresion']['metricas_test']['RMSE (€)'])}**."
                )
            except Exception as exc:
                st.error(f"No fue posible cargar o ejecutar el modelo de regresión: {exc}")
                st.info("Ejecuta el proyecto con Python 3.12 y las versiones fijadas en requirements.txt.")

# -----------------------------------------------------------------------------
# TAB 3: Segmentación
# -----------------------------------------------------------------------------
with tab_cluster:
    st.markdown("### Segmentación de reservas con K-Means")
    st.caption("Se utilizan seis variables, K=6 y el preprocesador/PCA persistidos en la Fase 3.")

    clusters = pd.read_csv(ROOT / "models" / "clusters_train.csv")
    cluster_df = train.merge(clusters, on="id_original", how="inner")
    cluster_vars = metadata["clustering"]["variables"]

    c1, c2 = st.columns([1.3, 1])
    with c1:
        display_mode = st.radio("Visualización", ["2D", "3D"], horizontal=True)
        x_var = st.selectbox("Eje X", cluster_vars, index=0)
        y_var = st.selectbox("Eje Y", cluster_vars, index=1)
        sample_cluster = cluster_df.sample(min(len(cluster_df), 9000), random_state=42)
        sample_cluster["cluster_str"] = sample_cluster["cluster"].astype(str)
        palette = [TEAL, CYAN, AMBER, RED, VIOLET, "#60A5FA"]
        color_map = {str(i): palette[i % len(palette)] for i in sorted(sample_cluster["cluster"].unique())}

        if display_mode == "2D":
            fig = px.scatter(
                sample_cluster,
                x=x_var,
                y=y_var,
                color="cluster_str",
                opacity=0.55,
                color_discrete_map=color_map,
                hover_data=["hotel", "is_canceled", "adr", "total_noches"],
                labels={"cluster_str": "Clúster"},
            )
        else:
            z_candidates = [v for v in cluster_vars if v not in {x_var, y_var}] or cluster_vars
            z_var = st.selectbox("Eje Z", z_candidates, index=0)
            fig = px.scatter_3d(
                sample_cluster,
                x=x_var,
                y=y_var,
                z=z_var,
                color="cluster_str",
                opacity=0.55,
                color_discrete_map=color_map,
                hover_data=["hotel", "is_canceled"],
                labels={"cluster_str": "Clúster"},
            )
        st.plotly_chart(style_figure(fig, "Mapa interactivo de clústeres"), use_container_width=True)

    with c2:
        st.markdown("#### Asignar una nueva reserva a un clúster")
        with st.form("cluster_form"):
            lead = st.number_input("Lead time", 0, 737, 90)
            adr_c = st.number_input("ADR (€)", 0.0, 1000.0, 100.0, step=1.0)
            nights = st.number_input("Total noches", 0, 60, 3)
            guests = st.number_input("Total huéspedes", 1, 12, 2)
            prop_we = st.slider("Proporción fin de semana", 0.0, 1.0, 0.33, 0.01)
            requests = st.number_input("Solicitudes especiales", 0, 8, 1)
            go_cluster = st.form_submit_button("Identificar segmento", type="primary", use_container_width=True)
        if go_cluster:
            cluster_row = pd.DataFrame([
                {
                    "lead_time": lead,
                    "adr": adr_c,
                    "total_noches": nights,
                    "total_huespedes": guests,
                    "prop_fin_semana": prop_we,
                    "total_of_special_requests": requests,
                }
            ])
            try:
                artifacts = get_models()
                cres = predict_cluster(cluster_row, artifacts)
                cid = str(cres["cluster"])
                profile = metadata["clustering"]["perfiles_train"].get(cid, {})
                st.success(f"### Clúster {cres['cluster']}")
                if profile:
                    st.metric("Tasa de cancelación del segmento", f"{profile['cancelacion']:.1%}")
                    st.metric("ADR medio del segmento", f"€ {profile['adr']:.2f}")
                    st.metric("Peso en entrenamiento", f"{profile['% reservas']:.1%}")
                st.caption(f"Coordenadas PCA: ({cres['pca1']:.2f}, {cres['pca2']:.2f})")
            except Exception as exc:
                st.error(f"No fue posible ejecutar el clustering: {exc}")

    st.markdown("#### Perfiles numéricos promedio")
    profiles = []
    for cid, p in metadata["clustering"]["perfiles_train"].items():
        profiles.append(
            {
                "Clúster": int(cid),
                "Reservas": int(p["n"]),
                "% reservas": p["% reservas"],
                "Lead time": p["lead_time"],
                "ADR (€)": p["adr"],
                "Noches": p["total_noches"],
                "Huéspedes": p["total_huespedes"],
                "Solicitudes": p["peticiones"],
                "Cancelación": p["cancelacion"],
                "Ingreso estancia (€)": p["ingreso_estancia"],
            }
        )
    profile_df = pd.DataFrame(profiles).sort_values("Clúster")
    st.dataframe(
        profile_df.style.format({
            "% reservas": "{:.1%}",
            "Lead time": "{:.1f}",
            "ADR (€)": "{:.2f}",
            "Noches": "{:.2f}",
            "Huéspedes": "{:.2f}",
            "Solicitudes": "{:.2f}",
            "Cancelación": "{:.1%}",
            "Ingreso estancia (€)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.scatter(
        profile_df,
        x="ADR (€)",
        y="Cancelación",
        size="Reservas",
        color=profile_df["Clúster"].astype(str),
        text="Clúster",
        color_discrete_sequence=palette,
        hover_data=["Lead time", "Noches", "Huéspedes", "Ingreso estancia (€)"],
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(style_figure(fig, "Riesgo y valor por segmento"), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: Modelos
# -----------------------------------------------------------------------------
with tab_models:
    st.markdown("### Comparación de modelos y métricas finales")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        kpi_card("Mejor clasificador", metadata["clasificacion"]["modelo"], f"AUC {metadata['clasificacion']['metricas_test']['ROC-AUC']:.3f}")
    with m2:
        kpi_card("F1 clasificación", f"{metadata['clasificacion']['metricas_test']['F1']:.3f}", f"Umbral {metadata['clasificacion']['umbral']:.3f}")
    with m3:
        kpi_card("Mejor regresor", metadata["regresion"]["modelo"], f"R² {metadata['regresion']['metricas_test']['R2']:.3f}")
    with m4:
        kpi_card("RMSE regresión", fmt_eur(metadata["regresion"]["metricas_test"]["RMSE (€)"]), f"MAE {fmt_eur(metadata['regresion']['metricas_test']['MAE (€)'])}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Clasificación")
        st.dataframe(metrics_class, use_container_width=True, hide_index=True)
        class_long = metrics_class.melt(id_vars=["Modelo"], value_vars=["F1", "ROC-AUC", "PR-AUC"], var_name="Métrica", value_name="Valor")
        fig = px.bar(class_long, x="Modelo", y="Valor", color="Métrica", barmode="group", color_discrete_sequence=[TEAL, CYAN, VIOLET])
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(style_figure(fig, "Desempeño de clasificación en test"), use_container_width=True)

    with right:
        st.markdown("#### Regresión")
        st.dataframe(metrics_reg, use_container_width=True, hide_index=True)
        fig = px.bar(metrics_reg, x="Modelo", y="RMSE (€)", color="R2", color_continuous_scale="Viridis")
        st.plotly_chart(style_figure(fig, "RMSE de regresión en test"), use_container_width=True)

    st.markdown("#### Evidencias de la Fase 3")
    gallery = [
        ("9_1_sintesis_prueba.png", "Síntesis de resultados en prueba"),
        ("4_2_roc_pr_validacion.png", "Curvas ROC y PR en validación"),
        ("7_1_shap_beeswarm.png", "Interpretabilidad SHAP de clasificación"),
        ("7_2_shap_regresion.png", "Interpretabilidad SHAP de regresión"),
        ("6_1_codo_silueta.png", "Selección de K por codo y silueta"),
        ("6_3_centroides_pca.png", "Centroides de clústeres en PCA"),
    ]
    cols = st.columns(2)
    for i, (filename, caption) in enumerate(gallery):
        path = ROOT / "figures" / filename
        if path.exists():
            with cols[i % 2]:
                st.image(str(path), caption=caption, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: Proyecto / reproducibilidad
# -----------------------------------------------------------------------------
with tab_project:
    st.markdown("### Trazabilidad del Proyecto Integrador")
    st.write(
        "Esta aplicación consolida los artefactos de las Fases 1, 2 y 3 y cumple los cuatro módulos exigidos para el dashboard: "
        "**filtros dinámicos**, **tarjetas KPI**, **predicción en vivo** y **segmentación interactiva**."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
<div class="section-card"><b>Fase 1 · EDA</b><br><span class="tiny">Problema, objetivos SMART, ficha técnica, calidad de datos, análisis univariado/bivariado y diagnóstico de fuga.</span></div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class="section-card"><b>Fase 2 · Preparación</b><br><span class="tiny">Limpieza, imputación justificada, variables derivadas, partición 70/15/15, preprocesamiento sin fuga y baselines.</span></div>
""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
<div class="section-card"><b>Fase 3 · Modelado</b><br><span class="tiny">XGBoost, Random Forest, MLP, K-Means, jerárquico, optimización, interpretabilidad y persistencia de modelos.</span></div>
""", unsafe_allow_html=True)

    st.markdown("#### Resultados consolidados")
    st.markdown(
        f"""
- **Dataset:** Hotel Booking Demand; 119.390 registros originales y 119.193 reservas tras depuración.
- **Clasificación:** XGBoost, ROC-AUC de prueba **{metadata['clasificacion']['metricas_test']['ROC-AUC']:.3f}**, F1 **{metadata['clasificacion']['metricas_test']['F1']:.3f}**.
- **Regresión ADR:** XGBoost, R² de prueba **{metadata['regresion']['metricas_test']['R2']:.3f}**, RMSE **€ {metadata['regresion']['metricas_test']['RMSE (€)']:.2f}**.
- **Clustering:** K-Means con **{metadata['clustering']['k']} clústeres**, visualizados y perfilados dentro del dashboard.
"""
    )

    st.markdown("#### Reproducibilidad")
    st.code(
        """Hotel_Booking_Fase4/
├── data/                 # train / validation / test
├── notebooks/            # cuadernos originales Fases 1-3
├── src/                  # transformadores y servicios
├── dashboard/app.py      # aplicación Streamlit
├── models/               # pipelines, métricas, metadatos y MLP
├── figures/              # evidencias gráficas Fase 3
├── docs/                 # guía, informes y documentación final
├── requirements.txt      # versiones fijadas
└── EJECUTAR_DASHBOARD.bat""",
        language="text",
    )

    st.info("Para Windows, basta con ejecutar **EJECUTAR_DASHBOARD.bat** desde la raíz del proyecto. La primera ejecución crea el entorno e instala las dependencias.")

    st.markdown("#### Integrantes")
    st.write("**Miguel Angel Ricardo** · **Jose Alexander Barahona Calderon**")
    st.caption("Universidad de La Salle · Maestría en Inteligencia Artificial · Ciencia de Datos · Docente Fabián Camilo Castro Riveros")
