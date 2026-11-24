"""Hotel Booking Analytics - dashboard de la Fase 4 (Streamlit)."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_service import MONTH_ORDER, load_all_data, numero_semana  # noqa: E402
from model_service import (NOMBRES_CLUSTER, assign_clusters, load_artifacts, load_json,  # noqa: E402
                           load_metadata, predict_adr, predict_cancellation, tasa_observada)

# -----------------------------------------------------------------------------
# Configuración y estilo
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Hotel Booking Analytics | Fase 4", page_icon="🏨", layout="wide",
                   initial_sidebar_state="expanded")

DARK_BG, CARD_BG, TEXT, MUTED = "#070B14", "#0F172A", "#E5E7EB", "#94A3B8"
TEAL, AMBER, GRIS = "#14B8A6", "#F59E0B", "#475569"
GRID = "rgba(148,163,184,0.15)"
# Paleta categórica validada (contraste y daltonismo) sobre fondo oscuro
AZUL, NARANJA = "#3987e5", "#d95926"
PALETA = [AZUL, NARANJA, "#199e70", "#c98500", "#d55181", "#9085e9"]
SIMBOLOS = ["circle", "square", "diamond", "cross", "x", "triangle-up"]
SIMBOLOS_3D = ["circle", "square", "diamond", "cross", "x", "circle-open"]
ESTADOS = {"No cancelada": AZUL, "Cancelada": NARANJA}

st.markdown(
    f"""
<style>
    .stApp {{ background: radial-gradient(circle at 20% 0%, #111827 0%, {DARK_BG} 38%, #05070d 100%); }}
    [data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0B1220 0%, {DARK_BG} 100%); border-right: 1px solid #1F2937; }}
    [data-testid="stHeader"] {{ background: rgba(7,11,20,0.75); }}
    [data-testid="stMainBlockContainer"] {{ padding-top: 3.2rem; }}
    h1, h2, h3 {{ letter-spacing: -0.02em; }}
    .hero {{ padding: 1.1rem 1.5rem; border: 1px solid #1F2937; border-radius: 18px; margin-bottom: 1rem;
             background: linear-gradient(135deg, rgba(20,184,166,.12), rgba(15,23,42,.92) 42%, rgba(139,92,246,.08)); }}
    .hero-title {{ font-size: 2.1rem; line-height: 1.2; font-weight: 800; margin: .45rem 0 .15rem 0; color: {TEXT}; }}
    .hero-sub {{ color: {MUTED}; font-size: 1rem; }}
    .badge {{ display:inline-block; padding:.2rem .6rem; border-radius:999px; border:1px solid rgba(20,184,166,.35);
              color:#99F6E4; background:rgba(20,184,166,.08); font-size:.75rem; margin-right:.35rem; }}
    .kpi-card {{ min-height: 112px; padding: .9rem 1rem; border: 1px solid #1F2937; border-radius: 16px; margin-bottom: .7rem;
                 background: linear-gradient(180deg, rgba(17,24,39,.95), rgba(15,23,42,.96)); }}
    .kpi-label {{ font-size:.75rem; text-transform:uppercase; letter-spacing:.07em; color:{MUTED}; }}
    .kpi-value {{ font-size:1.6rem; font-weight:800; color:{TEXT}; margin-top:.25rem; white-space:nowrap; }}
    .kpi-note {{ font-size:.78rem; color:{MUTED}; margin-top:.15rem; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:.4rem; }}
    .stTabs [data-baseweb="tab"] {{ border-radius:10px; padding:.45rem .8rem; background:{CARD_BG}; border:1px solid #1F2937; }}
    .stTabs [aria-selected="true"] {{ background:rgba(20,184,166,.14) !important; border-color:rgba(20,184,166,.45) !important; }}
</style>
""",
    unsafe_allow_html=True,
)


def style_figure(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT),
        margin=dict(l=20, r=20, t=55, b=20), height=height, separators=",.",
        hoverlabel=dict(bgcolor="#111827", font_color=TEXT),
        title=dict(text=title, x=0.01, xanchor="left"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def fmt_num(x: float, dec: int = 0) -> str:
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(x: float, dec: int = 1) -> str:
    return f"{fmt_num(x * 100, dec)} %"


def fmt_eur(x: float, dec: int = 2) -> str:
    return f"{fmt_num(x, dec)} €"


def fmt_millones(x: float) -> str:
    return f"{fmt_num(x / 1e6, 2)} M€" if abs(x) >= 1e6 else fmt_eur(x, 0)


def kpi_card(label: str, value: str, note: str = "") -> None:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
                f'<div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)


ETIQUETAS = {
    "lead_time": "Anticipación (días)", "adr": "ADR (€)", "total_noches": "Noches",
    "total_huespedes": "Huéspedes", "prop_fin_semana": "Proporción fin de semana",
    "total_of_special_requests": "Solicitudes especiales", "country": "País", "deposit_type": "Tipo de depósito",
    "agent": "Agencia", "market_segment": "Segmento de mercado", "hotel": "Hotel", "meal": "Plan de comidas",
    "customer_type": "Tipo de cliente", "distribution_channel": "Canal de distribución",
    "reserved_room_type": "Habitación reservada", "arrival_date_year": "Año de llegada",
    "arrival_date_month": "Mes de llegada", "arrival_date_week_number": "Semana de llegada",
    "arrival_date_day_of_month": "Día de llegada", "required_car_parking_spaces": "Estacionamiento",
    "previous_cancellations": "Cancelaciones previas", "previous_bookings_not_canceled": "Reservas previas cumplidas",
    "is_repeated_guest": "Cliente repetido", "days_in_waiting_list": "Días en lista de espera",
    "stays_in_weekend_nights": "Noches de fin de semana", "stays_in_week_nights": "Noches entre semana",
    "adults": "Adultos", "children": "Niños", "babies": "Bebés", "con_ninos": "Viaja con niños",
    "tiene_empresa": "Asociada a empresa", "tiene_agente": "Tiene agencia",
    "pca1": "Componente principal 1", "pca2": "Componente principal 2",
}


# -----------------------------------------------------------------------------
# Carga (en caché)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelos...")
def get_models() -> dict:
    return load_artifacts()


@st.cache_resource(show_spinner="Cargando datos...")   # recurso compartido: no se copia en cada sesión
def get_data() -> pd.DataFrame:
    df = load_all_data()
    return pd.concat([df, assign_clusters(df, get_models())], axis=1)


def version_modelos() -> float:
    """Fecha del archivo más reciente de models/: invalida la caché cuando se publican cifras nuevas."""
    return max(f.stat().st_mtime for f in (ROOT / "models").iterdir())


@st.cache_data(show_spinner=False)
def get_json(version: float) -> tuple[dict, list, dict]:
    return load_metadata(), load_json("confiabilidad_validacion.json"), load_json("metricas_clustering.json")


@st.cache_data(show_spinner=False)
def get_metrics(version: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_csv(ROOT / "models" / "metricas_test_clasificacion.csv"),
            pd.read_csv(ROOT / "models" / "metricas_test_regresion.csv"))


@st.cache_data(show_spinner=False)
def opciones_formulario() -> dict[str, list[str]]:
    """Valores vistos en entrenamiento, del más al menos frecuente."""
    train = get_data().query("split == 'train'")
    cols = ["hotel", "meal", "country", "market_segment", "distribution_channel", "reserved_room_type",
            "deposit_type", "customer_type", "agent"]
    conteos = {c: train[c].value_counts() for c in cols}
    return {c: [str(v) for v in k[k > 0].index] for c, k in conteos.items()}


data = get_data()
metadata, confiabilidad, met_cluster = get_json(version_modelos())
metrics_class, metrics_reg = get_metrics(version_modelos())
MC, MR = metadata["clasificacion"]["metricas_test"], metadata["regresion"]["metricas_test"]
UMBRAL = float(metadata["clasificacion"]["umbral"])
PERFILES = {int(k): v for k, v in metadata["clustering"]["perfiles_train"].items()}

st.markdown(
    """
<div class="hero">
  <span class="badge">FASE 4</span><span class="badge">CIENCIA DE DATOS</span><span class="badge">UNIVERSIDAD DE LA SALLE</span>
  <div class="hero-title">Hotel Booking Analytics</div>
  <div class="hero-sub">Riesgo de cancelación, estimación de la tarifa diaria (ADR) y segmentación de reservas · Dos hoteles de Portugal, 2015–2017</div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Filtros (afectan Resumen y Segmentación)
# -----------------------------------------------------------------------------
FECHA_MIN, FECHA_MAX = data["arrival_date"].min().date(), data["arrival_date"].max().date()
ADR_MAX = float(np.ceil(data["adr"].max() / 10) * 10)      # incluye todas las reservas
LEAD_MAX, NOCHES_MAX = int(data["lead_time"].max()), int(data["total_noches"].max())


def restablecer() -> None:
    """Cambia la versión de las claves: el navegador recibe controles nuevos con su valor inicial.
    Borrar las claves de session_state no basta, porque el navegador conserva lo seleccionado."""
    st.session_state["version_filtros"] = st.session_state.get("version_filtros", 0) + 1


def por_frecuencia(col: str) -> list:
    return list(data[col].value_counts().index)


with st.sidebar:
    vf = st.session_state.get("version_filtros", 0)
    st.markdown("### 🏨 Filtros dinámicos")
    st.caption("Actualizan las pestañas Resumen y Segmentación. Una lista vacía equivale a «todos».")
    f_fechas = st.date_input("Rango de llegada", value=(FECHA_MIN, FECHA_MAX), min_value=FECHA_MIN,
                             max_value=FECHA_MAX, format="DD/MM/YYYY", key=f"f_fechas_{vf}")
    f_hotel = st.multiselect("Hotel", por_frecuencia("hotel"), placeholder="Todos", key=f"f_hotel_{vf}")
    f_segmento = st.multiselect("Segmento de mercado", por_frecuencia("market_segment"), placeholder="Todos", key=f"f_segmento_{vf}")
    f_deposito = st.multiselect("Tipo de depósito", por_frecuencia("deposit_type"), placeholder="Todos", key=f"f_deposito_{vf}")
    f_pais = st.multiselect("País de origen", por_frecuencia("country"), placeholder="Todos", key=f"f_pais_{vf}")
    f_cluster = st.multiselect("Clúster", list(NOMBRES_CLUSTER), placeholder="Todos", key=f"f_cluster_{vf}",
                               format_func=lambda k: f"{k} · {NOMBRES_CLUSTER[k]}")
    f_adr = st.slider("ADR (€)", 0.0, ADR_MAX, (0.0, ADR_MAX), step=5.0, key=f"f_adr_{vf}")
    f_lead = st.slider("Anticipación (días)", 0, LEAD_MAX, (0, LEAD_MAX), key=f"f_lead_{vf}")
    f_noches = st.slider("Noches", 0, NOCHES_MAX, (0, NOCHES_MAX), key=f"f_noches_{vf}")
    st.button("Restablecer filtros", on_click=restablecer, width="stretch")
    st.divider()
    st.caption("Modelos de la Fase 3: XGBoost (cancelación y ADR) y K-Means (K = 6).")

mascara = (data["adr"].between(*f_adr) & data["lead_time"].between(*f_lead) & data["total_noches"].between(*f_noches))
if isinstance(f_fechas, (tuple, list)) and len(f_fechas) == 2:
    mascara &= data["arrival_date"].between(pd.Timestamp(f_fechas[0]), pd.Timestamp(f_fechas[1]))
else:
    st.sidebar.info("Elige también la fecha final del rango.")
for columna, valores in (("hotel", f_hotel), ("market_segment", f_segmento), ("deposit_type", f_deposito),
                         ("country", f_pais), ("cluster", f_cluster)):
    if valores:
        mascara &= data[columna].isin(valores)
filtered = data[mascara]

tab_overview, tab_predict, tab_cluster, tab_models, tab_project = st.tabs(
    ["📊 Resumen ejecutivo", "🔮 Predicción en vivo", "🧩 Segmentación", "🧠 Modelos", "📁 Proyecto"])

# -----------------------------------------------------------------------------
# 1. Resumen ejecutivo
# -----------------------------------------------------------------------------
with tab_overview:
    st.markdown("##### Rendimiento de la solución")
    cols = st.columns(5)
    tarjetas = [
        ("ROC-AUC cancelación", fmt_num(MC["ROC-AUC"], 3), "Meta: ≥ 0,85 ✓"),
        ("F1 cancelación", fmt_num(MC["F1"], 3), f"Recall {fmt_num(MC['Recall'], 3)} · Precisión {fmt_num(MC['Precision'], 3)}"),
        ("Error medio del ADR", fmt_eur(MR["MAE (€)"]), "MAE por noche"),
        ("R² del ADR", fmt_num(MR["R2"], 3), f"RMSE {fmt_eur(MR['RMSE (€)'])}"),
        ("Silueta K-Means", fmt_num(met_cluster["silueta_test"], 3), "Meta: ≥ 0,25 · no alcanzada"),
    ]
    for col, tarjeta in zip(cols, tarjetas):
        with col:
            kpi_card(*tarjeta)
    st.caption("Métricas en el conjunto de prueba (17.670 reservas no usadas para entrenar). No cambian con los filtros.")

    st.markdown("##### Indicadores de negocio")
    if filtered.empty:
        st.warning("Los filtros actuales no devuelven reservas. Amplía alguno de los rangos.")
    else:
        tasa = float(filtered["is_canceled"].mean())
        pagadas = filtered.loc[filtered["adr"] > 0, "adr"]
        reservado = float(filtered["valor_estancia"].sum())
        cancelado = float(filtered.loc[filtered["is_canceled"] == 1, "valor_estancia"].sum())
        cols = st.columns(5)
        tarjetas = [
            ("Reservas", fmt_num(len(filtered)), f"{fmt_pct(len(filtered) / len(data))} del total"),
            ("Cancelación", fmt_pct(tasa), "Tasa observada"),
            ("ADR promedio", fmt_eur(pagadas.mean()) if len(pagadas) else "—", "Reservas con ADR > 0"),
            ("Estancia media", f"{fmt_num(filtered['total_noches'].mean(), 1)} noches", "Duración reservada"),
            ("Valor cancelado", fmt_millones(cancelado), f"de {fmt_millones(reservado)} reservados"),
        ]
        for col, tarjeta in zip(cols, tarjetas):
            with col:
                kpi_card(*tarjeta)

        izq, der = st.columns(2)
        with izq:
            por_hotel = (filtered.groupby("hotel", observed=True)
                         .agg(Reservas=("is_canceled", "size"), tasa=("is_canceled", "mean")).reset_index())
            fig = px.bar(por_hotel, x="hotel", y="tasa", text=[fmt_pct(t) for t in por_hotel["tasa"]],
                         labels={"tasa": "Tasa de cancelación", "hotel": ""}, hover_data={"Reservas": ":,", "tasa": ":.1%"})
            fig.update_traces(marker_color=AZUL, textposition="outside", width=0.45, cliponaxis=False)
            fig.update_yaxes(tickformat=".0%", rangemode="tozero")
            st.plotly_chart(style_figure(fig, "Tasa de cancelación por hotel"), width="stretch")
        with der:
            mensual = (filtered.assign(mes=filtered["arrival_date"].dt.to_period("M").dt.to_timestamp())
                       .groupby("mes").agg(Reservas=("is_canceled", "size"), tasa=("is_canceled", "mean")).reset_index())
            fig = px.line(mensual, x="mes", y="tasa", markers=True, hover_data={"Reservas": ":,", "tasa": ":.1%", "mes": "|%m/%Y"},
                          labels={"mes": "Mes de llegada", "tasa": "Tasa de cancelación"})
            fig.update_traces(line=dict(color=AZUL, width=2), marker=dict(size=8))
            fig.update_yaxes(tickformat=".0%", rangemode="tozero")
            fig.update_xaxes(tickformat="%m/%Y")
            st.plotly_chart(style_figure(fig, "Evolución mensual de la cancelación"), width="stretch")

        muestra = filtered.sample(min(len(filtered), 6000), random_state=42)
        muestra = muestra.assign(Estado=np.where(muestra["is_canceled"].eq(1), "Cancelada", "No cancelada"))
        izq, der = st.columns(2)
        with izq:
            rangos = pd.cut(filtered["lead_time"], [-1, 7, 30, 90, 180, 365, 10_000],
                            labels=["0–7", "8–30", "31–90", "91–180", "181–365", "> 365"])
            por_lead = filtered.groupby(rangos, observed=True)["is_canceled"].agg(tasa="mean", Reservas="size").reset_index()
            fig = px.bar(por_lead, x="lead_time", y="tasa", text=[fmt_pct(t, 0) for t in por_lead["tasa"]],
                         labels={"lead_time": "Anticipación (días)", "tasa": "Tasa de cancelación"},
                         hover_data={"Reservas": ":,", "tasa": ":.1%"})
            fig.update_traces(marker_color=AZUL, textposition="outside", width=0.6, cliponaxis=False)
            fig.update_yaxes(tickformat=".0%", rangemode="tozero")
            st.plotly_chart(style_figure(fig, "A mayor anticipación, más cancelación"), width="stretch")
        with der:
            fig = px.scatter(muestra, x="lead_time", y="adr", color="Estado", opacity=0.5, color_discrete_map=ESTADOS,
                             category_orders={"Estado": list(ESTADOS)}, hover_data=["hotel", "market_segment", "total_noches"],
                             labels=ETIQUETAS | {"hotel": "Hotel"})
            fig.update_traces(marker=dict(size=5))
            st.plotly_chart(style_figure(fig, f"Anticipación y ADR (muestra de {fmt_num(len(muestra))})"), width="stretch")

        segmentos = filtered.groupby("market_segment", observed=True)["is_canceled"].agg(["mean", "size"]).query("size >= 50")
        if not segmentos.empty:
            peor = segmentos["mean"].idxmax()
            st.info(f"**Lectura ejecutiva.** Con estos filtros, el segmento **{peor}** tiene la mayor tasa de cancelación "
                    f"(**{fmt_pct(segmentos.loc[peor, 'mean'])}**) entre los segmentos con al menos 50 reservas. El valor "
                    f"reservado en cancelaciones es **{fmt_millones(cancelado)}**: ingreso en riesgo, no pérdida contable, "
                    "porque parte de esas habitaciones se revende.")
        st.download_button("⬇️ Descargar datos filtrados (CSV)", on_click="ignore", mime="text/csv",
                           data=lambda: filtered.to_csv(index=False).encode("utf-8-sig"),
                           file_name="hotel_booking_filtrado.csv")

# -----------------------------------------------------------------------------
# 2. Predicción en vivo
# -----------------------------------------------------------------------------
def formulario() -> tuple[pd.DataFrame, bool] | None:
    op = opciones_formulario()
    with st.form("reserva"):
        c1, c2, c3 = st.columns(3)
        with c1:
            hotel = st.selectbox("Tipo de hotel", op["hotel"])
            llegada = st.date_input("Fecha de llegada", value=date(2017, 7, 15), min_value=date(2015, 7, 1),
                                    max_value=date(2017, 8, 31), format="DD/MM/YYYY",
                                    help="Solo el periodo de los datos: los modelos de árboles no extrapolan a años posteriores.")
            lead = st.number_input("Anticipación (días)", 0, 737, 69)
            finde = st.number_input("Noches de fin de semana", 0, 19, 1)
            semana = st.number_input("Noches entre semana", 0, 50, 2)
            adultos = st.number_input("Adultos", 0, 10, 2)
            ninos = st.number_input("Niños", 0, 9, 0)
            bebes = st.number_input("Bebés", 0, 8, 0)
        with c2:
            meal = st.selectbox("Plan de comidas", op["meal"], help="BB: desayuno · HB: media pensión · FB: pensión completa · SC: sin plan")
            pais = st.selectbox("País de origen (código ISO)", op["country"],
                                help="Variable muy influyente: verifica que se conozca al reservar.")
            segmento = st.selectbox("Segmento de mercado", op["market_segment"])
            canal = st.selectbox("Canal de distribución", op["distribution_channel"])
            habitacion = st.selectbox("Habitación reservada", op["reserved_room_type"])
            deposito = st.selectbox("Tipo de depósito", op["deposit_type"])
            cliente = st.selectbox("Tipo de cliente", op["customer_type"])
            agencia = st.selectbox("Agencia (0 = sin agencia)", op["agent"])
        with c3:
            repetido = st.selectbox("Cliente repetido", [0, 1], format_func=lambda x: "Sí" if x else "No")
            prev_cancel = st.number_input("Cancelaciones previas", 0, 26, 0)
            prev_ok = st.number_input("Reservas previas no canceladas", 0, 72, 0)
            espera = st.number_input("Días en lista de espera", 0, 391, 0)
            parqueo = st.number_input("Plazas de estacionamiento", 0, 8, 0)
            especiales = st.number_input("Solicitudes especiales", 0, 5, 1)
            empresa = st.selectbox("Asociada a empresa", [0, 1], format_func=lambda x: "Sí" if x else "No")
            tarifa = st.number_input("ADR pactado (€ por noche)", 0.0, 1000.0, 95.0, step=1.0)
            usar_estimado = st.checkbox("No conozco el ADR: usar el estimado por el modelo")
        enviado = st.form_submit_button("Ejecutar predicción", type="primary", width="stretch")
    if not enviado:
        return None
    if adultos + ninos + bebes == 0:
        st.error("La reserva debe tener al menos un huésped.")
        return None
    noches = int(finde + semana)
    fila = {
        "hotel": hotel, "lead_time": lead, "arrival_date_year": llegada.year,
        "arrival_date_month": MONTH_ORDER[llegada.month - 1], "arrival_date_week_number": numero_semana(llegada),
        "arrival_date_day_of_month": llegada.day, "stays_in_weekend_nights": finde, "stays_in_week_nights": semana,
        "adults": adultos, "children": ninos, "babies": bebes, "meal": meal, "country": pais,
        "market_segment": segmento, "distribution_channel": canal, "is_repeated_guest": repetido,
        "previous_cancellations": prev_cancel, "previous_bookings_not_canceled": prev_ok,
        "reserved_room_type": habitacion, "deposit_type": deposito, "agent": agencia, "days_in_waiting_list": espera,
        "customer_type": cliente, "adr": tarifa, "required_car_parking_spaces": parqueo,
        "total_of_special_requests": especiales, "tiene_empresa": empresa, "tiene_agente": int(agencia != "0"),
        "total_huespedes": int(adultos + ninos + bebes), "con_ninos": int(ninos + bebes > 0), "total_noches": noches,
        "prop_fin_semana": finde / noches if noches else 0.0,
    }
    return pd.DataFrame([fila]), usar_estimado


def grafico_factores(factores: pd.Series, titulo: str, unidad: str, decimales: int) -> go.Figure:
    top = factores.head(8).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top.values, y=[ETIQUETAS.get(v, v) for v in top.index], orientation="h",
        marker_color=[NARANJA if v > 0 else AZUL for v in top.values], text=[("+" if v > 0 else "−") + fmt_num(abs(v), decimales) for v in top.values],
        textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{text} " + unidad + "<extra></extra>"))
    margen = float(top.abs().max()) * 1.3
    fig.update_xaxes(title=f"Efecto ({unidad}) · naranja aumenta, azul reduce", range=[-margen, margen])
    return style_figure(fig, titulo, 360)


with tab_predict:
    st.markdown("### Predicción en vivo de una reserva")
    st.write("Completa los datos: los modelos devuelven el riesgo de cancelación, la tarifa diaria estimada y el segmento.")
    envio = formulario()
    if envio is not None:
        fila, usar_estimado = envio
        modelos = get_models()
        r_adr = predict_adr(fila, modelos, metadata)
        if usar_estimado:
            fila["adr"] = r_adr["adr"]
        r_can = predict_cancellation(fila, modelos, metadata)
        r_clu = assign_clusters(fila, modelos).iloc[0]
        st.session_state["ultima"] = {"fila": fila, "adr": r_adr, "cancelacion": r_can, "cluster": r_clu,
                                      "usar_estimado": usar_estimado}

    if "ultima" not in st.session_state:
        st.caption("Pulsa «Ejecutar predicción» para ver el resultado.")
    else:
        u = st.session_state["ultima"]
        can, adr_est, clu = u["cancelacion"], u["adr"]["adr"], int(u["cluster"]["cluster"])
        noches = int(u["fila"]["total_noches"].iloc[0])
        banda = tasa_observada(can["puntaje"], confiabilidad)
        c1, c2, c3 = st.columns(3)
        with c1, st.container(border=True):
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=can["puntaje"] * 100, number={"valueformat": ".1f", "font": {"color": TEXT, "size": 40}},
                title={"text": "Puntaje de riesgo (0–100)", "font": {"color": MUTED, "size": 15}},
                gauge={"axis": {"range": [0, 100], "tickcolor": MUTED}, "bar": {"color": NARANJA if can["alerta"] else AZUL},
                       "bgcolor": CARD_BG, "bordercolor": "#1F2937",
                       "threshold": {"line": {"color": AMBER, "width": 4}, "thickness": 0.85, "value": UMBRAL * 100}}))
            style_figure(fig, height=230).update_layout(margin=dict(l=35, r=40, t=45, b=15))
            st.plotly_chart(fig, width="stretch")
            st.markdown("#### ⚠️ Alerta de cancelación" if can["alerta"] else "#### ✅ Por debajo del umbral")
            st.write(f"Umbral fijado en validación: **{fmt_num(UMBRAL * 100, 1)}** (línea ámbar).")
            st.caption(f"En validación, de {fmt_num(banda['n'])} reservas con puntaje entre {banda['desde'] * 100:.0f} y "
                       f"{banda['hasta'] * 100:.0f} canceló el **{fmt_pct(banda['tasa_observada'])}**. El puntaje ordena el "
                       "riesgo; no es una probabilidad calibrada.")
        with c2, st.container(border=True):
            st.markdown("#### 💶 Tarifa estimada")
            kpi_card("ADR estimado", fmt_eur(adr_est), "por noche")
            kpi_card("Valor de la estancia", fmt_eur(adr_est * noches), f"{noches} noches × ADR estimado")
            if not u["usar_estimado"]:
                pactado = float(u["fila"]["adr"].iloc[0])
                dif = pactado - adr_est
                st.write(f"El ADR pactado ({fmt_eur(pactado)}) está **{fmt_eur(abs(dif))} "
                         f"{'por encima' if dif >= 0 else 'por debajo'}** de lo estimado.")
            st.caption(f"Error medio del modelo en prueba: ± {fmt_eur(MR['MAE (€)'])}; una reserva concreta puede desviarse más.")
        with c3, st.container(border=True):
            p = PERFILES[clu]
            st.markdown("#### 🧩 Segmento")
            st.markdown(f"**{clu} · {NOMBRES_CLUSTER[clu]}**")
            kpi_card("Cancelación del segmento", fmt_pct(p["cancelacion"]), "media en entrenamiento")
            st.write(f"ADR medio: **{fmt_eur(p['adr'])}** · Anticipación media: **{fmt_num(p['lead_time'])} días**")
            st.write(f"Peso: **{fmt_pct(p['% reservas'])}** de las reservas de entrenamiento")
            st.caption("La pestaña Segmentación marca esta reserva en el mapa PCA.")
        izq, der = st.columns(2)
        with izq:
            st.plotly_chart(grafico_factores(can["factores"], "Qué mueve el riesgo de esta reserva", "log-odds", 2), width="stretch")
        with der:
            st.plotly_chart(grafico_factores(u["adr"]["factores"], "Qué mueve la tarifa estimada", "€", 1), width="stretch")
        st.caption("Contribuciones SHAP sumadas por variable. Describen el comportamiento del modelo, no relaciones causales. "
                   "El resultado apoya la decisión; no garantiza el comportamiento de una reserva individual.")

# -----------------------------------------------------------------------------
# 3. Segmentación
# -----------------------------------------------------------------------------
with tab_cluster:
    st.markdown("### Segmentación de reservas con K-Means (K = 6)")
    variables = metadata["clustering"]["variables"]
    modelos_pca = get_models()["clustering"]["pca"]
    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
    vista = c1.radio("Vista", ["PCA (2D)", "Variables (2D)", "Variables (3D)"], horizontal=True)
    es_pca = vista == "PCA (2D)"
    x_var = c2.selectbox("Eje X", variables, index=0, format_func=ETIQUETAS.get, disabled=es_pca)
    y_var = c3.selectbox("Eje Y", variables, index=1, format_func=ETIQUETAS.get, disabled=es_pca)
    z_var = c4.selectbox("Eje Z", variables, index=2, format_func=ETIQUETAS.get, disabled=vista != "Variables (3D)")
    if es_pca:
        x_var, y_var = "pca1", "pca2"
    resaltar = st.multiselect("Resaltar clústeres (el resto queda en gris)", list(NOMBRES_CLUSTER), placeholder="Todos en color",
                              format_func=lambda k: f"{k} · {NOMBRES_CLUSTER[k]}")

    if filtered.empty:
        st.warning("Los filtros actuales no devuelven reservas.")
    else:
        muestra = filtered.sample(min(len(filtered), 6000), random_state=42)
        ultima = st.session_state.get("ultima")
        tres_d = vista == "Variables (3D)"
        fig = go.Figure()
        # Los grises primero, para que los resaltados queden encima
        for k in sorted(NOMBRES_CLUSTER, key=lambda i: (bool(resaltar) and i in resaltar, i)):
            d = muestra[muestra["cluster"] == k]
            if d.empty:
                continue
            activo = not resaltar or k in resaltar
            comun = dict(name=f"{k} · {NOMBRES_CLUSTER[k]}", mode="markers", legendrank=k,
                         customdata=d[["adr", "lead_time", "total_noches", "total_huespedes"]].to_numpy(),
                         hovertemplate="ADR %{customdata[0]:.2f} €<br>Anticipación %{customdata[1]} días<br>%{customdata[2]} noches · "
                                       "%{customdata[3]} huéspedes<extra>" + f"{k} · {NOMBRES_CLUSTER[k]}" + "</extra>")
            color = PALETA[k] if activo else GRIS
            opacidad = 0.65 if activo else 0.18
            if tres_d:
                fig.add_trace(go.Scatter3d(x=d[x_var], y=d[y_var], z=d[z_var], **comun,
                                           marker=dict(size=1.6 if k == 4 else 3, color=color, opacity=opacidad,
                                                       symbol=SIMBOLOS_3D[k])))
            else:
                fig.add_trace(go.Scattergl(x=d[x_var], y=d[y_var], **comun,
                                           marker=dict(size=6, color=color, opacity=opacidad, symbol=SIMBOLOS[k])))
        if not tres_d:
            centros = muestra.groupby("cluster")[[x_var, y_var]].mean()
            amplitud = muestra[[x_var, y_var]].quantile(0.99) - muestra[[x_var, y_var]].quantile(0.01)
            for k, fila_c in centros.iterrows():
                cercanos = centros.loc[centros.index < k]          # separa los rótulos de centroides casi iguales
                encimado = (((cercanos - fila_c).abs() / amplitud.replace(0, 1)).max(axis=1) < 0.04).sum()
                fig.add_annotation(x=fila_c[x_var], y=fila_c[y_var], text=f"<b>{k}</b>", showarrow=False, xshift=26 * encimado,
                                   font=dict(color=TEXT, size=14), bgcolor="rgba(7,11,20,.85)", bordercolor=PALETA[k], borderpad=3)
        if ultima is not None:
            valores = ultima["cluster"] if es_pca else ultima["fila"].iloc[0]
            if tres_d:
                fig.add_trace(go.Scatter3d(x=[valores[x_var]], y=[valores[y_var]], z=[valores[z_var]], mode="markers",
                                           name="Reserva consultada", legendrank=99,
                                           marker=dict(size=9, color="#FFFFFF", symbol="diamond")))
            else:
                # Anotación (y no un punto): queda por encima de los puntos dibujados con WebGL
                fig.add_annotation(x=valores[x_var], y=valores[y_var], text="<b>★ Reserva consultada</b>", showarrow=True,
                                   arrowhead=2, arrowcolor="#FFFFFF", arrowwidth=2, ax=70, ay=-110,
                                   font=dict(color="#FFFFFF", size=13), bgcolor="rgba(7,11,20,.9)", bordercolor="#FFFFFF", borderpad=4)
        titulo = f"Mapa de clústeres · muestra de {fmt_num(len(muestra))} de {fmt_num(len(filtered))} reservas filtradas"
        style_figure(fig, titulo, 560)
        fig.update_layout(legend=dict(title="Clúster", itemsizing="constant"))
        if tres_d:
            fig.update_layout(scene=dict(xaxis_title=ETIQUETAS[x_var], yaxis_title=ETIQUETAS[y_var], zaxis_title=ETIQUETAS[z_var],
                                         bgcolor="rgba(0,0,0,0)", camera=dict(eye=dict(x=1.5, y=1.5, z=0.9))))
        else:
            fig.update_xaxes(title=ETIQUETAS[x_var])
            fig.update_yaxes(title=ETIQUETAS[y_var])
        st.plotly_chart(fig, width="stretch")
        var = modelos_pca.explained_variance_ratio_
        nota_pca = (f"El PCA resume seis variables en dos ejes que conservan el {fmt_pct(var.sum())} de la varianza, por lo que los "
                    "grupos se solapan en el plano. ") if es_pca else ""
        st.caption(nota_pca + f"La silueta es moderada ({fmt_num(met_cluster['silueta_test'], 3)}): los clústeres son zonas "
                   "de un continuo, no grupos separados.")

    st.markdown("#### Perfiles numéricos promedio")
    perfil_df = pd.DataFrame([
        {"Clúster": k, "Nombre": NOMBRES_CLUSTER[k], "Reservas": p["n"], "% res.": p["% reservas"] * 100, "Anticip. (días)": p["lead_time"],
         "ADR (€)": p["adr"], "Noches": p["total_noches"], "Huéspedes": p["total_huespedes"],
         "Fin sem. (%)": p["prop_fin_semana"] * 100, "Solicitudes": p["peticiones"],
         "Cancel. (%)": p["cancelacion"] * 100, "Ingreso (€)": p["ingreso_estancia"]}
        for k, p in sorted(PERFILES.items())])
    estilo = perfil_df.style.format({c: "{:,.1f}" for c in ["% res.", "ADR (€)", "Fin sem. (%)", "Cancel. (%)"]}
                                    | {c: "{:,.2f}" for c in ["Noches", "Huéspedes", "Solicitudes"]}
                                    | {c: "{:,.0f}" for c in ["Reservas", "Anticip. (días)", "Ingreso (€)"]},
                                    decimal=",", thousands=".")
    st.dataframe(estilo, width="stretch", hide_index=True)
    st.caption("Promedios del conjunto de entrenamiento (83.610 reservas); no cambian con los filtros. Ingreso = ADR medio × noches medias.")

    fig = go.Figure()
    for k, p in sorted(PERFILES.items()):
        fig.add_trace(go.Scatter(x=[p["adr"]], y=[p["cancelacion"]], mode="markers+text", text=[f"{k} · {NOMBRES_CLUSTER[k]}"],
                                 textposition={1: "middle left", 2: "middle right"}.get(k, "top center"), marker=dict(size=18 + 50 * p["% reservas"] ** .5, color=PALETA[k],
                                                                        line=dict(color=DARK_BG, width=2)),
                                 hovertemplate=f"ADR {fmt_eur(p['adr'])}<br>Cancelación {fmt_pct(p['cancelacion'])}<br>"
                                               f"{fmt_num(p['n'])} reservas<extra>{NOMBRES_CLUSTER[k]}</extra>"))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="ADR medio (€)", range=[60, 210])
    fig.update_yaxes(title="Tasa de cancelación", tickformat=".0%", range=[0, 0.7])
    st.plotly_chart(style_figure(fig, "Riesgo y valor por segmento (tamaño = peso)", 430), width="stretch")

# -----------------------------------------------------------------------------
# 4. Modelos
# -----------------------------------------------------------------------------
with tab_models:
    st.markdown("### Comparación de modelos en el conjunto de prueba")
    izq, der = st.columns(2)
    with izq:
        st.markdown("#### Clasificación: cancelación")
        st.dataframe(metrics_class[["Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]]
                     .style.format(precision=3, decimal=","), width="stretch", hide_index=True)
        largo = metrics_class.assign(Modelo=metrics_class["Modelo"].str.replace(" (", "<br>(", regex=False)).melt(id_vars="Modelo", value_vars=["F1", "ROC-AUC", "PR-AUC"], var_name="Métrica", value_name="Valor")
        fig = px.bar(largo, x="Modelo", y="Valor", color="Métrica", barmode="group", color_discrete_sequence=PALETA)
        fig.update_layout(bargap=0.3, bargroupgap=0.08)
        fig.update_yaxes(range=[0, 1], title="")
        fig.update_xaxes(tickangle=0, title="")
        st.plotly_chart(style_figure(fig, "Desempeño de clasificación"), width="stretch")
    with der:
        st.markdown("#### Regresión: ADR")
        st.dataframe(metrics_reg[["Modelo", "MAE (€)", "RMSE (€)", "R2", "R2 val"]]
                     .style.format(precision=3, decimal=","), width="stretch", hide_index=True)
        fig = px.bar(metrics_reg, x="RMSE (€)", y="Modelo", orientation="h", text=[fmt_num(v, 1) for v in metrics_reg["RMSE (€)"]])
        fig.update_traces(marker_color=AZUL, textposition="outside", cliponaxis=False)
        fig.update_yaxes(autorange="reversed", title="")
        st.plotly_chart(style_figure(fig, "Error (RMSE) de regresión: menor es mejor"), width="stretch")

    st.markdown("#### Qué significa por cada 1.000 reservas")
    n_prueba, n_canceladas = 17670, 6243               # conjunto de prueba (Fase 2)
    tp = MC["Recall"] * n_canceladas
    fn, fp = n_canceladas - tp, tp / MC["Precision"] - tp
    cols = st.columns(3)
    for col, (t, v, n) in zip(cols, [("Cancelaciones detectadas", tp, "de 353 que ocurren"), ("No detectadas", fn, "cancelan sin alerta"),
                                     ("Falsas alarmas", fp, "alerta, pero sí llegan")]):
        with col:
            kpi_card(t, fmt_num(v / n_prueba * 1000), n)

    st.markdown("#### Límites que debe conocer quien use el modelo")
    izq, der = st.columns(2)
    with izq, st.container(border=True):
        st.markdown("**Deterioro en el tiempo.** Al entrenar con 2015–2016 y evaluar en 2017, el rendimiento baja: "
                    "el modelo necesita reentrenarse y recalibrar el umbral periódicamente.")
        st.dataframe(pd.DataFrame({"Métrica": ["ROC-AUC cancelación", "Recall cancelación", "R² ADR", "MAE ADR (€)"],
                                   "Partición aleatoria": [MC["ROC-AUC"], MC["Recall"], MR["R2"], MR["MAE (€)"]],
                                   "Evaluado en 2017": [0.881, 0.628, 0.756, 18.491]}).style.format(precision=3, decimal=","),
                     width="stretch", hide_index=True)
    with der, st.container(border=True):
        st.markdown("**Dependencia del país.** Sin `country`, el ROC-AUC de validación cae de 0,949 a 0,903. El efecto se concentra en "
                    "Portugal (57,4 % de cancelación en entrenamiento). Si el país se completa en el check-in, habría fuga de "
                    "información: debe verificarse que se conozca al reservar.")
        st.markdown("**Depósito «Non Refund».** Casi siempre cancela porque refleja bloques de grupos y operadores; no es un efecto causal.")
        st.markdown("**Alcance.** Dos hoteles de Portugal entre 2015 y 2017; otros mercados requieren validación externa.")

    with st.expander("Evidencias gráficas de la Fase 3 (SHAP, curvas, codo y silueta, MLP)"):
        galeria = [("9_1_sintesis_prueba.png", "Síntesis en prueba"), ("4_2_roc_pr_validacion.png", "Curvas ROC y PR en validación"),
                   ("7_1_shap_beeswarm.png", "SHAP de clasificación"), ("7_2_shap_regresion.png", "SHAP de regresión"),
                   ("6_1_codo_silueta.png", "Selección de K: codo y silueta"), ("5_1_mlp_curvas_perdida.png", "Función de pérdida del MLP (PyTorch)")]
        cols = st.columns(2)
        for i, (archivo, pie) in enumerate(galeria):
            cols[i % 2].image(str(ROOT / "figures" / archivo), caption=pie, width="stretch")

# -----------------------------------------------------------------------------
# 5. Proyecto
# -----------------------------------------------------------------------------
with tab_project:
    st.markdown("### Trazabilidad del Proyecto Integrador")
    fases = [("Fase 1 · EDA", "Problema, objetivos SMART, ficha técnica, calidad de datos y análisis exploratorio."),
             ("Fase 2 · Preparación", "Limpieza, variables derivadas, partición agrupada 70/15/15 y baselines sin fuga."),
             ("Fase 3 · Modelado", "Regresión, clasificación, MLP, clustering, optimización e interpretabilidad."),
             ("Fase 4 · Producto", "Dashboard interactivo, informe final, repositorio y sustentación.")]
    for col, (titulo, texto) in zip(st.columns(4), fases):
        with col, st.container(border=True):
            st.markdown(f"**{titulo}**")
            st.caption(texto)
    st.markdown(f"""
#### Cómo se conectan los datos y los modelos
- **Datos:** Hotel Booking Demand (Antonio, Almeida y Nunes, 2019; Kaggle, CC BY 4.0): 119.390 registros originales y {fmt_num(len(data))} tras la depuración.
- **Modelos:** son los entrenados en la Fase 3 (Google Colab, 23 de noviembre de 2026). Se cargan en formato nativo de XGBoost, sin reentrenar, y `src/smoke_test.py` comprueba que reproducen sus métricas de prueba.
- **Métricas:** todas proceden del conjunto de prueba o, cuando se indica, de validación.
""")
    st.markdown("**Integrantes:** Miguel Angel Ricardo · Jose Alexander Barahona Calderon")
    st.caption("Universidad de La Salle · Maestría en Inteligencia Artificial · Ciencia de Datos · Docente Fabián Camilo Castro Riveros")
