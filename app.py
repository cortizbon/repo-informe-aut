import os
import tempfile

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import kaleido

import streamlit as st


# -----------------------------
# CONFIGURACIÓN BÁSICA
# -----------------------------
DATA_PATH = Path("data/ingresos_municipios.csv")

st.set_page_config(
    page_title="Informe de ingresos municipales",
    layout="wide",
)


# -----------------------------
# CARGA DE DATOS
# -----------------------------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Limpieza básica
    df["Año"] = df["Año"].astype(int)
    df["TotalRecaudo"] = pd.to_numeric(df["TotalRecaudo"], errors="coerce").fillna(0)

    # Normalizamos tipo de entidad para filtrar municipios
    df["tipo_norm"] = (
        df["Tipo de Entidad"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    return df


@st.cache_data
def calcular_crecimiento_promedio_municipios(df: pd.DataFrame):
    """
    Calcula el crecimiento promedio 2021-2024 del ingreso total
    para todos los municipios (Tipo de Entidad contiene 'MUNICIPIO').
    Devuelve el promedio en porcentaje o None si no se puede calcular.
    """
    df_mun = df[df["tipo_norm"].str.contains("MUNICIPIO", na=False)].copy()
    if df_mun.empty:
        return None

    g = (
        df_mun.groupby(["codigo_alt", "Año"], as_index=False)["TotalRecaudo"]
        .sum()
    )
    pvt = g.pivot(index="codigo_alt", columns="Año", values="TotalRecaudo")

    if 2021 not in pvt.columns or 2024 not in pvt.columns:
        return None

    # Sólo municipios con info en 2021 y 2024 y base > 0
    valid = pvt[(pvt[2021] > 0) & (pvt[2024] > 0)].copy()
    if valid.empty:
        return None

    valid["growth"] = (valid[2024] / valid[2021] - 1) * 100
    return float(valid["growth"].mean())


# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------
def obtener_series_municipio(df: pd.DataFrame, entidad: str, departamento: str):
    df_muni = df[
        (df["Entidad"] == entidad) &
        (df["Departamento"] == departamento)
    ].copy()

    # Serie total por año
    ts_total = (
        df_muni.groupby("Año", as_index=False)["TotalRecaudo"]
        .sum()
        .sort_values("Año")
    )

    # Composición por clas_gen (para área)
    df_area = (
        df_muni.groupby(["Año", "clas_gen"], as_index=False)["TotalRecaudo"]
        .sum()
        .sort_values(["Año", "clas_gen"])
    )

    # Treemap 2024
    df_2024 = df_muni[df_muni["Año"] == 2024].copy()
    if not df_2024.empty:
        df_2024 = (
            df_2024.groupby(
                ["clas_gen", "clasificacion_ofpuj"], as_index=False
            )["TotalRecaudo"]
            .sum()
        )

    return df_muni, ts_total, df_area, df_2024


def calcular_crecimiento_municipio(ts_total: pd.DataFrame):
    """
    Calcula el crecimiento 2021-2024 para un municipio.
    Devuelve crecimiento en % o None si no hay info suficiente.
    """
    if ts_total.empty:
        return None

    try:
        base_2021 = float(
            ts_total.loc[ts_total["Año"] == 2021, "TotalRecaudo"].iloc[0]
        )
        fin_2024 = float(
            ts_total.loc[ts_total["Año"] == 2024, "TotalRecaudo"].iloc[0]
        )
    except IndexError:
        return None

    if base_2021 <= 0:
        return None

    return (fin_2024 / base_2021 - 1) * 100


def composicion_por_clas_gen(df_muni: pd.DataFrame, year: int):
    """
    Devuelve un dict: {clas_gen: porcentaje} para un año dado.
    """
    df_year = df_muni[df_muni["Año"] == year]
    if df_year.empty:
        return {}

    comp = df_year.groupby("clas_gen")["TotalRecaudo"].sum()
    total = comp.sum()
    if total <= 0:
        return {}

    return (comp / total * 100).round(1).to_dict()


def top3_fuentes_ofpuj(df_2024: pd.DataFrame):
    """
    Devuelve lista de (nombre, porcentaje) de las 3 principales
    clasificacion_ofpuj en 2024.
    """
    if df_2024.empty:
        return []

    total = df_2024["TotalRecaudo"].sum()
    if total <= 0:
        return []

    comp = (
        df_2024.groupby("clasificacion_ofpuj")["TotalRecaudo"]
        .sum()
        .sort_values(ascending=False)
    )
    shares = (comp / total * 100).round(1)
    top = shares.head(3)
    return list(top.items())


def construir_texto_resumen(
    entidad: str,
    departamento: str,
    crecimiento_muni,
    crecimiento_promedio,
    comp_2021,
    comp_2024,
    top3,
):
    # 1. Comparación de crecimiento
    if crecimiento_muni is not None and crecimiento_promedio is not None:
        diff = crecimiento_muni - crecimiento_promedio
        if diff > 5:
            pos_text = "por encima del promedio de los municipios"
        elif diff < -5:
            pos_text = "por debajo del promedio de los municipios"
        else:
            pos_text = "en línea con el promedio de los municipios"

        lineas1 = [
            f"- Entre 2021 y 2024, el ingreso total de {entidad} ({departamento}) creció aproximadamente {crecimiento_muni:.1f}%.",
            f"- En el mismo periodo, el crecimiento promedio de los municipios fue de {crecimiento_promedio:.1f}%, por lo que el desempeño de {entidad} está {pos_text}.",
        ]
    elif crecimiento_muni is not None:
        lineas1 = [
            f"- Entre 2021 y 2024, el ingreso total de {entidad} ({departamento}) creció aproximadamente {crecimiento_muni:.1f}%.",
            "- No hay información suficiente para comparar con el promedio de los municipios."
        ]
    else:
        lineas1 = [
            "- No hay información suficiente para calcular el crecimiento del ingreso entre 2021 y 2024 para este municipio."
        ]

    # 2. Composición 2021 vs 2024
    if comp_2021:
        comp2021_str = "; ".join(
            f"{k}: {v:.1f}%" for k, v in comp_2021.items()
        )
        linea_2021 = f"- En 2021, la estructura del ingreso por tipo (clas_gen) era: {comp2021_str}."
    else:
        linea_2021 = "- No hay información suficiente para describir la composición del ingreso en 2021."

    if comp_2024:
        comp2024_str = "; ".join(
            f"{k}: {v:.1f}%" for k, v in comp_2024.items()
        )
        linea_2024 = f"- En 2024, la estructura del ingreso por tipo (clas_gen) es: {comp2024_str}."
    else:
        linea_2024 = "- No hay información suficiente para describir la composición del ingreso en 2024."

    # 3. Top 3 fuentes 2024
    if top3:
        lineas3 = [
            "- En 2024, las tres principales fuentes de ingreso según la clasificación OFPUJ son:"
        ]
        for i, (nombre, share) in enumerate(top3, start=1):
            lineas3.append(
                f"  {i}. {nombre}: {share:.1f}% del ingreso total de 2024."
            )
    else:
        lineas3 = [
            "- No se pudo identificar las principales fuentes de ingreso para 2024 (sin información suficiente)."
        ]

    # Armamos texto final con numeración
    texto = []
    texto.append("1) Crecimiento del ingreso total (2021-2024):")
    texto.extend(lineas1)
    texto.append("")
    texto.append("2) Composición del ingreso en 2021 y 2024 (clas_gen):")
    texto.append(linea_2021)
    texto.append(linea_2024)
    texto.append("")
    texto.append("3) Desempeño en 2024: principales fuentes de ingreso (clasificación OFPUJ):")
    texto.extend(lineas3)

    return "\n".join(texto)


def crear_graficos(ts_total, df_area, df_2024, entidad):
    # Línea de ingreso total
    fig_line = px.line(
        ts_total,
        x="Año",
        y="TotalRecaudo",
        markers=True,
        title=f"Ingreso total {entidad} (2021-2024)",
    )
    fig_line.update_layout(
        yaxis_title="Total Recaudo",
        xaxis_title="Año",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=30),
    )

    # Área relativa de clas_gen
    fig_area = px.area(
        df_area,
        x="Año",
        y="TotalRecaudo",
        color="clas_gen",
        groupnorm="percent",
        title=f"Composición relativa del ingreso por tipo (clas_gen) - {entidad}",
    )
    fig_area.update_layout(
        yaxis_title="% del total",
        xaxis_title="Año",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=30),
    )

    # Treemap 2024
    if not df_2024.empty:
        fig_tree = px.treemap(
            df_2024,
            path=["clas_gen", "clasificacion_ofpuj"],
            values="TotalRecaudo",
            title=f"Composición del ingreso 2024 (clas_gen / OFPUJ) - {entidad}",
        )
        fig_tree.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    else:
        fig_tree = None

    return fig_line, fig_area, fig_tree


def generar_pdf(entidad, departamento, fig_line, fig_area, fig_tree, texto_resumen):
    """
    Genera un PDF de una página en orientación horizontal (A4),
    con título, texto y tres gráficos (línea, área y treemap).
    Devuelve bytes del PDF.
    """
    import plotly.io as pio

    # Guardamos figuras como imágenes temporales
    tmp_files = []
    figs = [fig_line, fig_area, fig_tree]
    for f in figs:
        if f is None:
            tmp_files.append(None)
            continue
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.close()
        pio.write_image(f, tmp.name, width=1200, height=800, scale=2)
        tmp_files.append(tmp.name)

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False, margin=10)
    pdf.add_page()

    # Título
    pdf.set_font("Helvetica", "B", 16)
    titulo = f"Informe de ingresos - {entidad} ({departamento})"
    pdf.cell(0, 10, titulo, ln=1)

    # Texto resumen (lo intentamos mantener compacto)
    pdf.set_font("Helvetica", size=9)
    for linea in texto_resumen.split("\n"):
        if not linea.strip():
            pdf.ln(2)
        else:
            pdf.multi_cell(0, 4, linea)

    # Posición para los gráficos
    # Ajusta estos valores si ves que se montan en tu PDF
    y_top_plots = 70
    img_w = 130
    img_h = 70

    if tmp_files[0] is not None:
        pdf.image(tmp_files[0], x=10, y=y_top_plots, w=img_w, h=img_h)
    if tmp_files[1] is not None:
        pdf.image(tmp_files[1], x=150, y=y_top_plots, w=img_w, h=img_h)

    # Treemap en la parte inferior central
    if tmp_files[2] is not None:
        pdf.image(tmp_files[2], x=60, y=145, w=170)

    # Generar bytes
    pdf_bytes = pdf.output(dest="S").encode("latin-1")

    # Limpiamos archivos temporales
    for f in tmp_files:
        if f is not None and os.path.exists(f):
            os.remove(f)

    return pdf_bytes


# -----------------------------
# UI DE STREAMLIT
# -----------------------------
st.title("Informe automático de ingresos municipales")

if not DATA_PATH.exists():
    st.error(f"No se encontró el archivo de datos en {DATA_PATH}.")
    st.stop()

df = load_data(DATA_PATH)
crecimiento_promedio = calcular_crecimiento_promedio_municipios(df)

# Selección de departamento y municipio
departamentos = sorted(df["Departamento"].dropna().unique())
col_dep, col_ent = st.columns(2)

with col_dep:
    dep_sel = st.selectbox("Selecciona un departamento:", departamentos)

with col_ent:
    df_dep = df[df["Departamento"] == dep_sel]
    # En la UI mostramos sólo entidades que sean municipios
    df_dep_mun = df_dep[df_dep["tipo_norm"].str.contains("MUNICIPIO", na=False)]
    entidades_dep = sorted(df_dep_mun["Entidad"].dropna().unique())
    ent_sel = st.selectbox("Selecciona un municipio:", entidades_dep)

st.markdown("---")

if ent_sel:
    df_muni, ts_total, df_area, df_2024 = obtener_series_municipio(df, ent_sel, dep_sel)

    if ts_total.empty:
        st.warning("No se encontraron datos para este municipio.")
    else:
        # Gráficos para vista previa
        fig_line, fig_area_plot, fig_tree = crear_graficos(ts_total, df_area, df_2024, ent_sel)

        st.subheader("Vista previa de los gráficos")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_line, use_container_width=True)
        with c2:
            st.plotly_chart(fig_area_plot, use_container_width=True)

        if fig_tree is not None:
            st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.info("No hay información para 2024, por lo que no se genera treemap.")

        # Cálculos para el texto
        crecimiento_muni = calcular_crecimiento_municipio(ts_total)
        comp_2021 = composicion_por_clas_gen(df_muni, 2021)
        comp_2024 = composicion_por_clas_gen(df_muni, 2024)
        top3 = top3_fuentes_ofpuj(df_2024)

        texto_resumen = construir_texto_resumen(
            entidad=ent_sel,
            departamento=dep_sel,
            crecimiento_muni=crecimiento_muni,
            crecimiento_promedio=crecimiento_promedio,
            comp_2021=comp_2021,
            comp_2024=comp_2024,
            top3=top3,
        )

        st.subheader("Resumen estándar del municipio")
        st.text(texto_resumen)

        st.markdown("---")
        if st.button("Generar informe en PDF"):
            pdf_bytes = generar_pdf(
                entidad=ent_sel,
                departamento=dep_sel,
                fig_line=fig_line,
                fig_area=fig_area_plot,
                fig_tree=fig_tree,
                texto_resumen=texto_resumen,
            )

            st.download_button(
                label="Descargar informe en PDF",
                data=pdf_bytes,
                file_name=f"informe_ingresos_{ent_sel.replace(' ', '_')}.pdf",
                mime="application/pdf",
            )
