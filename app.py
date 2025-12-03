import os
import tempfile
from pathlib import Path
from textwrap import wrap

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import squarify
from fpdf import FPDF, FPDFException
import streamlit as st

# ======================================
# CONFIGURACIÓN GENERAL
# ======================================

DATA_PATH = Path("data/ingresos_municipios.csv")

st.set_page_config(
    page_title="Informe de ingresos municipales",
    layout="wide",
)

# Estilo Matplotlib tipo The Economist
plt.rcParams.update({
    "font.family": "DejaVu Sans",  # Sans serif limpia
    "axes.labelsize": 9,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

# Paleta tipo The Economist
ECONOMIST_RED = "#E3120B"
ECONOMIST_BLUE = "#006BA2"
ECONOMIST_GREEN = "#28A197"
ECONOMIST_YELLOW = "#FFBF00"
ECONOMIST_GREY = "#A7A8AA"
ECONOMIST_PALETTE = [
    ECONOMIST_RED,
    ECONOMIST_BLUE,
    ECONOMIST_GREEN,
    ECONOMIST_YELLOW,
    ECONOMIST_GREY,
]

GRID_COLOR = "#D0D0D0"
TEXT_COLOR = "#333333"


# ======================================
# FUNCIONES DE ESTILO
# ======================================

def apply_economist_style(ax):
    """Aplica estilo Economist a un Axes de Matplotlib."""
    ax.set_facecolor("white")
    if ax.figure is not None:
        ax.figure.set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(TEXT_COLOR)
    ax.spines["bottom"].set_color(TEXT_COLOR)

    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    ax.grid(False, axis="x")

    ax.tick_params(
        direction="out",
        length=4,
        width=0.8,
        colors=TEXT_COLOR,
        axis="both",
    )

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(TEXT_COLOR)

    ax.title.set_fontsize(11)
    ax.title.set_weight("bold")
    ax.title.set_color(TEXT_COLOR)


def economist_plotly_layout(fig, colorway=None):
    """Aplica layout tipo Economist a una figura de Plotly."""
    if colorway is None:
        colorway = [ECONOMIST_RED, ECONOMIST_BLUE, ECONOMIST_GREEN, ECONOMIST_YELLOW]

    fig.update_layout(
        template="simple_white",
        font=dict(
            family="Helvetica, Arial, sans-serif",
            size=12,
            color=TEXT_COLOR,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        colorway=colorway,
        xaxis=dict(
            showgrid=False,
            linecolor=TEXT_COLOR,
            ticks="outside",
            tickcolor=TEXT_COLOR,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            linecolor=TEXT_COLOR,
            ticks="outside",
            tickcolor=TEXT_COLOR,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ======================================
# CARGA Y PREPARACIÓN DE DATOS
# ======================================

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Año"] = df["Año"].astype(int)
    df["TotalRecaudo"] = pd.to_numeric(df["TotalRecaudo"], errors="coerce").fillna(0)

    df["tipo_norm"] = (
        df["Tipo de Entidad"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    return df


@st.cache_data
def calcular_crecimiento_promedio_municipios(df: pd.DataFrame):
    """Crecimiento promedio 2021-2024 del ingreso total para municipios."""
    df_mun = df[df["tipo_norm"].str.contains("MUNICIPIO", na=False)].copy()
    if df_mun.empty:
        return None

    g = df_mun.groupby(["codigo_alt", "Año"], as_index=False)["TotalRecaudo"].sum()
    pvt = g.pivot(index="codigo_alt", columns="Año", values="TotalRecaudo")

    if 2021 not in pvt.columns or 2024 not in pvt.columns:
        return None

    valid = pvt[(pvt[2021] > 0) & (pvt[2024] > 0)].copy()
    if valid.empty:
        return None

    valid["growth"] = (valid[2024] / valid[2021] - 1) * 100
    return float(valid["growth"].mean())


def obtener_series_municipio(df: pd.DataFrame, entidad: str, departamento: str):
    df_muni = df[
        (df["Entidad"] == entidad) &
        (df["Departamento"] == departamento)
    ].copy()

    ts_total = (
        df_muni.groupby("Año", as_index=False)["TotalRecaudo"]
        .sum()
        .sort_values("Año")
    )

    df_area = (
        df_muni.groupby(["Año", "clas_gen"], as_index=False)["TotalRecaudo"]
        .sum()
        .sort_values(["Año", "clas_gen"])
    )

    df_2024 = df_muni[df_muni["Año"] == 2024].copy()
    if not df_2024.empty:
        df_2024 = (
            df_2024.groupby(["clas_gen", "clasificacion_ofpuj"], as_index=False)["TotalRecaudo"]
            .sum()
        )

    return df_muni, ts_total, df_area, df_2024


def calcular_crecimiento_municipio(ts_total: pd.DataFrame):
    if ts_total.empty:
        return None

    try:
        base_2021 = float(ts_total.loc[ts_total["Año"] == 2021, "TotalRecaudo"].iloc[0])
        fin_2024 = float(ts_total.loc[ts_total["Año"] == 2024, "TotalRecaudo"].iloc[0])
    except IndexError:
        return None

    if base_2021 <= 0:
        return None

    return (fin_2024 / base_2021 - 1) * 100


def composicion_por_clas_gen(df_muni: pd.DataFrame, year: int):
    df_year = df_muni[df_muni["Año"] == year]
    if df_year.empty:
        return {}

    comp = df_year.groupby("clas_gen")["TotalRecaudo"].sum()
    total = comp.sum()
    if total <= 0:
        return {}

    return (comp / total * 100).round(1).to_dict()


def top3_fuentes_ofpuj(df_2024: pd.DataFrame):
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
    # 1) Crecimiento
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
            "- No hay información suficiente para comparar con el promedio de los municipios.",
        ]
    else:
        lineas1 = [
            "- No hay información suficiente para calcular el crecimiento del ingreso entre 2021 y 2024 para este municipio.",
        ]

    # 2) Composición
    if comp_2021:
        comp2021_str = "; ".join(f"{k}: {v:.1f}%" for k, v in comp_2021.items())
        linea_2021 = f"- En 2021, la estructura del ingreso por tipo (clas_gen) era: {comp2021_str}."
    else:
        linea_2021 = "- No hay información suficiente para describir la composición del ingreso en 2021."

    if comp_2024:
        comp2024_str = "; ".join(f"{k}: {v:.1f}%" for k, v in comp_2024.items())
        linea_2024 = f"- En 2024, la estructura del ingreso por tipo (clas_gen) es: {comp2024_str}."
    else:
        linea_2024 = "- No hay información suficiente para describir la composición del ingreso en 2024."

    # 3) Top 3 OFPUJ
    if top3:
        lineas3 = [
            "- En 2024, las tres principales fuentes de ingreso según la clasificación OFPUJ son:"
        ]
        for i, (nombre, share) in enumerate(top3, start=1):
            lineas3.append(f"  {i}. {nombre}: {share:.1f}% del ingreso total de 2024.")
    else:
        lineas3 = [
            "- No se pudo identificar las principales fuentes de ingreso para 2024 (sin información suficiente).",
        ]

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


# ======================================
# GRÁFICOS INTERACTIVOS (PLOTLY)
# ======================================

def crear_graficos_plotly(ts_total, df_area, df_2024, entidad):
    # Línea ingreso total
    fig_line = px.line(
        ts_total,
        x="Año",
        y="TotalRecaudo",
        markers=True,
        title=f"Ingreso total {entidad} (2021-2024)",
    )
    fig_line.update_traces(line=dict(width=2, color=ECONOMIST_RED))
    fig_line = economist_plotly_layout(fig_line, colorway=[ECONOMIST_RED])

    # Área relativa clas_gen
    fig_area = px.area(
        df_area,
        x="Año",
        y="TotalRecaudo",
        color="clas_gen",
        groupnorm="percent",
        title=f"Composición relativa del ingreso por tipo (clas_gen) - {entidad}",
    )
    fig_area = economist_plotly_layout(fig_area, colorway=ECONOMIST_PALETTE)
    fig_area.update_yaxes(title="% del total")

    # Treemap 2024 (sólo si suma > 0)
    fig_tree = None
    if not df_2024.empty:
        total_2024 = df_2024["TotalRecaudo"].sum()
        if total_2024 > 0:
            fig_tree = px.treemap(
                df_2024,
                path=["clas_gen", "clasificacion_ofpuj"],
                values="TotalRecaudo",
                title=f"Composición del ingreso 2024 (clas_gen / OFPUJ) - {entidad}",
            )
            fig_tree = economist_plotly_layout(fig_tree, colorway=ECONOMIST_PALETTE)

    return fig_line, fig_area, fig_tree


# ======================================
# GRÁFICOS PARA EL PDF (MATPLOTLIB)
# ======================================

def crear_imagenes_matplotlib(ts_total, df_area, df_2024, entidad, departamento):
    """
    Crea 3 imágenes PNG con estilo Economist usando Matplotlib:
    1) Línea de ingreso total
    2) Área relativa por clas_gen
    3) Treemap composicional 2024 (si es posible)
    Devuelve [path_line, path_area, path_tree]
    """
    temp_paths = []

    # 1) Línea
    if not ts_total.empty:
        tmp_line = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_line.close()

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.plot(
            ts_total["Año"],
            ts_total["TotalRecaudo"],
            marker="o",
            linewidth=2,
            color=ECONOMIST_RED,
        )
        ax.set_title(f"Ingreso total {entidad} ({departamento})")
        ax.set_xlabel("Año")
        ax.set_ylabel("Total recaudo")
        apply_economist_style(ax)
        fig.tight_layout()
        fig.savefig(tmp_line.name, bbox_inches="tight")
        plt.close(fig)

        temp_paths.append(tmp_line.name)
    else:
        temp_paths.append(None)

    # 2) Área relativa clas_gen
    if not df_area.empty:
        pvt = df_area.pivot(index="Año", columns="clas_gen", values="TotalRecaudo").fillna(0)
        totals = pvt.sum(axis=1)
        totals[totals == 0] = np.nan
        pvt_pct = pvt.div(totals, axis=0) * 100

        tmp_area = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_area.close()

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

        cols = list(pvt_pct.columns)
        xs = pvt_pct.index.values
        ys = [pvt_pct[col].fillna(0.0).values for col in cols]
        colors = [ECONOMIST_PALETTE[i % len(ECONOMIST_PALETTE)] for i in range(len(cols))]

        ax.stackplot(xs, ys, labels=cols, colors=colors)
        ax.set_title("Composición relativa del ingreso (clas_gen)")
        ax.set_xlabel("Año")
        ax.set_ylabel("% del total")
        ax.set_ylim(0, 100)
        apply_economist_style(ax)
        ax.legend(loc="upper left", fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(tmp_area.name, bbox_inches="tight")
        plt.close(fig)

        temp_paths.append(tmp_area.name)
    else:
        temp_paths.append(None)

    # 3) Treemap 2024 (sólo si suma > 0 y todas las celdas tienen >0)
    path_tree = None
    if (df_2024 is not None) and (not df_2024.empty):
        df_t = df_2024[df_2024["TotalRecaudo"] > 0].copy()
        if not df_t.empty:
            sizes = df_t["TotalRecaudo"].to_numpy(dtype=float)
            total_sizes = sizes.sum()
            if total_sizes > 0:
                tmp_tree = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp_tree.close()

                df_t["label"] = (
                    df_t["clas_gen"].astype(str)
                    + "\n"
                    + df_t["clasificacion_ofpuj"].astype(str)
                )

                labels = df_t["label"].values
                colors = [
                    ECONOMIST_PALETTE[i % len(ECONOMIST_PALETTE)]
                    for i in range(len(labels))
                ]

                fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
                try:
                    squarify.plot(
                        sizes=sizes,
                        label=labels,
                        color=colors,
                        alpha=0.92,
                        ax=ax,
                        text_kwargs={"fontsize": 7, "color": "white"},
                    )
                    ax.set_title(f"Composición del ingreso 2024 (clas_gen / OFPUJ) - {entidad}")
                    ax.axis("off")
                    fig.tight_layout()
                    fig.savefig(tmp_tree.name, bbox_inches="tight")
                    path_tree = tmp_tree.name
                except ZeroDivisionError:
                    path_tree = None
                finally:
                    plt.close(fig)

    temp_paths.append(path_tree)

    return temp_paths  # [line, area, treemap]


# ======================================
# GENERACIÓN DEL PDF ESTILO ECONOMIST
# ======================================

def generar_pdf(entidad, departamento, ts_total, df_area, df_2024, texto_resumen):
    """
    Genera un PDF horizontal (A4) de una página con:
    - Barra roja a la izquierda
    - Título
    - Texto resumen
    - Tres gráficos (línea, área, treemap si aplica)
    """
    img_paths = crear_imagenes_matplotlib(ts_total, df_area, df_2024, entidad, departamento)
    path_line, path_area, path_tree = img_paths

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_left_margin(10)
    pdf.set_right_margin(10)
    pdf.set_auto_page_break(auto=False, margin=10)
    pdf.add_page()

    # Barra roja vertical estilo Economist
    pdf.set_fill_color(227, 18, 11)  # ECONOMIST_RED
    pdf.rect(5, 5, 3, 200, style="F")

    # Título principal
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(10, 8)
    titulo = f"Informe de ingresos - {entidad} ({departamento})"
    pdf.cell(0, 10, titulo, ln=1)

    # Subtítulo
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.set_xy(10, 18)
    pdf.cell(0, 5, "Evolución reciente y composición de los ingresos municipales", ln=1)

    # Texto resumen
    pdf.set_xy(10, 26)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)

    max_chars = 110  # ancho máximo en caracteres por línea

    for raw_line in texto_resumen.split("\n"):
        if not raw_line.strip():
            pdf.ln(2)
            continue

        # dividir línea larga en trozos manejables
        wrapped_lines = wrap(raw_line, width=max_chars)
        if not wrapped_lines:
            wrapped_lines = [""]

        for linea in wrapped_lines:
            # asegurar encoding latino básico
            safe_line = linea.encode("latin-1", "replace").decode("latin-1")
            try:
                pdf.multi_cell(0, 4, safe_line)
            except FPDFException:
                # si aún así se queja, recortamos un poco más
                shorter = safe_line[:80]
                pdf.multi_cell(0, 4, shorter)

    # Posición para gráficos
    y_top_plots = 80
    img_w = 130
    img_h = 70

    if path_line is not None:
        pdf.image(path_line, x=10, y=y_top_plots, w=img_w, h=img_h)

    if path_area is not None:
        pdf.image(path_area, x=150, y=y_top_plots, w=img_w, h=img_h)

    if path_tree is not None:
        pdf.image(path_tree, x=60, y=155, w=170)

    pdf_bytes = pdf.output(dest="S").encode("latin-1")

    # Limpiar temporales
    for p in img_paths:
        if p is not None and os.path.exists(p):
            os.remove(p)

    return pdf_bytes


# ======================================
# INTERFAZ STREAMLIT
# ======================================

st.title("Informe automático de ingresos municipales")
st.caption("Estilo tipo The Economist · Ingresos 2021–2024")

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
    df_dep_mun = df_dep[df_dep["tipo_norm"].str.contains("MUNICIPIO", na=False)]
    entidades_dep = sorted(df_dep_mun["Entidad"].dropna().unique())
    ent_sel = st.selectbox("Selecciona un municipio:", entidades_dep)

st.markdown("---")

if ent_sel:
    df_muni, ts_total, df_area, df_2024 = obtener_series_municipio(df, ent_sel, dep_sel)

    if ts_total.empty:
        st.warning("No se encontraron datos para este municipio.")
    else:
        # Gráficos interactivos
        fig_line, fig_area_plot, fig_tree = crear_graficos_plotly(ts_total, df_area, df_2024, ent_sel)

        st.subheader("Vista previa de los gráficos")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_line, use_container_width=True)
        with c2:
            st.plotly_chart(fig_area_plot, use_container_width=True)

        if fig_tree is not None:
            st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.info("No hay información suficiente (o suma/fuentes iguales a cero) para generar el treemap 2024.")

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
                ts_total=ts_total,
                df_area=df_area,
                df_2024=df_2024,
                texto_resumen=texto_resumen,
            )

            st.download_button(
                label="Descargar informe en PDF",
                data=pdf_bytes,
                file_name=f"informe_ingresos_{ent_sel.replace(' ', '_')}.pdf",
                mime="application/pdf",
            )

