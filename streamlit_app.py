# streamlit_app.py — somente "outputs/" (sem caixa de confirmação)
from pathlib import Path
from typing import Any, Dict, List
import json
import traceback

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================
# CONFIG GERAL
# ============================
st.set_page_config(page_title="Previsão de Vendas", layout="centered")
st.title("🔮 Previsão de Vendas – App Streamlit")

# ============================
# CAMINHOS / NOMES-ALVO
# ============================
APP_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = APP_DIR / "outputs"

MODEL_PATH = OUTPUTS_DIR / "best_model_pipeline.pkl"
SCHEMA_PATH = OUTPUTS_DIR / "schema.json"
FALLBACK_PATH = OUTPUTS_DIR / "fallback_stats.json"

# ============================
# LOADERS
# ============================
def _missing(paths):
    return [str(p) for p in paths if not p.exists()]

@st.cache_resource(show_spinner=False)
def load_from_outputs() -> tuple:
    """
    Carrega os três arquivos diretamente de outputs/.
    """
    missing = _missing([MODEL_PATH, SCHEMA_PATH, FALLBACK_PATH])
    if missing:
        raise FileNotFoundError(
            "Arquivos não encontrados em outputs/: "
            + ", ".join(missing)
            + "\nDica: garanta que eles existam e estejam versionados no Git."
            "\nEsperados: best_model_pipeline.pkl, schema.json, fallback_stats.json."
        )

    model = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    with open(FALLBACK_PATH, "r", encoding="utf-8") as f:
        fallback = json.load(f)

    return model, schema, fallback

# ============================
# FEATURE HELPERS
# ============================
def add_calendar_features_inplace(df: pd.DataFrame, date_col: str) -> None:
    """
    Normaliza para o 1º dia do mês e cria:
    - year, month, quarter
    - codificação cíclica month_sin / month_cos
    """
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[date_col] = df[date_col].values.astype("datetime64[M]")  # primeiro dia do mês
    df["year"] = df[date_col].dt.year.astype("int16")
    df["month"] = df[date_col].dt.month.astype("int8")
    df["quarter"] = df[date_col].dt.quarter.astype("int8")
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype("float32")

def fill_and_order_features(
    row: pd.DataFrame,
    schema: Dict[str, Any],
    fallbacks: Dict[str, Any],
) -> pd.DataFrame:
    """
    - Garante presença de todas as colunas de treino (schema["features"])
    - Preenche faltantes com fallbacks (medianas numéricas e 'missing' para categóricas)
    - Reordena as colunas exatamente como no treino
    """
    features: List[str] = schema["features"]
    categoricals: List[str] = schema.get("categoricals", [])

    num_medians = fallbacks.get("numeric_median", {})
    cat_default = fallbacks.get("categorical_default", "missing")

    for col in features:
        if col not in row.columns:
            if col in categoricals:
                row[col] = cat_default
            else:
                row[col] = num_medians.get(col, 0.0)

    # Tipagem categórica conforme treino
    for col in categoricals:
        if col in row.columns:
            row[col] = row[col].astype("category")

    return row[features].copy()

def prepare_sample(
    inputs: Dict[str, Any],
    schema: Dict[str, Any],
    fallbacks: Dict[str, Any],
) -> pd.DataFrame:
    """
    - Recebe um dicionário com (date_col, item, store, etc.)
    - Cria features de calendário
    - Preenche e ordena as colunas conforme schema
    """
    date_col = schema["date_col"]
    df = pd.DataFrame([inputs])

    if date_col in df.columns:
        add_calendar_features_inplace(df, date_col)

    X = fill_and_order_features(df, schema, fallbacks)
    return X

# ============================
# SIDEBAR (INFO)
# ============================
st.sidebar.header("Arquivos do Modelo (somente outputs/)")
st.sidebar.markdown(
    "- `outputs/best_model_pipeline.pkl`\n"
    "- `outputs/schema.json`\n"
    "- `outputs/fallback_stats.json`"
)

# ============================
# CARREGAMENTO
# ============================
try:
    model, schema, fallback = load_from_outputs()
except Exception as e:
    st.sidebar.error(f"Erro ao carregar arquivos: {e}")
    with st.sidebar.expander("Detalhes do erro"):
        st.sidebar.code(traceback.format_exc())
    st.stop()

# ============================
# MAIN – FORM (MÊS/ANO)
# ============================
st.subheader("Previsão unitária")

# nomes típicos (ajuste se no seu schema for diferente)
date_col = schema.get("date_col", "year_month")
grain_item = schema.get("grain_item", "item")
grain_store = schema.get("grain_store", "store")

with st.form("form_pred"):
    st.write("**Campos mínimos:** Ano, Mês, item e store. Os demais são opcionais.")

    # --- PERÍODO: MÊS e ANO ---
    col_y, col_m = st.columns(2)
    today = pd.Timestamp.today()
    ano = col_y.number_input(
        label="Ano",
        min_value=2000,
        max_value=2100,
        value=int(today.year),
        step=1,
        key="ano_sel"
    )
    mes = col_m.selectbox(
        label="Mês",
        options=list(range(1, 13)),
        index=int(today.month) - 1,
        format_func=lambda m: f"{m:02d}",
        key="mes_sel"
    )
    periodo_str = f"{int(ano):04d}-{int(mes):02d}-01"  # YYYY-MM-01

    # Par item×loja
    item = st.text_input(grain_item)
    store = st.text_input(grain_store)

    st.write("**Opcionais (se não preencher, usamos fallbacks):**")
    mean_price = st.number_input("mean_price", min_value=0.0, value=0.0, step=0.01)
    region = st.text_input("region")
    category = st.text_input("category")
    department = st.text_input("department")
    store_code = st.text_input("store_code")
    cluster = st.text_input("cluster")
    lag1 = st.number_input("lag1 (vendas do mês anterior)", min_value=0.0, value=0.0, step=1.0)

    submitted = st.form_submit_button("Prever")

if submitted:
    if not item or not store:
        st.error(f"Informe '{grain_item}' e '{grain_store}'.")
        st.stop()

    sample: Dict[str, Any] = {
        date_col: periodo_str,
        grain_item: item,
        grain_store: store,
    }
    # opcionais
    if region:
        sample["region"] = region
    if category:
        sample["category"] = category
    if department:
        sample["department"] = department
    if store_code:
        sample["store_code"] = store_code
    if cluster:
        sample["cluster"] = cluster
    if mean_price and mean_price > 0:
        sample["mean_price"] = float(mean_price)
    if lag1 and lag1 > 0:
        sample["lag1"] = float(lag1)

    try:
        X = prepare_sample(sample, schema, fallback)
        pred = float(model.predict(X)[0])
        st.success(f"Previsão de vendas: {pred:,.2f}")
        with st.expander("Ver features enviadas"):
            st.dataframe(X)
    except Exception as e:
        st.error(f"Erro na previsão: {e}")
        with st.expander("Detalhes do erro"):
            st.code(traceback.format_exc())
