# src/scripts/dashboard.py
import streamlit as st
import connectorx as cx
import polars as pl
import pandas as pd
from src.config.settings import settings

st.set_page_config(page_title="Geoxiao Evolution Monitor", layout="wide")

# Coba import autorefresh, jika tidak ada, abaikan gracefully
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=10000, key="datarefresh")  # Refresh tiap 10 detik
except ImportError:
    pass

st.title("🧬 Geoxiao Evolution Dashboard")
st.markdown("Autonomous Evolutionary Strategy Search Engine")

@st.cache_data(ttl=5)
def fetch_data():
    """Fetch live data from PostgreSQL via connectorx."""
    try:
        query_dna = "SELECT * FROM strategy_dna ORDER BY created_at DESC"
        query_trials = "SELECT * FROM trial_logs ORDER BY created_at DESC"
        
        df_dna = cx.read_sql(settings.PG_DSN_SYNC, query_dna, return_type='pandas')
        df_trials = cx.read_sql(settings.PG_DSN_SYNC, query_trials, return_type='pandas')
        return df_dna, df_trials
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return pd.DataFrame(), pd.DataFrame()

df_dna, df_trials = fetch_data()

if df_dna.empty:
    st.warning("Belum ada data evolusi. Jalankan run_evolution.py terlebih dahulu.")
else:
    # ── KPI Metrics ──
    latest_gen = df_dna['generation'].max()
    total_evaluated = len(df_trials)
    best_fitness = df_trials['fitness_score'].max() if not df_trials.empty else 0.0
    passed_count = len(df_dna[df_dna['status'] == 'passed'])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Generation", f"{latest_gen}")
    col2.metric("Total Strategies Evaluated", f"{total_evaluated}")
    col3.metric("Highest Fitness Score", f"{best_fitness:.4f}" if pd.notnull(best_fitness) else "N/A")
    col4.metric("Strategies Passed Hard Constraints", f"{passed_count}")

    # ── Fitness Chart ──
    st.subheader("📈 Fitness Evolution per Generation")
    if not df_trials.empty and 'fitness_score' in df_trials.columns:
        # Merge untuk mendapatkan generasi dari trial
        df_merged = df_trials.merge(df_dna[['id', 'generation']], left_on='strategy_id', right_on='id', how='left')
        if not df_merged.empty:
            gen_fitness = df_merged.groupby('generation')['fitness_score'].max().reset_index()
            st.line_chart(gen_fitness.set_index('generation')['fitness_score'])
            
    # ── Hall of Fame (Top 10) ──
    st.subheader("🏆 Hall of Fame (Top 10 Strategies)")
    if not df_trials.empty:
        top_10 = df_trials.nlargest(10, 'fitness_score')
        top_10_display = top_10[['study_name', 'fitness_score', 'profit_factor', 'max_drawdown_pips', 'trade_count', 'duration_sec']]
        st.dataframe(top_10_display, use_container_width=True)
        
    # ── GP Tree Viewer ──
    st.subheader("🌳 Best GP Tree Representation")
    if not top_10.empty:
        best_strategy_id = top_10.iloc[0]['strategy_id']
        best_dna = df_dna[df_dna['id'] == best_strategy_id]
        if not best_dna.empty:
            st.code(best_dna.iloc[0]['tree_repr'], language='lisp')
            st.json(best_dna.iloc[0]['params_json'])
