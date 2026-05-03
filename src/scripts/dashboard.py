# src/scripts/dashboard.py
import streamlit as st
import connectorx as cx
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv
from src.data.repositories.base import postgres_sync_dsn

# Load environment variables
load_dotenv()

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Geoxiao | Strategy Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-Modern CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .main { background: radial-gradient(circle at top right, #1a1f35, #0b0e14); }
    
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #00F2FF, #7000FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 2.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Engine ---
@st.cache_data(ttl=5)
def fetch_master_data():
    try:
        dsn = postgres_sync_dsn()
        # High-level data
        df_dna = cx.read_sql(dsn, "SELECT * FROM strategy_dna ORDER BY created_at DESC", return_type='pandas')
        df_trials = cx.read_sql(dsn, "SELECT * FROM trial_logs ORDER BY fitness_score DESC", return_type='pandas')
        
        # Extended data
        df_monthly = cx.read_sql(dsn, "SELECT * FROM monthly_metrics ORDER BY backtest_month ASC", return_type='pandas')
        
        return df_dna, df_trials, df_monthly
    except Exception as e:
        st.error(f"Data Sync Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=10)
def fetch_trade_logs(trial_id: str):
    try:
        dsn = postgres_sync_dsn()
        query = f"SELECT * FROM trade_logs WHERE trial_id = '{trial_id}' ORDER BY entry_time ASC"
        return cx.read_sql(dsn, query, return_type='pandas')
    except:
        return pd.DataFrame()

# --- Main Logic ---
df_dna, df_trials, df_monthly = fetch_master_data()

st.markdown('<h1 class="gradient-text">Geoxiao Strategy Intelligence</h1>', unsafe_allow_html=True)

if df_dna.empty:
    st.warning("🌌 No genetic data found in the neural pool. Please run `run_evolution.py` to generate the first generation.")
    with st.expander("Diagnostic Info"):
        try:
            dsn = postgres_sync_dsn()
            st.write(f"DSN detected: `{dsn[:20]}...` (len: {len(dsn)})")
        except:
            st.error("Could not resolve DSN.")
else:
    # Sidebar Filters
    with st.sidebar:
        st.header("🧬 Neural Pool Filters")
        st.success("🟢 CORE CONNECTED")
        st.write(f"🧬 DNA Samples: {len(df_dna)}")
        st.write(f"🧪 Total Trials: {len(df_trials)}")
        
        st.markdown("---")
        symbols = df_dna['symbol'].unique() if not df_dna.empty else []
        selected_symbol = st.multiselect("Symbols", symbols, default=symbols)
        
        if not df_trials.empty:
            min_pf = st.slider("Min Profit Factor", 0.0, 5.0, 1.0, 0.1)
        else:
            min_pf = 0.0
        
        st.markdown("---")
        st.caption("Engine: v2.5.0-evolutionary")

    # Filtered Data
    if not df_trials.empty:
        filtered_trials = df_trials[df_trials['profit_factor'] >= min_pf]
    else:
        filtered_trials = pd.DataFrame()
    
    # Navigation
    tab_summary, tab_deep_dive, tab_genetics = st.tabs(["🚀 SYSTEM OVERVIEW", "🔍 STRATEGY DEEP-DIVE", "🌳 GENETIC LINEAGE"])

    target_trial = None # Define globally for the session

    with tab_summary:
        if df_trials.empty:
            st.info("💡 DNA pool is initialized, but no backtest trials have completed yet. Waiting for `run_evolution.py` backtest results...")
            st.dataframe(df_dna[['generation', 'symbol', 'timeframe', 'status']].head(10), width="stretch")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("TOP FITNESS", f"{df_trials['fitness_score'].max():.4f}")
            col2.metric("ELITE POOL", f"{len(df_trials[df_trials['profit_factor'] >= 2.0])} units")
            col3.metric("AVG PROFIT FACTOR", f"{df_trials['profit_factor'].mean():.2f}")
            col4.metric("TOTAL TRIALS", len(df_trials))

            # Convergence Plot
            st.markdown("### 📊 Pool Convergence Trend")
            df_merged = df_trials.merge(df_dna[['id', 'generation']], left_on='strategy_id', right_on='id', how='left')
            gen_stats = df_merged.groupby('generation')['fitness_score'].agg(['max', 'mean']).reset_index()
            
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(x=gen_stats['generation'], y=gen_stats['max'], name="Best", line=dict(color='#00F2FF', width=3)))
            fig_conv.add_trace(go.Scatter(x=gen_stats['generation'], y=gen_stats['mean'], name="Mean", line=dict(color='#7000FF', width=2, dash='dot')))
            fig_conv.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,b=0,t=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_conv, width="stretch")

    with tab_deep_dive:
        if df_trials.empty:
            st.info("Waiting for backtest trials to populate the analyzer...")
        else:
            st.markdown("### 🔍 Select Elite Unit for Decomposition")
            top_selectors = filtered_trials.nlargest(20, 'fitness_score')
            if top_selectors.empty:
                st.warning("⚠️ No strategies match the current filters. Adjust the Profit Factor or Symbol in the sidebar.")
            else:
                selected_study = st.selectbox("Elite Unit ID", top_selectors['study_name'].tolist())
                
                match = df_trials[df_trials['study_name'] == selected_study]
                if not match.empty:
                    target_trial = match.iloc[0]
                    trial_id = target_trial['id']
                    
                    # Detailed Stats Row
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("Profit Factor", f"{target_trial['profit_factor']:.2f}")
                    d2.metric("Total Pips", f"{target_trial['total_pips']:.1f}")
                    d3.metric("Max DD", f"{target_trial['max_drawdown_pips']:.1f} pips")
                    d4.metric("Trade Count", target_trial['trade_count'])

                    # Equity Curve
                    st.markdown("#### 📈 Cumulative Performance (Net Pips)")
                    df_trades = fetch_trade_logs(trial_id)
                    if not df_trades.empty:
                        df_trades['cum_pips'] = df_trades['net_pips'].cumsum()
                        fig_equity = px.line(df_trades, x='entry_time', y='cum_pips', color_discrete_sequence=['#00F2FF'])
                        fig_equity.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_equity, width="stretch")
                    else:
                        st.info("Generating trade data stream...")

    with tab_genetics:
        st.markdown("### 🌳 Genetic Architecture")
        if not df_dna.empty:
            # Determine which DNA to show
            if target_trial is not None:
                strat_dna = df_dna[df_dna['id'] == target_trial['strategy_id']].iloc[0]
                st.info(f"🧬 Showing DNA for Selected Elite Unit: **{target_trial['study_name']}**")
            else:
                dna_id = st.selectbox("Select DNA Sample to Analyze", df_dna['id'].tolist())
                strat_dna = df_dna[df_dna['id'] == dna_id].iloc[0]
            
            g1, g2 = st.columns([1, 1])
            with g1:
                st.markdown("**Symbol Configuration**")
                st.write(f"Symbol: `{strat_dna['symbol']}` | TF: `{strat_dna['timeframe']}`")
                st.write(f"Nodes: {strat_dna['tree_nodes']} | Depth: {strat_dna['tree_depth']}")
                st.markdown("**GP Blueprint (Lisp)**")
                st.code(strat_dna['tree_repr'], language='lisp')
            with g2:
                st.markdown("**Hyperparameters (DNA)**")
                st.json(strat_dna['params_json'])



# Auto Refresh
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=15000, key="global_framer")
except:
    pass


