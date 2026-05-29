import os
import joblib
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drift Monitor · Credit Fraud",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background-color: #0d1117; color: #e6edf3; }

[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* Dark dataframe */
[data-testid="stDataFrame"] iframe { background: #161b22 !important; }
.dvn-scroller { background-color: #161b22 !important; }
.cell-wrapper { background-color: #161b22 !important; color: #e6edf3 !important; }
[data-testid="glideDataEditor"] { background-color: #161b22 !important; }

/* Radio in sidebar */
[data-testid="stSidebar"] label { color: #c9d1d9 !important; font-size: 13px !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 2px; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background-color: #161b22 !important;
    color: #8b949e !important;
    border-color: #30363d !important;
}
.stTabs [aria-selected="true"] {
    background-color: #21262d !important;
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #161b22 !important;
    color: #e6edf3 !important;
    border-color: #30363d !important;
}

/* Metric cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
}
.metric-label {
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 28px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    color: #e6edf3;
}
.metric-delta-bad  { color: #f85149; font-size: 12px; }
.metric-delta-good { color: #3fb950; font-size: 12px; }
.metric-delta-neu  { color: #8b949e; font-size: 12px; }

/* Alert badges */
.badge-alert  { background:#3d1f1f; color:#f85149; border:1px solid #f85149;
                border-radius:4px; padding:2px 10px; font-size:12px;
                font-family:'IBM Plex Mono',monospace; font-weight:600; }
.badge-ok     { background:#1a2f1a; color:#3fb950; border:1px solid #3fb950;
                border-radius:4px; padding:2px 10px; font-size:12px;
                font-family:'IBM Plex Mono',monospace; font-weight:600; }
.badge-warn   { background:#2e2208; color:#d29922; border:1px solid #d29922;
                border-radius:4px; padding:2px 10px; font-size:12px;
                font-family:'IBM Plex Mono',monospace; font-weight:600; }

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #30363d;
    padding-bottom: 6px;
    margin: 1.5rem 0 1rem;
}

h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; color: #e6edf3; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_model():
    model         = joblib.load(os.path.join(REPO_ROOT, 'models/baseline_model.pkl'))
    features      = joblib.load(os.path.join(REPO_ROOT, 'models/feature_list.pkl'))
    amount_scaler = joblib.load(os.path.join(REPO_ROOT, 'models/amount_scaler.pkl'))
    return model, features, amount_scaler

@st.cache_data
def load_data():
    train = pd.read_csv(os.path.join(REPO_ROOT, 'data/creditcard.csv'))
    batches = {
        'batch_1': pd.read_csv(os.path.join(REPO_ROOT, 'data/drift_1.csv')),
        'batch_2': pd.read_csv(os.path.join(REPO_ROOT, 'data/drift_2.csv')),
        'batch_3': pd.read_csv(os.path.join(REPO_ROOT, 'data/drift_3.csv')),
        'batch_4': pd.read_csv(os.path.join(REPO_ROOT, 'data/drift_4.csv')),
        'batch_5': pd.read_csv(os.path.join(REPO_ROOT, 'data/drift_5.csv')),
    }
    return train, batches

def preprocess(df, amount_scaler):
    df = df.copy()
    df['Amount_scaled'] = amount_scaler.transform(df[['Amount']])
    df['Time_scaled']   = StandardScaler().fit_transform(df[['Time']])
    return df

def calculate_psi(reference, production, bins=10):
    bp = np.linspace(reference.min(), reference.max(), bins + 1)
    bp[0] = -np.inf; bp[-1] = np.inf
    r = np.histogram(reference,  bins=bp)[0]
    p = np.histogram(production, bins=bp)[0]
    rp = np.where(r == 0, 0.0001, r / len(reference))
    pp = np.where(p == 0, 0.0001, p / len(production))
    return round(float(np.sum((pp - rp) * np.log(pp / rp))), 4)

def get_shap_values(explainer, X):
    sv = explainer.shap_values(X)
    if isinstance(sv, list):   return sv[1]
    elif sv.ndim == 3:         return sv[:, :, 1]
    return sv

def batch_status(auc, fraud_rate_train=0.1727):
    if auc < 0.5:   return "CRITICAL", "badge-alert"
    if auc < 0.75:  return "WARNING",  "badge-warn"
    return "OK", "badge-ok"

PLOTLY_DARK = dict(
    paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
    font=dict(color='#8b949e', family='IBM Plex Mono'),
    xaxis=dict(gridcolor='#21262d', zerolinecolor='#30363d'),
    yaxis=dict(gridcolor='#21262d', zerolinecolor='#30363d'),
)
BATCH_COLORS = ['#58a6ff','#79c0ff','#388bfd','#1f6feb','#0d419d']

# ── Load everything ───────────────────────────────────────────────────────────
with st.spinner('Loading model and data...'):
    model, FEATURES, amount_scaler = load_model()
    train, batches = load_data()

v_features = [f'V{i}' for i in range(1, 29)]

# Compute metrics for all batches
@st.cache_data
def compute_all_metrics(_model, _amount_scaler, _features):
    rows = []
    for name, df in batches.items():
        proc = preprocess(df, _amount_scaler)
        X, y = proc[_features], proc['Class']
        proba = _model.predict_proba(X)[:, 1]
        pred  = _model.predict(X)
        rows.append({
            'batch':        name,
            'auc_roc':      round(roc_auc_score(y, proba), 4),
            'f1':           round(f1_score(y, pred), 4),
            'precision':    round(precision_score(y, pred, zero_division=0), 4),
            'recall':       round(recall_score(y, pred), 4),
            'fraud_rate':   round(y.mean() * 100, 4),
            'fraud_count':  int(y.sum()),
        })
    return pd.DataFrame(rows)

metrics_df = compute_all_metrics(model, amount_scaler, FEATURES)
BASELINE = {'auc_roc': 0.9800, 'f1': 0.5608, 'precision': 0.4192, 'recall': 0.8469}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Drift Monitor")
    st.markdown("<div style='font-size:11px;color:#8b949e;margin-bottom:1rem;'>Credit Card Fraud Detection</div>",
                unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "📊 Overview",
        "📉 Performance",
        "🌊 Drift Detection",
        "🔬 Feature Analysis",
        "🧠 SHAP Explainability",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div class='section-header'>Batch Status</div>", unsafe_allow_html=True)
    for _, row in metrics_df.iterrows():
        status, badge = batch_status(row['auc_roc'])
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:4px 0;font-size:12px;font-family:IBM Plex Mono,monospace;'>"
            f"<span style='color:#e6edf3'>{row['batch']}</span>"
            f"<span class='{badge}'>{status}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:10px;color:#484f58;font-family:IBM Plex Mono,monospace;'>"
        "Baseline AUC-ROC: 0.9800<br>Baseline F1: 0.5608</div>",
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("## Drift Monitor — Overview")
    st.markdown(
        "<div style='color:#8b949e;margin-bottom:1.5rem;'>Baseline model trained on creditcard.csv · "
        "Evaluated across 5 production batches</div>", unsafe_allow_html=True
    )

    # Top KPI row
    worst_auc   = metrics_df['auc_roc'].min()
    worst_batch = metrics_df.loc[metrics_df['auc_roc'].idxmin(), 'batch']
    critical    = (metrics_df['auc_roc'] < 0.5).sum()
    max_fraud   = metrics_df['fraud_rate'].max()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Worst AUC-ROC</div>
            <div class='metric-value' style='color:#f85149'>{worst_auc:.4f}</div>
            <div class='metric-delta-bad'>↓ {worst_batch}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Critical Batches</div>
            <div class='metric-value' style='color:#f85149'>{critical}/5</div>
            <div class='metric-delta-bad'>AUC &lt; 0.5 (worse than random)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Max Fraud Rate</div>
            <div class='metric-value' style='color:#d29922'>{max_fraud:.2f}%</div>
            <div class='metric-delta-bad'>↑ vs baseline 0.17%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Baseline AUC-ROC</div>
            <div class='metric-value' style='color:#3fb950'>{BASELINE['auc_roc']:.4f}</div>
            <div class='metric-delta-good'>Test set performance</div>
        </div>""", unsafe_allow_html=True)

    # Metrics table
    st.markdown("<div class='section-header'>Batch Metrics Summary</div>", unsafe_allow_html=True)

    def color_auc(val):
        if val < 0.5:  return 'color: #f85149; font-weight: 600'
        if val < 0.75: return 'color: #d29922; font-weight: 600'
        return 'color: #3fb950; font-weight: 600'

    display_df = metrics_df.copy()
    display_df.columns = ['Batch','AUC-ROC','F1','Precision','Recall','Fraud Rate (%)','Fraud Count']
    # Build styled HTML table
    def row_html(row):
        auc = row['AUC-ROC']
        auc_color = '#f85149' if auc < 0.5 else '#d29922' if auc < 0.75 else '#3fb950'
        cells = f"<td style='padding:8px 12px;border-bottom:1px solid #21262d;font-family:IBM Plex Mono,monospace;font-size:12px;color:#e6edf3'>{row['Batch']}</td>"
        cells += f"<td style='padding:8px 12px;border-bottom:1px solid #21262d;font-family:IBM Plex Mono,monospace;font-size:12px;color:{auc_color};font-weight:600'>{auc:.4f}</td>"
        for col in ['F1','Precision','Recall']:
            cells += f"<td style='padding:8px 12px;border-bottom:1px solid #21262d;font-family:IBM Plex Mono,monospace;font-size:12px;color:#e6edf3'>{row[col]:.4f}</td>"
        cells += f"<td style='padding:8px 12px;border-bottom:1px solid #21262d;font-family:IBM Plex Mono,monospace;font-size:12px;color:#e6edf3'>{row['Fraud Rate (%)']:.4f}%</td>"
        cells += f"<td style='padding:8px 12px;border-bottom:1px solid #21262d;font-family:IBM Plex Mono,monospace;font-size:12px;color:#e6edf3'>{int(row['Fraud Count'])}</td>"
        return f"<tr>{cells}</tr>"

    headers = ''.join([f"<th style='padding:8px 12px;border-bottom:1px solid #30363d;font-family:IBM Plex Mono,monospace;font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:#8b949e;text-align:left;background:#161b22'>{h}</th>"
                       for h in ['Batch','AUC-ROC','F1','Precision','Recall','Fraud Rate (%)','Fraud Count']])
    rows_html = ''.join([row_html(row) for _, row in display_df.iterrows()])
    table_html = f"<div style='border:1px solid #30363d;border-radius:8px;overflow:hidden;background:#161b22'><table style='width:100%;border-collapse:collapse;background:#161b22'><thead><tr>{headers}</tr></thead><tbody>{rows_html}</tbody></table></div>"
    st.markdown(table_html, unsafe_allow_html=True)

    # AUC overview chart
    st.markdown("<div class='section-header'>AUC-ROC Across Batches</div>", unsafe_allow_html=True)
    fig = go.Figure()
    colors = ['#f85149' if v < 0.5 else '#d29922' if v < 0.75 else '#3fb950'
              for v in metrics_df['auc_roc']]
    fig.add_trace(go.Bar(
        x=metrics_df['batch'], y=metrics_df['auc_roc'],
        marker_color=colors, name='AUC-ROC',
        text=metrics_df['auc_roc'].round(4), textposition='outside',
        textfont=dict(family='IBM Plex Mono', size=11)
    ))
    fig.add_hline(y=0.5,  line_dash='dot', line_color='#f85149',
                  annotation_text='Random (0.5)', annotation_position='right')
    fig.add_hline(y=BASELINE['auc_roc'], line_dash='dash', line_color='#3fb950',
                  annotation_text=f'Baseline ({BASELINE["auc_roc"]})', annotation_position='right')
    fig.update_layout(**PLOTLY_DARK, height=350,
                      showlegend=False, margin=dict(t=20, b=20))
    fig.update_yaxes(range=[0, 1.1])
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Performance":
    st.markdown("## Model Performance Degradation")

    # 4 metric line charts
    fig = make_subplots(rows=2, cols=2, subplot_titles=['AUC-ROC','F1 Score','Precision','Recall'])
    metrics_list = ['auc_roc','f1','precision','recall']
    positions    = [(1,1),(1,2),(2,1),(2,2)]
    colors_line  = ['#58a6ff','#f85149','#3fb950','#d29922']
    baseline_vals = [BASELINE['auc_roc'], BASELINE['f1'], BASELINE['precision'], BASELINE['recall']]

    for metric, (r,c), col, bval in zip(metrics_list, positions, colors_line, baseline_vals):
        vals = metrics_df[metric].tolist()
        x    = metrics_df['batch'].tolist()
        fig.add_trace(go.Scatter(x=x, y=vals, mode='lines+markers+text',
                                 line=dict(color=col, width=2),
                                 marker=dict(size=8),
                                 text=[f'{v:.3f}' for v in vals],
                                 textposition='top center',
                                 textfont=dict(size=9, family='IBM Plex Mono'),
                                 name=metric), row=r, col=c)
        fig.add_hline(y=bval, line_dash='dash', line_color='#484f58',
                      annotation_text=f'Baseline {bval}',
                      annotation_font_size=9, row=r, col=c)

    fig.update_layout(**PLOTLY_DARK, height=550, showlegend=False,
                      margin=dict(t=40, b=20))
    fig.update_annotations(font=dict(family='IBM Plex Mono', size=10, color='#8b949e'))
    st.plotly_chart(fig, use_container_width=True)

    # Delta table
    st.markdown("<div class='section-header'>Performance Delta vs Baseline</div>",
                unsafe_allow_html=True)
    delta_rows = []
    for _, row in metrics_df.iterrows():
        delta_rows.append({
            'Batch': row['batch'],
            'Fraud Rate (%)': row['fraud_rate'],
            'ΔAUC-ROC':   round(row['auc_roc']   - BASELINE['auc_roc'],   4),
            'ΔF1':        round(row['f1']        - BASELINE['f1'],        4),
            'ΔPrecision': round(row['precision'] - BASELINE['precision'], 4),
            'ΔRecall':    round(row['recall']    - BASELINE['recall'],    4),
        })
    delta_df = pd.DataFrame(delta_rows)

    def color_delta(val):
        if isinstance(val, float):
            if val < -0.1: return 'color: #f85149'
            if val < 0:    return 'color: #d29922'
            if val > 0:    return 'color: #3fb950'
        return ''

    def delta_row_html(row):
        def cell_color(v):
            if v < -0.1: return '#f85149'
            if v < 0:    return '#d29922'
            if v > 0:    return '#3fb950'
            return '#8b949e'
        cells  = f"<td style='padding:8px 12px;border-bottom:1px solid #21262d;font-family:IBM Plex Mono,monospace;font-size:12px;color:#e6edf3'>{row['Batch']}</td>"
        cells += f"<td style='padding:8px 12px;border-bottom:1px solid #21262d;font-family:IBM Plex Mono,monospace;font-size:12px;color:#e6edf3'>{row['Fraud Rate (%)']:.4f}%</td>"
        for col in ['ΔAUC-ROC','ΔF1','ΔPrecision','ΔRecall']:
            v = row[col]
            sign = '+' if v > 0 else ''
            cells += f"<td style='padding:8px 12px;border-bottom:1px solid #21262d;font-family:IBM Plex Mono,monospace;font-size:12px;color:{cell_color(v)};font-weight:600'>{sign}{v:.4f}</td>"
        return f"<tr>{cells}</tr>"

    d_headers = ''.join([f"<th style='padding:8px 12px;border-bottom:1px solid #30363d;font-family:IBM Plex Mono,monospace;font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:#8b949e;text-align:left;background:#161b22'>{h}</th>"
                         for h in ['Batch','Fraud Rate (%)','ΔAUC-ROC','ΔF1','ΔPrecision','ΔRecall']])
    d_rows = ''.join([delta_row_html(row) for _, row in delta_df.iterrows()])
    d_html = f"<div style='border:1px solid #30363d;border-radius:8px;overflow:hidden;background:#161b22'><table style='width:100%;border-collapse:collapse;background:#161b22'><thead><tr>{d_headers}</tr></thead><tbody>{d_rows}</tbody></table></div>"
    st.markdown(d_html, unsafe_allow_html=True)

    # Prediction score distributions
    st.markdown("<div class='section-header'>Prediction Score Distributions</div>",
                unsafe_allow_html=True)
    selected_batch = st.selectbox("Select batch", list(batches.keys()))
    df_proc = preprocess(batches[selected_batch], amount_scaler)
    proba   = model.predict_proba(df_proc[FEATURES])[:, 1]
    y_true  = df_proc['Class']

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=proba[y_true==0], nbinsx=50, name='Legitimate',
                                marker_color='#58a6ff', opacity=0.6, histnorm='density'))
    fig2.add_trace(go.Histogram(x=proba[y_true==1], nbinsx=50, name='Fraud',
                                marker_color='#f85149', opacity=0.6, histnorm='density'))
    fig2.add_vline(x=0.5, line_dash='dash', line_color='#e6edf3',
                   annotation_text='Threshold 0.5')
    fig2.update_layout(**PLOTLY_DARK, height=350, barmode='overlay',
                       title=f'Prediction scores — {selected_batch}',
                       xaxis_title='Predicted fraud probability',
                       yaxis_title='Density', margin=dict(t=40, b=20))
    st.plotly_chart(fig2, use_container_width=True)

    auc_val = metrics_df[metrics_df.batch == selected_batch]['auc_roc'].values[0]
    status, badge = batch_status(auc_val)
    if status == "CRITICAL":
        st.markdown(
            "⚠️ **Decision boundary collapse detected.** Fraud and legitimate score distributions "
            "are overlapping — the model cannot distinguish fraud in this batch. "
            "Retraining on recent data is recommended.",
            unsafe_allow_html=False
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DRIFT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌊 Drift Detection":
    st.markdown("## Drift Detection")

    # Compute KS stats on demand
    @st.cache_data
    def compute_ks():
        features = v_features + ['Amount']
        rows = {}
        for name, df in batches.items():
            rows[name] = {}
            for f in features:
                stat, p = ks_2samp(train[f], df[f])
                rows[name][f] = round(stat, 4)
        return pd.DataFrame(rows).T

    @st.cache_data
    def compute_psi():
        features = v_features + ['Amount']
        rows = {}
        for name, df in batches.items():
            rows[name] = {}
            for f in features:
                rows[name][f] = calculate_psi(train[f].values, df[f].values)
        return pd.DataFrame(rows).T

    ks_df  = compute_ks()
    psi_df = compute_psi()

    tab1, tab2 = st.tabs(["KS Statistic", "PSI"])

    with tab1:
        st.markdown("**KS test** — p < 0.05 flags a statistically significant distribution shift. "
                    "Higher statistic = larger shift.")
        fig = px.imshow(ks_df, color_continuous_scale='YlOrRd', aspect='auto',
                        zmin=0, zmax=0.35, text_auto='.2f')
        fig.update_layout(**PLOTLY_DARK, height=300,
                          coloraxis_colorbar=dict(tickfont=dict(family='IBM Plex Mono', size=9)),
                          margin=dict(t=10, b=10))
        fig.update_traces(textfont=dict(size=8, family='IBM Plex Mono'))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-header'>Most Drifted Features (KS)</div>",
                    unsafe_allow_html=True)
        top_ks = ks_df.max().sort_values(ascending=False).head(10).reset_index()
        top_ks.columns = ['Feature','Max KS Stat']
        fig2 = px.bar(top_ks, x='Max KS Stat', y='Feature', orientation='h',
                      color='Max KS Stat', color_continuous_scale='YlOrRd')
        fig2.update_layout(**PLOTLY_DARK, height=300, showlegend=False,
                           margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("**PSI** — < 0.1: stable · 0.1–0.2: monitor · > 0.2: action required")
        fig = px.imshow(psi_df, color_continuous_scale='YlOrRd', aspect='auto',
                        zmin=0, zmax=0.5, text_auto='.2f')
        fig.update_layout(**PLOTLY_DARK, height=300,
                          coloraxis_colorbar=dict(tickfont=dict(family='IBM Plex Mono', size=9)),
                          margin=dict(t=10, b=10))
        fig.update_traces(textfont=dict(size=8, family='IBM Plex Mono'))
        st.plotly_chart(fig, use_container_width=True)

    # Label drift
    st.markdown("<div class='section-header'>Label Drift — Fraud Rate Over Batches</div>",
                unsafe_allow_html=True)
    fraud_rates = [b['Class'].mean()*100 for b in batches.values()]
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=list(batches.keys()), y=fraud_rates,
        marker_color=['#f85149' if r > 0.3 else '#58a6ff' for r in fraud_rates],
        text=[f'{r:.2f}%' for r in fraud_rates], textposition='outside',
        textfont=dict(family='IBM Plex Mono', size=11)
    ))
    fig3.add_hline(y=0.1727, line_dash='dash', line_color='#3fb950',
                   annotation_text='Train baseline (0.17%)')
    fig3.update_layout(**PLOTLY_DARK, height=300, yaxis_title='Fraud Rate (%)',
                       margin=dict(t=20, b=20), showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Feature Analysis":
    st.markdown("## Feature Analysis")

    col1, col2 = st.columns(2)
    with col1:
        selected_feature = st.selectbox("Feature", v_features + ['Amount'])
    with col2:
        selected_batch   = st.selectbox("Compare against", list(batches.keys()))

    df_batch = batches[selected_batch]

    fig = go.Figure()
    train_vals = train[selected_feature].clip(
        lower=train[selected_feature].quantile(0.01),
        upper=train[selected_feature].quantile(0.99)
    )
    batch_vals = df_batch[selected_feature].clip(
        lower=train[selected_feature].quantile(0.01),
        upper=train[selected_feature].quantile(0.99)
    )
    fig.add_trace(go.Histogram(x=train_vals, nbinsx=60, name='Training',
                               marker_color='#58a6ff', opacity=0.6, histnorm='density'))
    fig.add_trace(go.Histogram(x=batch_vals, nbinsx=60, name=selected_batch,
                               marker_color='#f85149', opacity=0.6, histnorm='density'))
    ks_stat, ks_p = ks_2samp(train[selected_feature], df_batch[selected_feature])
    psi_val = calculate_psi(train[selected_feature].values, df_batch[selected_feature].values)
    fig.update_layout(**PLOTLY_DARK, barmode='overlay', height=350,
                      title=f'{selected_feature} — KS stat: {ks_stat:.4f} | PSI: {psi_val:.4f} | p-value: {ks_p:.4f}',
                      xaxis_title=selected_feature, yaxis_title='Density',
                      margin=dict(t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Drift status for this feature
    if ks_p < 0.05:
        st.markdown(f"<span class='badge-alert'>DRIFT DETECTED</span> &nbsp; "
                    f"KS test p-value = {ks_p:.4f} (< 0.05)", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='badge-ok'>STABLE</span> &nbsp; "
                    f"KS test p-value = {ks_p:.4f}", unsafe_allow_html=True)

    # All features KS for selected batch
    st.markdown("<div class='section-header'>All Features — KS Statistics</div>",
                unsafe_allow_html=True)
    feat_ks = {}
    for f in v_features + ['Amount']:
        stat, _ = ks_2samp(train[f], df_batch[f])
        feat_ks[f] = round(stat, 4)
    ks_series = pd.Series(feat_ks).sort_values(ascending=True)
    fig2 = px.bar(x=ks_series.values, y=ks_series.index, orientation='h',
                  color=ks_series.values, color_continuous_scale='YlOrRd')
    fig2.update_layout(**PLOTLY_DARK, height=500, showlegend=False,
                       xaxis_title='KS Statistic', margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SHAP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Explainability":
    st.markdown("## SHAP Explainability")
    st.markdown(
        "<div style='color:#8b949e;margin-bottom:1rem;'>SHAP values show which features "
        "drive predictions. Comparing across batches reveals concept drift.</div>",
        unsafe_allow_html=True
    )

    selected_batch = st.selectbox("Select batch to analyse", list(batches.keys()))

    with st.spinner('Computing SHAP values (may take ~30s)...'):
        @st.cache_resource
        def get_explainer(_model):
            return shap.TreeExplainer(_model)

        explainer = get_explainer(model)

        train_proc   = preprocess(train, amount_scaler)
        train_sample = train_proc[FEATURES].sample(500, random_state=42)
        shap_train   = get_shap_values(explainer, train_sample)

        batch_proc   = preprocess(batches[selected_batch], amount_scaler)
        batch_sample = batch_proc[FEATURES].sample(min(500, len(batch_proc)), random_state=42)
        shap_batch   = get_shap_values(explainer, batch_sample)

    train_imp = pd.Series(np.abs(shap_train).mean(axis=0), index=FEATURES)
    batch_imp = pd.Series(np.abs(shap_batch).mean(axis=0), index=FEATURES)
    shift     = (batch_imp - train_imp).sort_values()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Training — Mean |SHAP|**")
        top_train = train_imp.sort_values(ascending=True).tail(15)
        fig = px.bar(x=top_train.values, y=top_train.index, orientation='h',
                     color=top_train.values, color_continuous_scale='Blues')
        fig.update_layout(**PLOTLY_DARK, height=400, showlegend=False,
                          xaxis_title='Mean |SHAP|', margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"**{selected_batch} — Mean |SHAP|**")
        top_batch = batch_imp.sort_values(ascending=True).tail(15)
        auc = metrics_df[metrics_df.batch == selected_batch]['auc_roc'].values[0]
        color_scale = 'Reds' if auc < 0.5 else 'Blues'
        fig = px.bar(x=top_batch.values, y=top_batch.index, orientation='h',
                     color=top_batch.values, color_continuous_scale=color_scale)
        fig.update_layout(**PLOTLY_DARK, height=400, showlegend=False,
                          xaxis_title='Mean |SHAP|', margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>Importance Shift vs Training</div>",
                unsafe_allow_html=True)
    shift_colors = ['#f85149' if v > 0 else '#58a6ff' for v in shift.values]
    fig3 = go.Figure(go.Bar(
        x=shift.values, y=shift.index, orientation='h',
        marker_color=shift_colors,
        text=[f'{v:+.4f}' for v in shift.values], textposition='outside',
        textfont=dict(size=8, family='IBM Plex Mono')
    ))
    fig3.add_vline(x=0, line_color='#8b949e', line_width=1)
    fig3.update_layout(**PLOTLY_DARK, height=550,
                       title='Red = model relies MORE on feature in this batch vs training',
                       xaxis_title='Change in mean |SHAP value|',
                       margin=dict(t=40, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    auc = metrics_df[metrics_df.batch == selected_batch]['auc_roc'].values[0]
    if auc < 0.5:
        st.info(
            f"**{selected_batch} AUC = {auc:.3f}** — The model is performing worse than random. "
            "SHAP importance ranking is similar to training, which means the failure is caused by "
            "**decision boundary collapse** rather than feature importance shift. "
            "The fraud in this batch is indistinguishable from legitimate transactions "
            "in the model's feature space."
        )
