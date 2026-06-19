"""
스마트제조 공정분석 대시보드
8주차: 공정능력분석  |  9주차: 통계적공정관리(SPC)
강의록 코드·공식 기반 구현
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from scipy.stats import shapiro, boxcox
from scipy.special import gammaln

# ─────────────────────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="공정능력분석 & SPC 대시보드",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
div[data-testid="metric-container"] {background:#1e293b;border:1px solid #334155;
  border-radius:10px;padding:10px;}
.stTabs [data-baseweb="tab-list"] {gap:8px;}
.stTabs [data-baseweb="tab"] {background:#1e293b;border:1px solid #334155;
  border-radius:8px;padding:8px 20px;}
.stTabs [aria-selected="true"] {background:#3b82f6!important;border-color:#3b82f6!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SPC 불편화 상수표 (n = 2 ~ 25)
# ─────────────────────────────────────────────────────────────
SPC_TABLE = {
    2:  dict(A2=1.880,A3=2.659,B3=0,    B4=3.267,D3=0,    D4=3.267,d2=1.128,c4=0.7979),
    3:  dict(A2=1.023,A3=1.954,B3=0,    B4=2.568,D3=0,    D4=2.574,d2=1.693,c4=0.8862),
    4:  dict(A2=0.729,A3=1.628,B3=0,    B4=2.266,D3=0,    D4=2.282,d2=2.059,c4=0.9213),
    5:  dict(A2=0.577,A3=1.427,B3=0,    B4=2.089,D3=0,    D4=2.114,d2=2.326,c4=0.9400),
    6:  dict(A2=0.483,A3=1.287,B3=0.030,B4=1.970,D3=0,    D4=2.004,d2=2.534,c4=0.9515),
    7:  dict(A2=0.419,A3=1.182,B3=0.118,B4=1.882,D3=0.076,D4=1.924,d2=2.704,c4=0.9594),
    8:  dict(A2=0.373,A3=1.099,B3=0.185,B4=1.815,D3=0.136,D4=1.864,d2=2.847,c4=0.9650),
    9:  dict(A2=0.337,A3=1.032,B3=0.239,B4=1.761,D3=0.184,D4=1.816,d2=2.970,c4=0.9693),
    10: dict(A2=0.308,A3=0.975,B3=0.284,B4=1.716,D3=0.223,D4=1.777,d2=3.078,c4=0.9727),
    11: dict(A2=0.285,A3=0.927,B3=0.321,B4=1.679,D3=0.256,D4=1.744,d2=3.173,c4=0.9754),
    12: dict(A2=0.266,A3=0.886,B3=0.354,B4=1.646,D3=0.283,D4=1.717,d2=3.258,c4=0.9776),
    13: dict(A2=0.249,A3=0.850,B3=0.382,B4=1.618,D3=0.307,D4=1.693,d2=3.336,c4=0.9794),
    14: dict(A2=0.235,A3=0.817,B3=0.406,B4=1.594,D3=0.328,D4=1.672,d2=3.407,c4=0.9810),
    15: dict(A2=0.223,A3=0.789,B3=0.428,B4=1.572,D3=0.347,D4=1.653,d2=3.472,c4=0.9823),
    16: dict(A2=0.212,A3=0.763,B3=0.448,B4=1.552,D3=0.363,D4=1.637,d2=3.532,c4=0.9835),
    17: dict(A2=0.203,A3=0.739,B3=0.466,B4=1.534,D3=0.378,D4=1.622,d2=3.588,c4=0.9845),
    18: dict(A2=0.194,A3=0.718,B3=0.482,B4=1.518,D3=0.391,D4=1.609,d2=3.640,c4=0.9854),
    19: dict(A2=0.187,A3=0.698,B3=0.497,B4=1.503,D3=0.403,D4=1.597,d2=3.689,c4=0.9862),
    20: dict(A2=0.180,A3=0.680,B3=0.510,B4=1.490,D3=0.415,D4=1.585,d2=3.735,c4=0.9869),
    21: dict(A2=0.173,A3=0.663,B3=0.523,B4=1.477,D3=0.425,D4=1.575,d2=3.778,c4=0.9876),
    22: dict(A2=0.167,A3=0.647,B3=0.534,B4=1.466,D3=0.434,D4=1.566,d2=3.819,c4=0.9882),
    23: dict(A2=0.162,A3=0.633,B3=0.545,B4=1.455,D3=0.443,D4=1.557,d2=3.858,c4=0.9887),
    24: dict(A2=0.157,A3=0.619,B3=0.555,B4=1.445,D3=0.451,D4=1.548,d2=3.895,c4=0.9892),
    25: dict(A2=0.153,A3=0.606,B3=0.565,B4=1.435,D3=0.459,D4=1.541,d2=3.931,c4=0.9896),
}

def get_k(coef: str, n: int) -> float:
    n_c = max(2, min(25, int(n)))
    return SPC_TABLE[n_c][coef]

def calc_c4(m: int) -> float:
    """공정능력분석용 c4 (자유도 m). 강의록 8주차 공식."""
    if m <= 1: return 1.0
    return np.sqrt(2/(m-1)) * np.exp(gammaln(m/2) - gammaln((m-1)/2))

# ─────────────────────────────────────────────────────────────
# 샘플 데이터
# ─────────────────────────────────────────────────────────────
SAMPLE_MEAS = """x1,x2,x3,x4,x5
106,94,103,97,100
98,100,96,103,93
104,102,95,101,108
96,100,95,103,101
98,101,107,104,95
102,105,100,95,98
105,103,109,102,96
94,101,93,99,98
113,110,119,116,122
101,106,103,99,96
98,101,99,92,105
100,105,106,101,108
99,102,111,108,105
102,106,99,109,104
108,112,106,99,105
101,106,102,107,109
110,101,107,98,104
108,111,101,106,104
104,111,105,98,107
100,106,101,105,108
77,80,83,86,89
102,100,98,105,95
101,98,99,105,92
97,102,98,103,105
103,106,97,94,100"""

SAMPLE_DEF = """로트,검사수,불량수
1,200,5
2,200,7
3,200,4
4,200,6
5,200,8
6,200,3
7,200,9
8,200,5
9,200,6
10,200,4
11,200,7
12,200,16
13,200,5
14,200,8
15,200,6
16,200,4
17,200,7
18,200,5
19,200,18
20,200,6
21,200,5
22,200,7
23,200,4
24,200,8
25,200,5"""

# ─────────────────────────────────────────────────────────────
# 통계 함수 (강의록 8주차)
# ─────────────────────────────────────────────────────────────
def run_normality(values: np.ndarray) -> dict:
    """Shapiro-Wilk 정규성 검정."""
    if len(values) < 3:
        return dict(stat=None, p=None, normal=True)
    w, p = shapiro(values)
    return dict(stat=float(w), p=float(p), normal=bool(p >= 0.05))


def calc_capability(values: np.ndarray, sg_labels, LSL: float, USL: float) -> dict:
    """Cp, Cpk, Pp, Ppk 계산 (강의록 8주차 공식 그대로)."""
    xbar = np.mean(values)
    n_total = len(values)
    sg_unique = np.unique(sg_labels)
    k = len(sg_unique)

    # σ_overall
    sigma_overall = np.std(values, ddof=1) / calc_c4(n_total)

    # σ_within (합동 표준편차)
    sigma_sg = np.array([np.std(values[sg_labels == sg], ddof=1)
                         for sg in sg_unique if len(values[sg_labels == sg]) > 1])
    sigma_p = np.sqrt(np.sum(sigma_sg**2) / len(sigma_sg))
    d = n_total - k + 1
    sigma_within = sigma_p / calc_c4(d)

    Cp  = (USL-LSL) / (6*sigma_within)
    Cpk = min((USL-xbar)/(3*sigma_within), (xbar-LSL)/(3*sigma_within))
    Pp  = (USL-LSL) / (6*sigma_overall)
    Ppk = min((USL-xbar)/(3*sigma_overall), (xbar-LSL)/(3*sigma_overall))
    return dict(xbar=xbar, sigma_w=sigma_within, sigma_o=sigma_overall,
                Cp=Cp, Cpk=Cpk, Pp=Pp, Ppk=Ppk, n=n_total, k=k, d=d)


def grade_cp(cp: float):
    if cp >= 1.67: return "0등급", "매우 충분 ✅", "#22c55e"
    if cp >= 1.33: return "1등급", "충분 ✅",     "#3b82f6"
    if cp >= 1.00: return "2등급", "괜찮음 ⚠️",   "#f59e0b"
    if cp >= 0.67: return "3등급", "모자람 🔴",   "#f97316"
    return           "4등급", "매우 부족 ❌",      "#ef4444"


# ─────────────────────────────────────────────────────────────
# SPC 관리도 계산 (강의록 9주차)
# ─────────────────────────────────────────────────────────────
def calc_xbar_r(df: pd.DataFrame, sg: str, val: str):
    grp = df.groupby(sg)[val]
    xb = grp.mean(); R = grp.apply(lambda x: x.max()-x.min())
    n = int(grp.apply(len).mode().iloc[0])
    xb_bar, R_bar = xb.mean(), R.mean()
    xb_ch = pd.DataFrame({'point':xb,'CL':xb_bar,'UCL':xb_bar+get_k('A2',n)*R_bar,'LCL':xb_bar-get_k('A2',n)*R_bar})
    R_ch  = pd.DataFrame({'point':R, 'CL':R_bar, 'UCL':get_k('D4',n)*R_bar,'LCL':max(0,get_k('D3',n)*R_bar)})
    info  = f"X̄={xb_bar:.4f}  R̄={R_bar:.4f}  n={n}  A₂={get_k('A2',n)}"
    return (xb_ch, R_ch), [f"X̄ 관리도 ({val})", f"R 관리도 ({val})"], info

def calc_xbar_s(df: pd.DataFrame, sg: str, val: str):
    grp = df.groupby(sg)[val]
    xb = grp.mean(); s = grp.std(ddof=1)
    n = int(grp.apply(len).mode().iloc[0])
    xb_bar, s_bar = xb.mean(), s.mean()
    xb_ch = pd.DataFrame({'point':xb,'CL':xb_bar,'UCL':xb_bar+get_k('A3',n)*s_bar,'LCL':xb_bar-get_k('A3',n)*s_bar})
    s_ch  = pd.DataFrame({'point':s, 'CL':s_bar, 'UCL':get_k('B4',n)*s_bar,'LCL':max(0,get_k('B3',n)*s_bar)})
    info  = f"X̄={xb_bar:.4f}  s̄={s_bar:.4f}  n={n}  A₃={get_k('A3',n)}"
    return (xb_ch, s_ch), [f"X̄ 관리도 ({val})", f"S 관리도 ({val})"], info

def calc_imr(df: pd.DataFrame, sg: str, val: str):
    v = df.set_index(sg)[val]
    mr = v.diff().abs()
    xb, mr_bar = v.mean(), mr.mean()
    d2 = get_k('d2',2)
    i_ch  = pd.DataFrame({'point':v,  'CL':xb,    'UCL':xb+3*mr_bar/d2,'LCL':xb-3*mr_bar/d2})
    mr_ch = pd.DataFrame({'point':mr, 'CL':mr_bar, 'UCL':get_k('D4',2)*mr_bar,'LCL':max(0,get_k('D3',2)*mr_bar)})
    info  = f"X̄={xb:.4f}  MR̄={mr_bar:.4f}  d₂={d2}"
    return (i_ch, mr_ch), [f"I 관리도 ({val})", "MR 관리도"], info

def calc_np(df: pd.DataFrame, sg: str, n_col: str, d_col: str):
    n = int(df[n_col].iloc[0])
    np_v = df.groupby(sg)[d_col].sum()
    np_bar = np_v.mean(); p_bar = np_bar/n
    sd = np.sqrt(np_bar*(1-p_bar))
    ch = pd.DataFrame({'point':np_v,'CL':np_bar,'UCL':np_bar+3*sd,'LCL':max(0,np_bar-3*sd)})
    info = f"np̄={np_bar:.4f}  p̄={p_bar:.4f}  n={n}"
    return (ch,), [f"NP 관리도"], info

def calc_p(df: pd.DataFrame, sg: str, n_col: str, d_col: str):
    g = df.groupby(sg).agg({n_col:'first', d_col:'sum'})
    p_i = g[d_col]/g[n_col]; p_bar = g[d_col].sum()/g[n_col].sum()
    sd = 3*np.sqrt(p_bar*(1-p_bar)/g[n_col])
    ch = pd.DataFrame({'point':p_i,'CL':p_bar,'UCL':p_bar+sd,'LCL':(p_bar-sd).clip(lower=0)})
    info = f"p̄={p_bar:.4f}"
    return (ch,), ["P 관리도"], info

def calc_c(df: pd.DataFrame, sg: str, d_col: str):
    c_v = df.groupby(sg)[d_col].sum()
    c_bar = c_v.mean()
    sd = 3*np.sqrt(c_bar)
    ch = pd.DataFrame({'point':c_v,'CL':c_bar,'UCL':c_bar+sd,'LCL':max(0,c_bar-sd)})
    info = f"c̄={c_bar:.4f}"
    return (ch,), ["C 관리도"], info

def calc_u(df: pd.DataFrame, sg: str, n_col: str, d_col: str):
    g = df.groupby(sg).agg({n_col:'first', d_col:'sum'})
    u_i = g[d_col]/g[n_col]; u_bar = g[d_col].sum()/g[n_col].sum()
    ch = pd.DataFrame({'point':u_i,'CL':u_bar,
                        'UCL':u_bar+3*np.sqrt(u_bar/g[n_col]),
                        'LCL':(u_bar-3*np.sqrt(u_bar/g[n_col])).clip(lower=0)})
    info = f"ū={u_bar:.4f}"
    return (ch,), ["U 관리도"], info

def get_ooc(charts) -> list:
    ch = charts[0]
    mask = (ch['point'] > ch['UCL']) | (ch['point'] < ch['LCL'])
    return ch.index[mask].tolist()

# ─────────────────────────────────────────────────────────────
# 시각화 함수
# ─────────────────────────────────────────────────────────────
_DL = dict(plot_bgcolor='#1e293b', paper_bgcolor='#0f172a',
           font=dict(color='#f1f5f9', size=12))

def _axes(fig):
    fig.update_xaxes(gridcolor='#374151', zerolinecolor='#374151')
    fig.update_yaxes(gridcolor='#374151', zerolinecolor='#374151')
    return fig

def fig_qq(values, title='Q-Q Plot'):
    n = len(values); sv = np.sort(values)
    th = stats.norm.ppf((np.arange(1,n+1)-0.375)/(n+0.25))
    sl, ic, *_ = stats.linregress(th, sv)
    fig = go.Figure([
        go.Scatter(x=th, y=sv, mode='markers', name='표본',
                   marker=dict(color='#3b82f6', size=8)),
        go.Scatter(x=[th.min(),th.max()], y=[sl*th.min()+ic, sl*th.max()+ic],
                   mode='lines', name='기준선', line=dict(color='#ef4444', width=2)),
    ])
    fig.update_layout(title=title, xaxis_title='이론적 분위수', yaxis_title='표본 분위수',
                      height=400, **_DL)
    return _axes(fig)

def fig_capability_hist(values, LSL, USL, sigma_w, sigma_o, xbar):
    margin = max(4*sigma_w, (USL-LSL)*0.3)
    xr = np.linspace(xbar-margin, xbar+margin, 400)
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Histogram(x=values.tolist(), nbinsx=15, name='측정값',
                               marker_color='rgba(59,130,246,0.45)'), secondary_y=False)
    fig.add_trace(go.Scatter(x=xr.tolist(), y=stats.norm.pdf(xr, xbar, sigma_w).tolist(),
                             mode='lines', name=f'Within σ={sigma_w:.3f}',
                             line=dict(color='#22c55e', width=2)), secondary_y=True)
    fig.add_trace(go.Scatter(x=xr.tolist(), y=stats.norm.pdf(xr, xbar, sigma_o).tolist(),
                             mode='lines', name=f'Overall σ={sigma_o:.3f}',
                             line=dict(color='#f59e0b', width=2, dash='dash')), secondary_y=True)
    fig.add_vline(x=LSL, line_dash='dash', line_color='#ef4444',
                  annotation_text='LSL', annotation_position='top left')
    fig.add_vline(x=USL, line_dash='dash', line_color='#ef4444',
                  annotation_text='USL', annotation_position='top right')
    fig.add_vline(x=xbar, line_dash='dot', line_color='#a78bfa',
                  annotation_text=f'X̄={xbar:.2f}', annotation_position='top')
    fig.update_layout(title='공정능력 분포', barmode='overlay', height=420,
                      legend=dict(orientation='h', y=1.08, x=0), **_DL)
    fig.update_yaxes(title_text='빈도', secondary_y=False, gridcolor='#374151')
    fig.update_yaxes(title_text='밀도', secondary_y=True, gridcolor='#374151')
    fig.update_xaxes(gridcolor='#374151')
    return fig

def fig_control_chart(charts, titles, phase_label, ooc_list=None):
    nrows = len(charts)
    fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True,
                        subplot_titles=titles, vertical_spacing=0.14)
    for i, ch in enumerate(charts):
        row = i+1
        x = ch.index.tolist()
        ooc_mask = (ch['point'] > ch['UCL']) | (ch['point'] < ch['LCL'])
        colors = ['#ef4444' if v else '#3b82f6' for v in ooc_mask]

        fig.add_trace(go.Scatter(x=x, y=ch['point'].tolist(), mode='lines+markers',
                                  name='측정값' if i==0 else '',
                                  line=dict(color='#3b82f6', width=2),
                                  marker=dict(color=colors, size=9),
                                  showlegend=(i==0)), row=row, col=1)
        if ooc_mask.any():
            fig.add_trace(go.Scatter(x=ch.index[ooc_mask].tolist(),
                                      y=ch['point'][ooc_mask].tolist(),
                                      mode='markers', name='이상점(OOC)' if i==0 else '',
                                      marker=dict(color='#ef4444', size=14, symbol='circle-open',
                                                  line=dict(width=3, color='#ef4444')),
                                      showlegend=(i==0)), row=row, col=1)

        # UCL / LCL 선
        ucl_v = ch['UCL']
        if ucl_v.nunique() == 1:
            u, l, c = float(ucl_v.iloc[0]), float(ch['LCL'].iloc[0]), float(ch['CL'].iloc[0])
            fig.add_hline(y=u, line_dash='dash', line_color='#ef4444',
                          annotation_text=f'UCL={u:.3f}', annotation_position='right',
                          row=row, col=1)
            fig.add_hline(y=l, line_dash='dash', line_color='#ef4444',
                          annotation_text=f'LCL={l:.3f}', annotation_position='right',
                          row=row, col=1)
            fig.add_hline(y=c, line_dash='solid', line_color='#22c55e',
                          annotation_text=f'CL={c:.3f}', annotation_position='right',
                          row=row, col=1)
        else:
            # 변동 UCL/LCL (P, U)
            fig.add_trace(go.Scatter(x=x, y=ch['UCL'].tolist(), mode='lines',
                                      name='UCL', line=dict(color='#ef4444', width=1.5, dash='dash'),
                                      showlegend=(i==0)), row=row, col=1)
            fig.add_trace(go.Scatter(x=x, y=ch['LCL'].tolist(), mode='lines',
                                      name='LCL', line=dict(color='#ef4444', width=1.5, dash='dash'),
                                      showlegend=False), row=row, col=1)
            c = float(ch['CL'].iloc[0])
            fig.add_hline(y=c, line_dash='solid', line_color='#22c55e',
                          annotation_text=f'CL={c:.3f}', annotation_position='right',
                          row=row, col=1)

    fig.update_layout(title=phase_label, height=350*nrows, hovermode='x unified',
                      legend=dict(orientation='h', y=1.02, x=1, xanchor='right'), **_DL)
    _axes(fig)
    return fig

def fig_limits_compare(ch1, ch2, label):
    """Phase 1 vs Phase 2 관리한계 비교 바 차트."""
    ucl1 = float(ch1['UCL'].iloc[0]) if ch1['UCL'].nunique()==1 else ch1['UCL'].mean()
    lcl1 = float(ch1['LCL'].iloc[0]) if ch1['LCL'].nunique()==1 else ch1['LCL'].mean()
    cl1  = float(ch1['CL'].iloc[0])
    ucl2 = float(ch2['UCL'].iloc[0]) if ch2['UCL'].nunique()==1 else ch2['UCL'].mean()
    lcl2 = float(ch2['LCL'].iloc[0]) if ch2['LCL'].nunique()==1 else ch2['LCL'].mean()
    cl2  = float(ch2['CL'].iloc[0])
    cats = ['LCL','CL','UCL']
    fig = go.Figure([
        go.Bar(name='1단계 (초기)', x=cats, y=[lcl1,cl1,ucl1], marker_color='#f59e0b'),
        go.Bar(name='2단계 (수정)', x=cats, y=[lcl2,cl2,ucl2], marker_color='#3b82f6'),
    ])
    fig.update_layout(title=f'{label} — 관리한계 비교', barmode='group',
                      height=350, yaxis_title='값', **_DL)
    _axes(fig)
    return fig


# ─────────────────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ 설정 패널")
st.sidebar.caption("파라미터를 변경하면 모든 탭이 자동으로 갱신됩니다.")

data_type = st.sidebar.radio("📊 데이터 유형",
                              ["계량형 (측정값)", "계수형 (불량/결함)"])
st.sidebar.markdown("---")

# ─── 계량형 설정 ─────────────────────────────────────────────
if data_type == "계량형 (측정값)":
    st.sidebar.subheader("📥 데이터 입력")
    upfile = st.sidebar.file_uploader("CSV 파일 업로드", type="csv",
        help="Wide(각 열=측정값) 또는 Long([부분군,측정값]) 형식 지원")
    use_sample = st.sidebar.button("📋 샘플 데이터 로드 (measurements.csv)")

    if use_sample or ('df_meas' not in st.session_state and upfile is None):
        df_raw = pd.read_csv(io.StringIO(SAMPLE_MEAS))
        st.session_state['df_meas'] = df_raw
        st.session_state['meas_wide'] = True

    if upfile is not None:
        df_raw = pd.read_csv(upfile)
        st.session_state['df_meas'] = df_raw
        st.session_state['meas_wide'] = (df_raw.shape[1] > 2)

    df_raw = st.session_state.get('df_meas', pd.read_csv(io.StringIO(SAMPLE_MEAS)))
    is_wide = st.session_state.get('meas_wide', True)

    if is_wide:
        st.sidebar.info(f"Wide 형식 감지: {df_raw.shape[0]}개 부분군 × {df_raw.shape[1]}개 측정값")
        # Wide → Long 변환
        df_raw = df_raw.reset_index().rename(columns={'index':'Lot'})
        df_raw['Lot'] = df_raw['Lot'] + 1
        val_cols = [c for c in df_raw.columns if c != 'Lot']
        df_long = df_raw.melt(id_vars='Lot', value_vars=val_cols,
                               var_name='Obs', value_name='Value')
        sg_col, val_col = 'Lot', 'Value'
    else:
        cols = df_raw.columns.tolist()
        sg_col  = st.sidebar.selectbox("부분군 컬럼", cols, index=0)
        val_col = st.sidebar.selectbox("측정값 컬럼", cols, index=min(1, len(cols)-1))
        df_long = df_raw[[sg_col, val_col]].dropna()

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 공정능력분석 설정")
    all_vals = df_long[val_col].dropna().values
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    span = vmax - vmin
    LSL = st.sidebar.number_input("LSL (하한 규격)", value=round(vmin - span*0.1, 1), step=0.5)
    USL = st.sidebar.number_input("USL (상한 규격)", value=round(vmax + span*0.1, 1), step=0.5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 SPC 설정")
    n_sg = int(df_long.groupby(sg_col)[val_col].count().mode().iloc[0])
    chart_options = ["Xbar-R","Xbar-S"] if n_sg >= 2 else ["I-MR"]
    if n_sg == 1:
        chart_options = ["I-MR"]
    else:
        chart_options = ["Xbar-R","Xbar-S","I-MR"]
    spc_type = st.sidebar.selectbox("관리도 종류", chart_options)

    df_count = None
    sg_count, n_col_c, def_col = None, None, None

# ─── 계수형 설정 ─────────────────────────────────────────────
else:
    st.sidebar.subheader("📥 데이터 입력")
    upfile2 = st.sidebar.file_uploader("CSV 파일 업로드", type="csv",
        help="컬럼 구성: [부분군명, 표본크기, 결함/불량수]")
    use_sample2 = st.sidebar.button("📋 샘플 데이터 로드 (defects.csv)")

    if use_sample2 or 'df_count' not in st.session_state:
        st.session_state['df_count'] = pd.read_csv(io.StringIO(SAMPLE_DEF))
    if upfile2 is not None:
        st.session_state['df_count'] = pd.read_csv(upfile2)

    df_count = st.session_state.get('df_count', pd.read_csv(io.StringIO(SAMPLE_DEF)))
    cols = df_count.columns.tolist()
    sg_count = st.sidebar.selectbox("부분군 컬럼", cols, index=0)
    n_col_c  = st.sidebar.selectbox("표본크기 컬럼", cols, index=min(1, len(cols)-1))
    def_col  = st.sidebar.selectbox("결함/불량 컬럼", cols, index=min(2, len(cols)-1))

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 SPC 설정")
    n_unique = df_count[n_col_c].nunique()
    spc_type = st.sidebar.selectbox("관리도 종류",
                                     ["NP","P"] if n_unique <= 1 else ["P","NP","C","U"])

    df_long, sg_col, val_col = None, None, None
    LSL, USL = 0.0, 1.0

# ─────────────────────────────────────────────────────────────
# 헤더
# ─────────────────────────────────────────────────────────────
st.title("⚙️ 스마트제조 공정분석 대시보드")
st.markdown(
    "강의록 **8주차 공정능력분석** + **9주차 통계적공정관리(SPC)** 기반 의사결정 지원 도구. "
    "좌측 사이드바에서 데이터·파라미터를 변경하면 모든 탭이 즉시 갱신됩니다."
)

tab_data, tab_cap, tab_spc = st.tabs(
    ["📊 데이터 탐색", "🎯 공정능력분석 (8주차)", "📈 통계적공정관리 SPC (9주차)"]
)

# ─────────────────────────────────────────────────────────────
# TAB 1: 데이터 탐색
# ─────────────────────────────────────────────────────────────
with tab_data:
    if data_type == "계량형 (측정값)":
        st.subheader(f"계량형 데이터 — {df_long.shape[0]}행 × {df_long.shape[1]}열")
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.write(df_long.head(50).to_html(index=False), unsafe_allow_html=True)
        with col_b:
            desc = df_long[val_col].describe()
            st.metric("데이터 수",     f"{int(desc['count'])}")
            st.metric("평균",           f"{desc['mean']:.3f}")
            st.metric("표준편차 (s)",   f"{desc['std']:.3f}")
            st.metric("최솟값 / 최댓값", f"{desc['min']:.1f} / {desc['max']:.1f}")
            st.metric("부분군 수",       f"{df_long[sg_col].nunique()}")

        # 부분군별 박스플롯
        import plotly.express as px
        fig_box = px.box(df_long, x=sg_col, y=val_col, points='all',
                         title='부분군별 분포 (Box Plot)',
                         color_discrete_sequence=['#3b82f6'])
        if LSL is not None and USL is not None:
            fig_box.add_hline(y=LSL, line_dash='dash', line_color='#ef4444',
                              annotation_text='LSL')
            fig_box.add_hline(y=USL, line_dash='dash', line_color='#ef4444',
                              annotation_text='USL')
        fig_box.update_layout(height=420, **_DL)
        _axes(fig_box)
        st.plotly_chart(fig_box, use_container_width=True, key='box_main')

        # 시계열
        grp_mean = df_long.groupby(sg_col)[val_col].mean().reset_index()
        fig_ts = go.Figure([
            go.Scatter(x=grp_mean[sg_col], y=grp_mean[val_col],
                       mode='lines+markers', name='부분군 평균',
                       line=dict(color='#3b82f6', width=2))
        ])
        fig_ts.update_layout(title='부분군 평균 추이', height=300,
                              xaxis_title='부분군', yaxis_title=val_col, **_DL)
        _axes(fig_ts)
        st.plotly_chart(fig_ts, use_container_width=True, key='ts_main')

    else:
        st.subheader(f"계수형 데이터 — {df_count.shape[0]}행")
        col_a, col_b = st.columns([3,1])
        with col_a:
            st.write(df_count.to_html(index=False), unsafe_allow_html=True)
        with col_b:
            total_n   = df_count[n_col_c].sum()
            total_def = df_count[def_col].sum()
            st.metric("전체 검사 수", f"{total_n:,}")
            st.metric("전체 불량/결함 수", f"{total_def:,}")
            st.metric("전체 불량률 p̄", f"{total_def/total_n:.4f}")
            st.metric("부분군 수", f"{df_count[sg_count].nunique()}")

        fig_bar = go.Figure([
            go.Bar(x=df_count[sg_count].astype(str).tolist(),
                   y=df_count[def_col].tolist(),
                   name='불량/결함 수', marker_color='#3b82f6')
        ])
        fig_bar.update_layout(title='부분군별 불량/결함 수', height=380,
                               xaxis_title='부분군', yaxis_title='수', **_DL)
        _axes(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True, key='bar_count')

        p_series = df_count[def_col] / df_count[n_col_c]
        fig_p = go.Figure([
            go.Scatter(x=df_count[sg_count].astype(str).tolist(),
                       y=p_series.tolist(),
                       mode='lines+markers', name='불량률',
                       line=dict(color='#f59e0b', width=2))
        ])
        fig_p.update_layout(title='부분군별 불량률 추이', height=300, **_DL)
        _axes(fig_p)
        st.plotly_chart(fig_p, use_container_width=True, key='p_trend')

        # 다운로드
        buf = io.StringIO()
        df_count.to_csv(buf, index=False)
        st.download_button("📄 데이터 CSV 다운로드", buf.getvalue(),
                           file_name="count_data.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────────
# TAB 2: 공정능력분석 (계량형 전용)
# ─────────────────────────────────────────────────────────────
with tab_cap:
    if data_type != "계량형 (측정값)":
        st.info("공정능력분석은 계량형 데이터에서만 수행됩니다. 좌측에서 '계량형'을 선택하세요.")
    else:
        values = df_long[val_col].dropna().values
        sg_arr = df_long[sg_col].values

        # ── 정규성 검정 ──────────────────────────────────────
        st.subheader("1️⃣ 정규성 검정 (Shapiro-Wilk)")
        nr = run_normality(values)
        col1, col2, col3 = st.columns(3)
        col1.metric("W 통계량", f"{nr['stat']:.4f}" if nr['stat'] else "N/A")
        col2.metric("p-value",  f"{nr['p']:.4f}"    if nr['p']    else "N/A")
        col3.metric("판정",     "정규 분포 ✅" if nr['normal'] else "비정규 ❌")

        if nr['normal']:
            st.success(f"p-value = {nr['p']:.4f} ≥ 0.05 → 정규성 만족. 공정능력지수 계산 진행합니다.")
        else:
            st.warning(f"p-value = {nr['p']:.4f} < 0.05 → 정규성 불만족. Box-Cox 변환을 권장합니다.")

        col_qq, col_hist0 = st.columns(2)
        with col_qq:
            st.plotly_chart(fig_qq(values, 'Q-Q Plot (정규성 시각 확인)'),
                            use_container_width=True, key='qq_main')

        # Box-Cox 옵션
        use_bc = False
        if not nr['normal']:
            with st.expander("⚗️ Box-Cox 변환 (비정규 데이터용)", expanded=True):
                st.markdown("정규성이 만족되지 않으면 Box-Cox 변환 후 Cp를 계산합니다.")
                if (values > 0).all():
                    bc_vals, lam = boxcox(values)
                    bc_nr = run_normality(bc_vals)
                    col_bc1, col_bc2 = st.columns(2)
                    col_bc1.metric("최적 λ", f"{lam:.4f}")
                    col_bc2.metric("변환 후 p-value", f"{bc_nr['p']:.4f}" if bc_nr['p'] else "N/A")
                    if bc_nr['normal']:
                        st.success("Box-Cox 변환 후 정규성 만족!")
                    else:
                        st.error("변환 후에도 정규성 미충족. 결과 해석에 주의하세요.")
                    use_bc = st.checkbox("Box-Cox 변환 값으로 공정능력 계산", value=bc_nr['normal'])
                    if use_bc:
                        st.plotly_chart(fig_qq(bc_vals, 'Q-Q Plot (Box-Cox 변환 후)'),
                                        use_container_width=True, key='qq_bc')
                        # 변환 후 규격 재설정
                        st.info("Box-Cox 변환 후 규격(USL/LSL)은 변환 공간의 ±3σ 범위로 자동 설정됩니다.")
                        bc_USL = np.mean(bc_vals) + 3*np.std(bc_vals)
                        bc_LSL = np.mean(bc_vals) - 3*np.std(bc_vals)
                        bc_cap = calc_capability(bc_vals, sg_arr, bc_LSL, bc_USL)
                        st.markdown(f"**변환 후 Cp = {bc_cap['Cp']:.4f}**  "
                                    f"Cpk = {bc_cap['Cpk']:.4f}")
                else:
                    st.error("Box-Cox 변환은 모든 값이 양수인 경우에만 가능합니다.")

        # ── 공정능력 지수 ────────────────────────────────────
        st.subheader("2️⃣ 공정능력 지수")
        cap = calc_capability(values, sg_arr, LSL, USL)
        indices = {'Cp': cap['Cp'], 'Cpk': cap['Cpk'],
                   'Pp': cap['Pp'], 'Ppk': cap['Ppk']}

        cols = st.columns(4)
        for i, (name, val) in enumerate(indices.items()):
            g, desc, color = grade_cp(val)
            with cols[i]:
                st.metric(name, f"{val:.4f}")
                st.markdown(
                    f"<div style='text-align:center;color:{color};font-weight:600'>"
                    f"{g} — {desc}</div>", unsafe_allow_html=True
                )

        # 상세 계산 정보
        with st.expander("📐 계산 상세 (강의록 공식)"):
            st.markdown(f"""
| 항목 | 값 |
|------|-----|
| 전체 데이터 수 (N) | {cap['n']} |
| 부분군 수 (k) | {cap['k']} |
| 자유도 d = N-k+1 | {cap['d']} |
| 전체 평균 X̄ | {cap['xbar']:.4f} |
| σ_within (c₄ 보정) | {cap['sigma_w']:.4f} |
| σ_overall (c₄ 보정) | {cap['sigma_o']:.4f} |
| LSL | {LSL} |
| USL | {USL} |
""")

        # ── 등급 기준표 ──────────────────────────────────────
        with st.expander("📊 Cp 등급 기준표 (강의록 8주차)"):
            grade_df = pd.DataFrame({
                'Cp 범위': ['≥ 1.67', '≥ 1.33', '≥ 1.00', '≥ 0.67', '< 0.67'],
                '등급': ['0등급', '1등급', '2등급', '3등급', '4등급'],
                '판정': ['매우 충분', '충분', '괜찮음', '모자람', '매우 부족'],
            })
            st.write(grade_df.to_html(index=False), unsafe_allow_html=True)

        # ── 공정능력 히스토그램 ───────────────────────────────
        st.subheader("3️⃣ 공정능력 분포 차트")
        st.plotly_chart(
            fig_capability_hist(values, LSL, USL, cap['sigma_w'], cap['sigma_o'], cap['xbar']),
            use_container_width=True, key='cap_hist'
        )

# ─────────────────────────────────────────────────────────────
# TAB 3: SPC
# ─────────────────────────────────────────────────────────────
with tab_spc:
    st.subheader(f"관리도 유형: **{spc_type}**")

    # ── 계산 ────────────────────────────────────────────────
    if data_type == "계량형 (측정값)":
        if spc_type == "Xbar-R":
            charts1, labels1, info1 = calc_xbar_r(df_long, sg_col, val_col)
        elif spc_type == "Xbar-S":
            charts1, labels1, info1 = calc_xbar_s(df_long, sg_col, val_col)
        else:
            charts1, labels1, info1 = calc_imr(df_long, sg_col, val_col)
    else:
        if spc_type == "NP":
            charts1, labels1, info1 = calc_np(df_count, sg_count, n_col_c, def_col)
        elif spc_type == "P":
            charts1, labels1, info1 = calc_p(df_count, sg_count, n_col_c, def_col)
        elif spc_type == "C":
            charts1, labels1, info1 = calc_c(df_count, sg_count, def_col)
        else:
            charts1, labels1, info1 = calc_u(df_count, sg_count, n_col_c, def_col)

    ooc1 = get_ooc(charts1)

    # ── 1단계 관리도 ─────────────────────────────────────────
    st.markdown("### 1단계 — 초기 관리도")
    st.info(f"📐 관리한계 파라미터: {info1}")
    st.plotly_chart(
        fig_control_chart(charts1, labels1, f"{spc_type} — 1단계 (초기)", ooc1),
        use_container_width=True, key='spc_phase1'
    )

    # ── OOC 결과 ─────────────────────────────────────────────
    if ooc1:
        st.error(f"🔴 이상점(OOC) 발견 — 이탈 부분군: **{ooc1}** ({len(ooc1)}개)")
        st.markdown("#### 이상점 데이터")
        ch1 = charts1[0]
        ooc_df = ch1.loc[ooc1, ['point','CL','UCL','LCL']].copy()
        ooc_df.columns = ['측정값','CL','UCL','LCL']
        ooc_df['이탈 방향'] = ooc_df.apply(
            lambda r: '↑ UCL 초과' if r['측정값'] > r['UCL'] else '↓ LCL 미달', axis=1
        )
        st.write(ooc_df.to_html(index=False), unsafe_allow_html=True)

        # ── 2단계 ───────────────────────────────────────────
        st.markdown("### 2단계 — 이상 부분군 제거 후 재작성")
        if data_type == "계량형 (측정값)":
            df_clean = df_long[~df_long[sg_col].isin(ooc1)].copy()
            n_removed = len(df_long) - len(df_clean)
            st.success(f"이상 부분군 {ooc1} 제거: {len(df_long)}개 → {len(df_clean)}개 ({n_removed}개 제거)")

            if spc_type == "Xbar-R":
                charts2, labels2, info2 = calc_xbar_r(df_clean, sg_col, val_col)
            elif spc_type == "Xbar-S":
                charts2, labels2, info2 = calc_xbar_s(df_clean, sg_col, val_col)
            else:
                charts2, labels2, info2 = calc_imr(df_clean, sg_col, val_col)
        else:
            df_clean = df_count[~df_count[sg_count].isin(ooc1)].copy()
            n_removed = len(df_count) - len(df_clean)
            st.success(f"이상 부분군 {ooc1} 제거: {len(df_count)}개 → {len(df_clean)}개 ({n_removed}개 제거)")

            if spc_type == "NP":
                charts2, labels2, info2 = calc_np(df_clean, sg_count, n_col_c, def_col)
            elif spc_type == "P":
                charts2, labels2, info2 = calc_p(df_clean, sg_count, n_col_c, def_col)
            elif spc_type == "C":
                charts2, labels2, info2 = calc_c(df_clean, sg_count, def_col)
            else:
                charts2, labels2, info2 = calc_u(df_clean, sg_count, n_col_c, def_col)

        ooc2 = get_ooc(charts2)
        st.info(f"📐 수정 후 파라미터: {info2}")
        st.plotly_chart(
            fig_control_chart(charts2, labels2, f"{spc_type} — 2단계 (이상점 제거 후)", ooc2),
            use_container_width=True, key='spc_phase2'
        )

        if ooc2:
            st.warning(f"⚠️ 2단계에도 이상점 존재: {ooc2} — 원인 분석 필요")
        else:
            st.success("✅ 2단계: 이상점 없음. 관리 상태 양호.")

        # ── 관리한계 비교 ────────────────────────────────────
        st.markdown("### 관리한계 비교 (1단계 vs 2단계)")
        col_cmp1, col_cmp2 = st.columns(2)
        with col_cmp1:
            st.plotly_chart(
                fig_limits_compare(charts1[0], charts2[0], labels1[0]),
                use_container_width=True, key='cmp1'
            )
        if len(charts1) > 1:
            with col_cmp2:
                st.plotly_chart(
                    fig_limits_compare(charts1[1], charts2[1], labels1[1]),
                    use_container_width=True, key='cmp2'
                )

        # ── 관리한계 요약표 ──────────────────────────────────
        st.markdown("#### 관리한계 요약")
        def _lim_row(ch, phase):
            u = float(ch['UCL'].iloc[0]) if ch['UCL'].nunique()==1 else float(ch['UCL'].mean())
            c = float(ch['CL'].iloc[0])
            l = float(ch['LCL'].iloc[0]) if ch['LCL'].nunique()==1 else float(ch['LCL'].mean())
            return {'단계': phase, 'LCL': round(l,4), 'CL': round(c,4), 'UCL': round(u,4),
                    '폭(UCL-LCL)': round(u-l,4)}

        lim_rows = [_lim_row(charts1[0], '1단계 (초기)'),
                    _lim_row(charts2[0], '2단계 (수정)')]
        lim_df = pd.DataFrame(lim_rows)
        st.write(lim_df.to_html(index=False), unsafe_allow_html=True)
        narrow = lim_df.iloc[1]['폭(UCL-LCL)'] < lim_df.iloc[0]['폭(UCL-LCL)']
        if narrow:
            st.success("✅ 이상점 제거 후 관리한계 폭이 좁아졌습니다 (공정 산포 감소 확인).")
        else:
            st.info("관리한계 폭 변화 없음.")

    else:
        st.success("✅ 이상점(OOC) 없음 — 공정이 통계적 관리 상태에 있습니다.")

    # ── 하단: 관리도 설명 ────────────────────────────────────
    with st.expander("📚 관리도 종류 및 공식 (강의록 9주차)"):
        st.markdown("""
| 구분 | 차트 | 데이터 | 주요 공식 |
|------|------|--------|-----------|
| 계량형 | X̄-R | 부분군 ≥2 | UCL = X̄ + A₂R̄ / LCL = X̄ - A₂R̄ |
| 계량형 | X̄-S | 부분군 ≥2 | UCL = X̄ + A₃s̄ / LCL = X̄ - A₃s̄ |
| 계량형 | I-MR | 개별 측정 | UCL = X̄ + 3MR̄/d₂ |
| 계수형 | NP | 불량수 (일정n) | UCL = np̄ + 3√(np̄(1-p̄)) |
| 계수형 | P | 불량률 (변동n) | UCL = p̄ + 3√(p̄(1-p̄)/nᵢ) |
| 계수형 | C | 결함수 (일정n) | UCL = c̄ + 3√c̄ |
| 계수형 | U | 단위당결함 (변동n) | UCL = ū + 3√(ū/nᵢ) |
""")

# ─────────────────────────────────────────────────────────────
# 데이터 다운로드 (공통)
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📥 결과 다운로드")
dl_cols = st.columns(3)
if data_type == "계량형 (측정값)":
    with dl_cols[0]:
        buf = io.StringIO(); df_long.to_csv(buf, index=False)
        st.download_button("📄 Long 형식 데이터", buf.getvalue(),
                           "long_data.csv", "text/csv")
    with dl_cols[1]:
        spc_result = charts1[0].copy()
        spc_result['OOC'] = (spc_result['point'] > spc_result['UCL']) | \
                             (spc_result['point'] < spc_result['LCL'])
        buf2 = io.StringIO(); spc_result.to_csv(buf2)
        st.download_button("📊 관리도 결과 CSV", buf2.getvalue(),
                           f"{spc_type}_result.csv", "text/csv")
else:
    with dl_cols[0]:
        buf = io.StringIO(); df_count.to_csv(buf, index=False)
        st.download_button("📄 계수형 데이터", buf.getvalue(),
                           "count_data.csv", "text/csv")

st.caption("© 2026 스마트제조 기말 프로젝트 — 강의록 8주차(공정능력분석) + 9주차(SPC) 기반 구현")
