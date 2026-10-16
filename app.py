# -*- coding: utf-8 -*-
# 跌倒风险预测 APP（SSA-XGBoost，校准后模型）—— 中文紧凑版
import os
import pandas as pd
import joblib
import streamlit as st
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV

st.set_page_config(
    page_title="跌倒风险预测",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'SSA_XGBoost最优模型.pkl')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, '结果', '模型', 'SSA_XGBoost最优模型.pkl')
MODEL_AUC = 0.891
FEATURES = ['FallHistory', 'BarthelIndex', 'Osteoarthritis', 'Anemia',
            'ElectrolyteAbnormal', 'Cerebrovascular', 'Hypertension']


@st.cache_resource
def load_bundle():
    b = joblib.load(MODEL_PATH)
    m = b['model']
    xgb = m.calibrated_classifiers_[0].estimator if isinstance(m, CalibratedClassifierCV) else m
    explainer = shap.TreeExplainer(xgb, model_output='raw', feature_perturbation='tree_path_dependent')
    dummy = pd.DataFrame([[0, 60, 0, 0, 0, 0, 0]], columns=FEATURES)
    m.predict_proba(dummy.values)
    return b, explainer


bundle, shap_explainer = load_bundle()
MODEL = bundle['model']
THR = float(bundle.get('threshold', 0.220))

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                     'axes.unicode_minus': False})

st.markdown(
    """
    <style>
    [data-testid="stToolbar"], [data-testid="stHeader"], [data-testid="stDecoration"],
    #MainMenu, footer, [data-testid="stFooter"], [data-testid="stBottom"],
    [data-testid="stAppDeployButton"], [data-testid="stStatusWidget"],
    [data-testid="stMainMenu"] {
        display: none !important;
    }
    .block-container {padding: .85rem 1.4rem .8rem 1.4rem; max-width: 1400px;}
    [data-testid="stAppViewContainer"] {background: #f2f7f5;}
    [data-testid="stVerticalBlock"] {gap: .35rem !important;}

    .h-title {
        text-align: center; font-size: 1.52rem; font-weight: 700; color: #123f3a;
        margin: .2rem 0 1.7rem 0; letter-spacing: .01em;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.section-title),
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:has(.section-title) {
        background: #ffffff; border: 1px solid #e0ece9; border-radius: 12px;
        box-shadow: 0 2px 10px rgba(15,118,110,.07);
        padding: .95rem 1.1rem 1.05rem 1.1rem;
    }
    .model-info {
        font-size: .85rem; color: #4f6a65; line-height: 1.6; margin-top: 1.1rem;
        padding-top: .85rem; border-top: 1px dashed #cfe0dc;
    }
    .mi-title {font-weight: 700; color: #123f3a; font-size: .92rem; margin-bottom: .4rem;}
    .mi-row {display: flex; margin-bottom: .22rem;}
    .mi-k {flex: 0 0 2.9rem; color: #8aa39e;}
    .mi-note {margin-top: .5rem; color: #8aa39e; font-size: .8rem; line-height: 1.6;}
    .section-title {
        font-size: .95rem; letter-spacing: .06em; color: #0f766e; font-weight: 700;
        border-bottom: 2px solid #0f766e; padding-bottom: .22rem;
        margin: .75rem 0 .6rem 0;
    }
    .risk-card {border-radius: 10px; padding: .6rem 1rem; margin-top: .1rem; margin-bottom: .45rem; border: 1px solid transparent;}
    .risk-low {border-left: 6px solid #16a34a; background: #f2fbf5; border-color: #d5eddd;}
    .risk-mid {border-left: 6px solid #d97706; background: #fdf8ee; border-color: #f0e2c4;}
    .risk-high {border-left: 6px solid #dc2626; background: #fdf4f4; border-color: #f2d7d5;}
    .risk-label {font-size: 1.02rem; color: #2e4441;}
    .outcome-line {font-size: 1.02rem; color: #4a625e; margin-top: .3rem;}
    .outcome-pos {color: #b91c1c; font-weight: 700;}
    .outcome-neg {color: #15803d; font-weight: 700;}

    .note {font-size: .85rem; color: #6b827e; margin-top: .2rem;}

    .stNumberInput label, .stRadio label {font-size: .95rem !important; font-weight: 500; color: #2e4441;}
    .stNumberInput div[data-baseweb="input"] {max-height: 2.3rem; border-radius: 6px; border-color: #c8dcd7;}
    .stNumberInput div[data-baseweb="input"] input {font-size: 1rem; font-weight: 500;}
    .stRadio > div {gap: .05rem;}
    .stRadio div[role="radio"] {padding: .05rem .3rem;}
    .stNumberInput, .stRadio {margin-bottom: .1rem;}
    .stButton {margin-top: .4rem; margin-bottom: .35rem;}
    .stButton button {
        background: linear-gradient(90deg, #0f766e, #14b8a6); color: #ffffff;
        font-size: 1.05rem; height: 2.5rem; font-weight: 600;
        border-radius: 8px; border: none; box-shadow: 0 2px 8px rgba(15,118,110,.3);
    }
    [data-testid="stImage"] {margin-top: .55rem;}
    .stButton button:hover {box-shadow: 0 4px 14px rgba(15,118,110,.42); color: #ffffff;}
    .stInfo {
        background: #eef7f5; border: 1px solid #cfe6e1; border-left: 4px solid #0f766e;
        border-radius: 8px; padding: .7rem .9rem; font-size: .95rem; color: #33514c;
    }
    .stInfo p {margin: 0;}
    [data-testid="stTooltipIcon"] svg {fill: #9db8b3;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='h-title'>基于 SSA-XGBoost 的老年住院患者跌倒风险预测模型</div>",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 1], gap="medium")

with left_col:
    st.markdown(
        "<div class='section-title'>输入变量</div>",
        unsafe_allow_html=True,
    )

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fall_history = st.radio(
            "FallHistory", options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=0, horizontal=True,
        )
    with r1c2:
        barthel = st.number_input("BarthelIndex", value=60, step=5)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        osteoarthritis = st.radio(
            "Osteoarthritis", options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=0, horizontal=True,
        )
    with r2c2:
        anemia = st.radio(
            "Anemia", options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=0, horizontal=True,
        )

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        electrolyte = st.radio(
            "ElectrolyteAbnormal", options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=0, horizontal=True,
        )
    with r3c2:
        cerebrovascular = st.radio(
            "Cerebrovascular", options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=0, horizontal=True,
        )

    r4c1, r4c2 = st.columns(2)
    with r4c1:
        hypertension = st.radio(
            "Hypertension", options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=0, horizontal=True,
        )

    st.markdown(
        "<div class='model-info'>"
        "<div class='mi-title'>模型说明</div>"
        "<div class='mi-row'><span class='mi-k'>模型</span><span>SSA-XGBoost（7 项预测因子，sigmoid 概率校准）</span></div>"
        "<div class='mi-row'><span class='mi-k'>队列</span><span>760 例老年住院患者（训练 532 / 测试 228）</span></div>"
        "<div class='mi-row'><span class='mi-k'>效能</span><span>测试集 AUC = 0.891</span></div>"
        "<div class='mi-row'><span class='mi-k'>分层</span><span>低 &lt;0.11 ｜ 中 0.11–0.22 ｜ 高 ≥0.22</span></div>"
        "<div class='mi-note'>分层依据：以 Youden 最优阈值（0.22）为高风险线、其半值（0.11）为低/中分界；三层实际跌倒率 5.1% / 12.6% / 54.7%。仅供科研参考，不用于临床诊断。</div>"
        "</div>",
        unsafe_allow_html=True,
    )


@st.fragment
def result_panel():
    predict_clicked = st.button("预 测", type="primary", use_container_width=True)

    if predict_clicked:
        row = pd.DataFrame([[float(fall_history), float(barthel), float(osteoarthritis),
                             float(anemia), float(electrolyte), float(cerebrovascular),
                             float(hypertension)]], columns=FEATURES)
        proba = MODEL.predict_proba(row.values)[0][1]

        if proba < THR / 2:
            group, cls, outcome, o_cls = "低风险", "risk-low", "未发生跌倒", "outcome-neg"
        elif proba < THR:
            group, cls, outcome, o_cls = "中风险", "risk-mid", "未发生跌倒", "outcome-neg"
        else:
            group, cls, outcome, o_cls = "高风险", "risk-high", "发生跌倒", "outcome-pos"

        st.markdown(
            f"""
            <div class='risk-card {cls}'>
                <div class='risk-label'>跌倒风险分层：<b>{group}</b></div>
                <div class='outcome-line'>预测结局：<b class='{o_cls}'>{outcome}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-title'>SHAP 单样本解释</div>", unsafe_allow_html=True)

        try:
            sv = shap_explainer(row)
            fig = plt.figure()
            shap.plots.waterfall(sv[0], max_display=7, show=False)
            fig = plt.gcf()
            fig.set_size_inches(6.8, 3.3)
            st.pyplot(fig, use_container_width=True)
            plt.close('all')
            shap_ok = True
        except Exception as e:
            shap_ok = False
            st.error(f"SHAP 解释生成失败: {e}")

        if shap_ok:
            st.markdown(
                "<div class='note'>红色推高风险、蓝色降低风险（相对队列平均水平）；"
                "数值为对数几率单位。</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("在左侧填写变量后点击「预 测」查看跌倒风险。")


with right_col:
    st.markdown(
        "<div class='section-title'>预测结果</div>",
        unsafe_allow_html=True,
    )
    result_panel()
