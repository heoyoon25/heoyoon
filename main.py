import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import LabelEncoder
import re
import warnings
warnings.filterwarnings('ignore')

# ── 한글 폰트 설정 ──────────────────────────────────────────────
matplotlib.rcParams['axes.unicode_minus'] = False
try:
    matplotlib.rcParams['font.family'] = 'NanumGothic'
except:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ── 페이지 기본 설정 ────────────────────────────────────────────
st.set_page_config(
    page_title="이탈 고객 예측 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 공통 CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* 사이드바 */
    [data-testid="stSidebar"] {background-color: #1e2a3a;}
    [data-testid="stSidebar"] * {color: #ffffff !important;}

    /* 메인 타이틀 */
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        font-size: 1.05rem; color: #6c757d; margin-bottom: 1.5rem;
    }

    /* 카드 */
    .card {
        background: #ffffff; border-radius: 12px;
        padding: 1.4rem 1.6rem; margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-left: 4px solid #667eea;
    }
    .card-title {
        font-size: 1.05rem; font-weight: 700;
        color: #343a40; margin-bottom: 0.6rem;
    }

    /* 메트릭 박스 */
    .metric-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px; padding: 1rem 1.2rem;
        text-align: center; color: white;
    }
    .metric-box .val {font-size: 1.9rem; font-weight: 800;}
    .metric-box .lbl {font-size: 0.82rem; opacity: 0.85; margin-top: 2px;}

    /* 배지 */
    .badge {
        display: inline-block; padding: 3px 10px;
        border-radius: 20px; font-size: 0.78rem; font-weight: 600;
        background: #e9ecef; color: #495057; margin: 2px;
    }

    /* 구분선 */
    .section-divider {
        border: none; border-top: 2px solid #e9ecef; margin: 1.5rem 0;
    }

    /* 결과 테이블 */
    .result-table th {background: #667eea; color: white; text-align: center;}
    .result-table td {text-align: center;}

    /* 업로드 영역 */
    [data-testid="stFileUploader"] {
        border: 2px dashed #667eea !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }

    /* 버튼 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {opacity: 0.88;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  세션 상태 초기화
# ══════════════════════════════════════════════════════════════════
defaults = {
    "df_raw": None,          # 원본 데이터
    "df": None,              # 작업 데이터
    "df_processed": None,    # 전처리 완료 데이터
    "X_train": None, "X_test": None,
    "y_train": None, "y_test": None,
    "lr_model": None, "dt_model": None,
    "lr_result": None, "dt_result": None,
    "selected_X": [],
    "selected_y": None,
    "split_ratio": "7:3",
    "missing_handled": False,
    "outlier_handled": False,
    "encoded": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════
#  사이드바 네비게이션
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 이탈 고객 예측")
    st.markdown("---")
    pages = {
        "🏠  메인 / 데이터 업로드": "main",
        "🔍  데이터 탐색":          "eda",
        "⚙️  데이터 전처리":        "preprocess",
        "🤖  연구 모형":            "model",
        "📈  연구 결과":            "result",
    }
    page = st.radio("페이지 선택", list(pages.keys()), label_visibility="collapsed")
    current = pages[page]

    st.markdown("---")
    if st.session_state.df is not None:
        df_info = st.session_state.df
        st.markdown(f"**📁 데이터 현황**")
        st.markdown(f"- 행: `{df_info.shape[0]:,}`")
        st.markdown(f"- 열: `{df_info.shape[1]}`")
        status_items = [
            ("결측치 처리", st.session_state.missing_handled),
            ("이상치 처리", st.session_state.outlier_handled),
            ("인코딩",     st.session_state.encoded),
        ]
        for label, done in status_items:
            icon = "✅" if done else "⬜"
            st.markdown(f"{icon} {label}")
    else:
        st.markdown("*데이터를 먼저 업로드하세요*")

# ══════════════════════════════════════════════════════════════════
#  헬퍼 함수
# ══════════════════════════════════════════════════════════════════
def check_data():
    if st.session_state.df is None:
        st.warning("⚠️ 먼저 **메인 페이지**에서 데이터를 업로드해 주세요.")
        st.stop()

def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = (model.predict_proba(X_test)[:, 1]
              if hasattr(model, "predict_proba") else None)
    fpr, tpr, _ = roc_curve(y_test, y_prob) if y_prob is not None else (None, None, None)
    roc_auc = auc(fpr, tpr) if fpr is not None else None
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "fpr": fpr, "tpr": tpr, "auc": roc_auc,
        "y_pred": y_pred,
        "cm": confusion_matrix(y_test, y_pred),
    }

# ══════════════════════════════════════════════════════════════════
#  PAGE 1 ── 메인 / 데이터 업로드
# ══════════════════════════════════════════════════════════════════
if current == "main":
    st.markdown('<p class="main-title">📊 신용평가모형</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">고객 이탈 예측을 위한 머신러닝 분석 플랫폼</p>',
                unsafe_allow_html=True)

    # 소개 카드
    col1, col2, col3 = st.columns(3)
    cards = [
        ("🔍", "데이터 탐색",   "변수 분포·상관관계를 시각화합니다."),
        ("⚙️", "데이터 전처리", "결측치·이상치·인코딩을 처리합니다."),
        ("🤖", "예측 모형",     "Logistic Regression · Decision Tree"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div style="font-size:2rem;">{icon}</div>
                <div class="card-title">{title}</div>
                <div style="color:#6c757d;font-size:0.88rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # 업로드 섹션
    st.markdown("### 📂 데이터 업로드")
    st.markdown("CSV 또는 Excel 파일을 업로드하세요.")

    uploaded = st.file_uploader(
        "파일 선택 (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
        help="UTF-8 인코딩 CSV 또는 Excel 파일을 지원합니다."
    )

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded, encoding="utf-8-sig")
            else:
                df = pd.read_excel(uploaded)

            # 세션 저장
            st.session_state.df_raw = df.copy()
            st.session_state.df = df.copy()
            st.session_state.df_processed = None
            # 전처리 상태 초기화
            for k in ["missing_handled", "outlier_handled", "encoded"]:
                st.session_state[k] = False
            for k in ["X_train", "X_test", "y_train", "y_test",
                      "lr_model", "dt_model", "lr_result", "dt_result"]:
                st.session_state[k] = None

            st.success(f"✅ **{uploaded.name}** 업로드 완료!")

            # 요약 메트릭
            c1, c2, c3, c4 = st.columns(4)
            metrics_data = [
                (df.shape[0], "총 행 수"),
                (df.shape[1], "총 열 수"),
                (int(df.isnull().sum().sum()), "결측치 수"),
                (df.select_dtypes(include=np.number).shape[1], "수치형 변수"),
            ]
            for col, (val, lbl) in zip([c1, c2, c3, c4], metrics_data):
                with col:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="val">{val:,}</div>
                        <div class="lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # 미리보기
            with st.expander("📋 데이터 미리보기 (상위 10행)", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

            with st.expander("📊 기술 통계량"):
                st.dataframe(df.describe(include="all").T, use_container_width=True)

        except Exception as e:
            st.error(f"❌ 파일 로드 오류: {e}")

    else:
        # 샘플 데이터 생성 버튼
        st.markdown("---")
        st.markdown("#### 💡 샘플 데이터로 시작하기")
        if st.button("🎲 샘플 데이터 생성"):
            np.random.seed(42)
            n = 500
            sample_df = pd.DataFrame({
                "age":           np.random.randint(20, 70, n),
                "tenure":        np.random.randint(1, 60, n),
                "balance":       np.random.uniform(0, 200000, n).round(2),
                "num_products":  np.random.randint(1, 5, n),
                "credit_score":  np.random.randint(300, 850, n),
                "is_active":     np.random.randint(0, 2, n),
                "gender":        np.random.choice(["Male", "Female"], n),
                "geography":     np.random.choice(["France", "Germany", "Spain"], n),
                "salary":        np.random.uniform(20000, 150000, n).round(2),
                "churn":         np.random.choice([0, 1], n, p=[0.8, 0.2]),
            })
            # 결측치 5% 삽입
            for col in ["balance", "credit_score", "salary"]:
                idx = np.random.choice(n, int(n * 0.05), replace=False)
                sample_df.loc[idx, col] = np.nan

            st.session_state.df_raw = sample_df.copy()
            st.session_state.df = sample_df.copy()
            for k in ["missing_handled", "outlier_handled", "encoded"]:
                st.session_state[k] = False
            st.success("✅ 샘플 데이터가 생성되었습니다! (500행 × 10열)")
            st.dataframe(sample_df.head(), use_container_width=True)
            st.rerun()

# ══════════════════════════════════════════════════════════════════
#  PAGE 2 ── 데이터 탐색 (EDA)
# ══════════════════════════════════════════════════════════════════
elif current == "eda":
    check_data()
    df = st.session_state.df

    st.markdown("## 🔍 데이터 탐색")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 기본 정보 ──────────────────────────────────────────────
    st.markdown("### 📐 기본 정보")
    c1, c2, c3, c4 = st.columns(4)
    info_data = [
        (df.shape[0], "행 수"),
        (df.shape[1], "열 수"),
        (int(df.isnull().sum().sum()), "결측치"),
        (df.duplicated().sum(), "중복 행"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], info_data):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="val">{val:,}</div>
                <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 변수 목록 & 타입 ───────────────────────────────────────
    st.markdown("### 📋 변수 목록 및 타입")
    col_left, col_right = st.columns([1, 1])

    num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]

    with col_left:
        dtype_df = pd.DataFrame({
            "변수명":         df.columns.tolist(),
            "데이터 타입":    df.dtypes.astype(str).values,
            "변수 구분":      ["수치형" if c in num_cols else "범주형" for c in df.columns],
            "결측치 수":      df.isnull().sum().values,
            "결측치 비율(%)": (df.isnull().mean() * 100).round(2).values,
            "고유값 수":      df.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, height=320)

    with col_right:
        type_counts = pd.Series([
            "수치형" if pd.api.types.is_numeric_dtype(df[col]) else "범주형"
            for col in df.columns
        ]).value_counts()

        fig_pie, ax_pie = plt.subplots(figsize=(4, 3.5))
        colors = ["#667eea", "#f093fb"]
        ax_pie.pie(
            type_counts.values,
            labels=type_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            textprops={"fontsize": 11}
        )
        ax_pie.set_title("변수 타입 분포", fontsize=12, fontweight="bold")
        st.pyplot(fig_pie, use_container_width=True)
        plt.close()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 시각화 ─────────────────────────────────────────────────
    st.markdown("### 📊 변수 시각화")

    all_cols = df.columns.tolist()

    v_col1, v_col2, v_col3 = st.columns([1, 1, 1])
    with v_col1:
        x_var = st.selectbox("X축 변수", all_cols, key="eda_x")
    with v_col2:
        y_var = st.selectbox("Y축 변수", ["(없음)"] + all_cols, key="eda_y")
    with v_col3:
        chart_type = st.selectbox(
            "그래프 유형",
            ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart"],
            key="eda_chart"
        )

    if st.button("📊 그래프 생성", key="btn_chart"):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        palette = "#667eea"

        try:
            x_data = df[x_var].copy()
            y_data = df[y_var].copy() if y_var != "(없음)" else None

            x_is_num = pd.api.types.is_numeric_dtype(x_data)
            y_is_num = pd.api.types.is_numeric_dtype(y_data) if y_data is not None else False

            # ── Histogram ─────────────────────────────────────
            if chart_type == "Histogram":
                if x_is_num:
                    ax.hist(
                        x_data.dropna().astype(float),
                        bins=30, color=palette,
                        edgecolor="white", alpha=0.85
                    )
                    ax.set_xlabel(x_var)
                    ax.set_ylabel("빈도")
                else:
                    counts = x_data.value_counts()
                    ax.bar(
                        range(len(counts)), counts.values,
                        color=palette, edgecolor="white", alpha=0.85
                    )
                    ax.set_xticks(range(len(counts)))
                    ax.set_xticklabels(counts.index, rotation=45, ha="right")
                    ax.set_xlabel(x_var)
                    ax.set_ylabel("빈도")

            # ── Box Plot ──────────────────────────────────────
            elif chart_type == "Box Plot":
                if not x_is_num:
                    st.warning("Box Plot의 X축은 수치형 변수를 선택해 주세요.")
                    plt.close()
                    st.stop()

                if y_data is not None and not y_is_num:
                    groups = []
                    labels = []
                    for g in df[y_var].dropna().unique():
                        grp = df[df[y_var] == g][x_var].dropna().astype(float)
                        if len(grp) > 0:
                            groups.append(grp.values)
                            labels.append(str(g))
                    ax.boxplot(
                        groups, patch_artist=True,
                        boxprops=dict(facecolor=palette, alpha=0.6),
                        medianprops=dict(color="red", linewidth=2)
                    )
                    ax.set_xticks(range(1, len(labels) + 1))
                    ax.set_xticklabels(labels, rotation=45, ha="right")
                    ax.set_xlabel(y_var)
                    ax.set_ylabel(x_var)
                else:
                    ax.boxplot(
                        x_data.dropna().astype(float).values,
                        patch_artist=True,
                        boxprops=dict(facecolor=palette, alpha=0.6),
                        medianprops=dict(color="red", linewidth=2)
                    )
                    ax.set_ylabel(x_var)
                    ax.set_xticks([1])
                    ax.set_xticklabels([x_var])

            # ── Scatter Plot ──────────────────────────────────
            elif chart_type == "Scatter Plot":
                if y_var == "(없음)":
                    st.warning("Scatter Plot은 Y축 변수를 선택해야 합니다.")
                    plt.close()
                    st.stop()
                if not x_is_num or not y_is_num:
                    st.warning("Scatter Plot은 X축, Y축 모두 수치형 변수를 선택해 주세요.")
                    plt.close()
                    st.stop()

                valid = df[[x_var, y_var]].dropna()
                ax.scatter(
                    valid[x_var].astype(float),
                    valid[y_var].astype(float),
                    alpha=0.4, color=palette,
                    edgecolors="white", linewidths=0.3
                )
                ax.set_xlabel(x_var)
                ax.set_ylabel(y_var)

            # ── Bar Chart ─────────────────────────────────────
            elif chart_type == "Bar Chart":
                if not x_is_num:
                    counts = x_data.value_counts()
                    ax.bar(
                        range(len(counts)), counts.values,
                        color=palette, edgecolor="white", alpha=0.85
                    )
                    ax.set_xticks(range(len(counts)))
                    ax.set_xticklabels(counts.index, rotation=45, ha="right")
                    ax.set_xlabel(x_var)
                    ax.set_ylabel("빈도")
                else:
                    if y_data is not None and y_is_num:
                        valid = df[[x_var, y_var]].dropna()
                        valid = valid.copy()
                        valid[x_var] = valid[x_var].astype(float)
                        valid[y_var] = valid[y_var].astype(float)
                        bins = pd.cut(valid[x_var], bins=10)
                        grouped = valid.groupby(bins, observed=True)[y_var].mean()
                        labels = [str(i) for i in grouped.index]
                        ax.bar(
                            range(len(grouped)), grouped.values,
                            color=palette, edgecolor="white", alpha=0.85
                        )
                        ax.set_xticks(range(len(labels)))
                        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                        ax.set_xlabel(x_var)
                        ax.set_ylabel(f"{y_var} (평균)")
                    else:
                        counts = x_data.value_counts().sort_index()
                        ax.bar(
                            range(len(counts)), counts.values,
                            color=palette, edgecolor="white", alpha=0.85
                        )
                        ax.set_xticks(range(len(counts)))
                        ax.set_xticklabels(
                            [str(i) for i in counts.index],
                            rotation=45, ha="right", fontsize=8
                        )
                        ax.set_xlabel(x_var)
                        ax.set_ylabel("빈도")

            # ── Line Chart ────────────────────────────────────
            elif chart_type == "Line Chart":
                if not x_is_num:
                    st.warning("Line Chart의 X축은 수치형 변수를 선택해 주세요.")
                    plt.close()
                    st.stop()

                if y_data is not None and y_is_num:
                    valid = df[[x_var, y_var]].dropna().sort_values(x_var)
                    ax.plot(
                        valid[x_var].astype(float),
                        valid[y_var].astype(float),
                        color=palette, alpha=0.8, linewidth=1.5
                    )
                    ax.set_xlabel(x_var)
                    ax.set_ylabel(y_var)
                else:
                    sorted_data = x_data.dropna().astype(float).reset_index(drop=True)
                    ax.plot(
                        sorted_data.index,
                        sorted_data.values,
                        color=palette, alpha=0.8, linewidth=1.5
                    )
                    ax.set_xlabel("Index")
                    ax.set_ylabel(x_var)

            # ── 공통 스타일 ───────────────────────────────────
            title = f"{chart_type}  |  {x_var}"
            if y_var != "(없음)":
                title += f"  vs  {y_var}"
            ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
            ax.spines[["top", "right"]].set_visible(False)
            ax.yaxis.grid(True, alpha=0.3, linestyle="--")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"그래프 생성 오류: {e}")
        finally:
            plt.close()

    # ── 상관관계 히트맵 ────────────────────────────────────────
    if len(num_cols) >= 2:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 🌡️ 수치형 변수 상관관계 히트맵")
        fig_hm, ax_hm = plt.subplots(figsize=(10, 6))
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlBu_r", center=0, ax=ax_hm,
            linewidths=0.5, cbar_kws={"shrink": 0.8}
        )
        ax_hm.set_title("상관관계 히트맵", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_hm, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════════════
#  PAGE 3 ── 데이터 전처리 / Feature Selection / Partitioning
# ══════════════════════════════════════════════════════════════════
# ── 필요 import ──────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ════════════════════════════════════════════════════════════
# 전처리 페이지 메인
# ════════════════════════════════════════════════════════════
def show_preprocess_page():
    st.markdown("## ⚙️ 데이터 전처리")

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ 먼저 데이터를 업로드해주세요.")
        return

    df = st.session_state.df.copy()

    # ── 현재 데이터 현황 ──────────────────────────────────────
    st.markdown("### 📊 현재 데이터 현황")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("전체 행 수", f"{df.shape[0]:,}행")
    with col2:
        st.metric("전체 열 수", f"{df.shape[1]:,}개")
    with col3:
        st.metric("결측치 수", f"{df.isnull().sum().sum():,}개")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # ⚡ 빠른 전처리 (One-Click)
    # ════════════════════════════════════════════════════════
    st.markdown("### ⚡ 빠른 전처리 (One-Click)")

    with st.expander("🚀 전처리 한번에 실행", expanded=False):
        st.markdown("""
        **아래 순서로 자동 처리됩니다:**
        1. 🔴 이상치 → IQR 방법으로 처리
        2. 🟡 결측치 → 수치형: 중앙값, 범주형: 최빈값으로 대체
        3. 🟢 인코딩 → 고유값 10개 이하: One-Hot, 초과: Label Encoding
        """)

        col1, col2 = st.columns(2)
        with col1:
            target_col_quick = st.selectbox(
                "🎯 Target 컬럼 선택 (인코딩/이상치 처리 제외)",
                options=st.session_state.df.columns.tolist(),
                key="quick_target"
            )
        with col2:
            outlier_method_quick = st.radio(
                "이상치 처리 방법",
                ["IQR 대체(중앙값)", "IQR 제거"],
                horizontal=True,
                key="quick_outlier_method"
            )

        if st.button("⚡ 전처리 한번에 실행", key="btn_quick_preprocess", type="primary"):
            df_work = st.session_state.df.copy()
            log = []

            try:
                with st.spinner("전처리 진행 중..."):

                    # ── STEP 1: 이상치 처리 ──────────────────
                    num_cols = [
                        c for c in df_work.select_dtypes(include='number').columns
                        if c != target_col_quick  # ✅ Target 제외
                    ]
                    outlier_count = 0

                    for c in num_cols:
                        Q1 = df_work[c].quantile(0.25)
                        Q3 = df_work[c].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        mask = (df_work[c] < lower) | (df_work[c] > upper)
                        cnt = mask.sum()

                        if cnt > 0:
                            if outlier_method_quick == "IQR 제거":
                                # ✅ 제거 후 클래스 2개 이상인지 확인
                                df_temp = df_work[~mask]
                                if df_temp[target_col_quick].nunique() >= 2:
                                    df_work = df_temp
                                else:
                                    # 클래스 손실 위험 → 대체로 변경
                                    df_work.loc[mask, c] = df_work[c].median()
                            else:
                                df_work.loc[mask, c] = df_work[c].median()
                            outlier_count += cnt

                    log.append(f"✅ 이상치 처리 완료: {outlier_count}개 처리")

                    # ── STEP 2: 결측치 처리 ──────────────────
                    missing_count = df_work.isnull().sum().sum()
                    for c in df_work.columns:
                        if df_work[c].isnull().any():
                            if pd.api.types.is_numeric_dtype(df_work[c]):
                                df_work[c] = df_work[c].fillna(df_work[c].median())
                            else:
                                mode_val = df_work[c].mode()
                                df_work[c] = df_work[c].fillna(
                                    mode_val[0] if len(mode_val) > 0 else "Unknown"
                                )
                    log.append(f"✅ 결측치 처리 완료: {missing_count}개 대체")

                    # ── STEP 3: 인코딩 ───────────────────────
                    cat_cols = [
                        c for c in df_work.columns
                        if not pd.api.types.is_numeric_dtype(df_work[c])
                        and c != target_col_quick  # ✅ Target 제외
                    ]

                    ohe_cols = [c for c in cat_cols if df_work[c].nunique() <= 10]
                    le_cols  = [c for c in cat_cols if df_work[c].nunique() > 10]

                    if ohe_cols:
                        df_work = pd.get_dummies(df_work, columns=ohe_cols, drop_first=True)
                        bool_cols = [c for c in df_work.columns if df_work[c].dtype == bool]
                        if bool_cols:
                            df_work[bool_cols] = df_work[bool_cols].astype(int)

                    if le_cols:
                        le = LabelEncoder()
                        for c in le_cols:
                            df_work[c] = le.fit_transform(df_work[c].astype(str))

                    log.append(
                        f"✅ 인코딩 완료: OHE {len(ohe_cols)}개 / Label {len(le_cols)}개"
                    )

                    # ── STEP 4: Target 인코딩 (필요시) ───────
                    if not pd.api.types.is_numeric_dtype(df_work[target_col_quick]):
                        le = LabelEncoder()
                        df_work[target_col_quick] = le.fit_transform(
                            df_work[target_col_quick].astype(str)
                        )
                        log.append(f"✅ Target({target_col_quick}) 인코딩 완료")

                # ── 결과 저장 ─────────────────────────────────
                st.session_state.df = df_work
                st.session_state.encoded = True
                st.session_state.quick_preprocessed = True

                st.success("🎉 전처리 완료!")
                for msg in log:
                    st.write(msg)

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("최종 행 수", f"{df_work.shape[0]:,}행")
                with col_b:
                    st.metric("최종 열 수", f"{df_work.shape[1]:,}개")
                with col_c:
                    st.metric("잔여 결측치", f"{df_work.isnull().sum().sum()}개")

                import time
                time.sleep(0.5)
                st.rerun()

            except Exception as e:
                st.error(f"❌ 전처리 오류: {e}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # 탭 구성
    # ════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 이상치 처리",
        "🟡 결측치 처리",
        "🔵 수치형 변환",
        "🟢 원핫인코딩"
    ])

    # ════════════════════════════════════════════════════════
    # TAB 1: 이상치 처리
    # ════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### 🔴 이상치 처리")

        df_tab1 = st.session_state.df.copy()
        num_cols_tab1 = df_tab1.select_dtypes(include='number').columns.tolist()

        if not num_cols_tab1:
            st.info("수치형 컬럼이 없습니다.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                target_col_tab1 = st.selectbox(
                    "🎯 Target 컬럼 선택 (이상치 처리 제외)",
                    options=df_tab1.columns.tolist(),
                    key="tab1_target"
                )
            with col2:
                outlier_cols = st.multiselect(
                    "이상치 처리할 컬럼 선택",
                    options=[c for c in num_cols_tab1 if c != target_col_tab1],
                    default=[c for c in num_cols_tab1 if c != target_col_tab1],
                    key="outlier_cols"
                )

            outlier_method_tab1 = st.radio(
                "처리 방법",
                ["IQR 대체(중앙값)", "IQR 대체(평균)", "IQR 제거"],
                horizontal=True,
                key="outlier_method_tab1"
            )

            # 이상치 현황 미리보기
            if outlier_cols:
                st.markdown("**📊 이상치 현황**")
                outlier_summary = []
                for c in outlier_cols:
                    Q1 = df_tab1[c].quantile(0.25)
                    Q3 = df_tab1[c].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    cnt = ((df_tab1[c] < lower) | (df_tab1[c] > upper)).sum()
                    if cnt > 0:
                        outlier_summary.append({
                            "컬럼": c,
                            "이상치 수": cnt,
                            "하한": round(lower, 4),
                            "상한": round(upper, 4)
                        })

                if outlier_summary:
                    st.dataframe(
                        pd.DataFrame(outlier_summary),
                        use_container_width=True
                    )
                else:
                    st.success("✅ 이상치가 없습니다!")

            if st.button("🔴 이상치 처리 실행", key="btn_outlier", type="primary"):
                df_work = st.session_state.df.copy()
                total_cnt = 0

                try:
                    for c in outlier_cols:
                        Q1 = df_work[c].quantile(0.25)
                        Q3 = df_work[c].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        mask = (df_work[c] < lower) | (df_work[c] > upper)
                        cnt = mask.sum()

                        if cnt > 0:
                            if outlier_method_tab1 == "IQR 제거":
                                df_temp = df_work[~mask]
                                # ✅ 클래스 보호
                                if df_temp[target_col_tab1].nunique() >= 2:
                                    df_work = df_temp
                                else:
                                    df_work.loc[mask, c] = df_work[c].median()
                                    st.warning(
                                        f"⚠️ {c}: 제거 시 클래스 손실 → 중앙값 대체로 변경"
                                    )
                            elif outlier_method_tab1 == "IQR 대체(중앙값)":
                                df_work.loc[mask, c] = df_work[c].median()
                            else:
                                df_work.loc[mask, c] = df_work[c].mean()
                            total_cnt += cnt

                    st.session_state.df = df_work
                    st.success(f"✅ 이상치 처리 완료: 총 {total_cnt}개 처리")
                    st.metric("처리 후 행 수", f"{df_work.shape[0]:,}행")

                    import time
                    time.sleep(0.5)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ 이상치 처리 오류: {e}")

    # ════════════════════════════════════════════════════════
    # TAB 2: 결측치 처리
    # ════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 🟡 결측치 처리")

        df_tab2 = st.session_state.df.copy()
        missing_info = df_tab2.isnull().sum()
        missing_info = missing_info[missing_info > 0]

        if missing_info.empty:
            st.success("✅ 결측치가 없습니다!")
        else:
            # 결측치 현황
            st.markdown("**📊 결측치 현황**")
            missing_df = pd.DataFrame({
                "컬럼": missing_info.index,
                "결측치 수": missing_info.values,
                "결측 비율(%)": (missing_info.values / len(df_tab2) * 100).round(2),
                "데이터 타입": [str(df_tab2[c].dtype) for c in missing_info.index]
            })
            st.dataframe(missing_df, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                num_fill = st.radio(
                    "수치형 결측치 처리",
                    ["중앙값", "평균값", "최빈값", "0으로 대체"],
                    horizontal=True,
                    key="num_fill"
                )
            with col2:
                cat_fill = st.radio(
                    "범주형 결측치 처리",
                    ["최빈값", "Unknown으로 대체"],
                    horizontal=True,
                    key="cat_fill"
                )

            if st.button("🟡 결측치 처리 실행", key="btn_missing", type="primary"):
                df_work = st.session_state.df.copy()
                filled_count = 0

                try:
                    for c in df_work.columns:
                        if df_work[c].isnull().any():
                            cnt = df_work[c].isnull().sum()
                            if pd.api.types.is_numeric_dtype(df_work[c]):
                                if num_fill == "중앙값":
                                    df_work[c] = df_work[c].fillna(df_work[c].median())
                                elif num_fill == "평균값":
                                    df_work[c] = df_work[c].fillna(df_work[c].mean())
                                elif num_fill == "최빈값":
                                    mode_val = df_work[c].mode()
                                    df_work[c] = df_work[c].fillna(
                                        mode_val[0] if len(mode_val) > 0 else 0
                                    )
                                else:
                                    df_work[c] = df_work[c].fillna(0)
                            else:
                                if cat_fill == "최빈값":
                                    mode_val = df_work[c].mode()
                                    df_work[c] = df_work[c].fillna(
                                        mode_val[0] if len(mode_val) > 0 else "Unknown"
                                    )
                                else:
                                    df_work[c] = df_work[c].fillna("Unknown")
                            filled_count += cnt

                    st.session_state.df = df_work
                    st.success(f"✅ 결측치 처리 완료: {filled_count}개 대체")
                    st.metric("잔여 결측치", f"{df_work.isnull().sum().sum()}개")

                    import time
                    time.sleep(0.5)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ 결측치 처리 오류: {e}")

    # ════════════════════════════════════════════════════════
    # TAB 3: 수치형 변환
    # ════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 🔵 수치형 변환")

        df_tab3 = st.session_state.df.copy()
        num_cols_tab3 = df_tab3.select_dtypes(include='number').columns.tolist()

        if not num_cols_tab3:
            st.info("수치형 컬럼이 없습니다.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                target_col_tab3 = st.selectbox(
                    "🎯 Target 컬럼 선택 (변환 제외)",
                    options=df_tab3.columns.tolist(),
                    key="tab3_target"
                )
            with col2:
                scale_method = st.radio(
                    "변환 방법",
                    ["StandardScaler", "MinMaxScaler", "Log변환"],
                    horizontal=True,
                    key="scale_method"
                )

            scale_cols = st.multiselect(
                "변환할 컬럼 선택",
                options=[c for c in num_cols_tab3 if c != target_col_tab3],
                default=[c for c in num_cols_tab3 if c != target_col_tab3],
                key="scale_cols"
            )

            if st.button("🔵 수치형 변환 실행", key="btn_scale", type="primary"):
                df_work = st.session_state.df.copy()

                try:
                    from sklearn.preprocessing import StandardScaler, MinMaxScaler

                    if scale_method == "StandardScaler":
                        scaler = StandardScaler()
                        df_work[scale_cols] = scaler.fit_transform(df_work[scale_cols])
                    elif scale_method == "MinMaxScaler":
                        scaler = MinMaxScaler()
                        df_work[scale_cols] = scaler.fit_transform(df_work[scale_cols])
                    else:
                        for c in scale_cols:
                            if (df_work[c] > 0).all():
                                df_work[c] = np.log(df_work[c])
                            else:
                                df_work[c] = np.log1p(df_work[c])

                    st.session_state.df = df_work
                    st.success(f"✅ 수치형 변환 완료: {len(scale_cols)}개 컬럼")

                    import time
                    time.sleep(0.5)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ 수치형 변환 오류: {e}")

    # ════════════════════════════════════════════════════════
    # TAB 4: 원핫인코딩
    # ════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 🟢 원핫인코딩")

        df_tab4 = st.session_state.df.copy()
        cat_cols_tab4 = [
            c for c in df_tab4.columns
            if not pd.api.types.is_numeric_dtype(df_tab4[c])
        ]

        if not cat_cols_tab4:
            st.success("✅ 인코딩할 범주형 컬럼이 없습니다!")
        else:
            col1, col2 = st.columns(2)
            with col1:
                target_col_tab4 = st.selectbox(
                    "🎯 Target 컬럼 선택 (인코딩 제외)",
                    options=df_tab4.columns.tolist(),
                    key="tab4_target"
                )
            with col2:
                encode_method = st.radio(
                    "인코딩 방법",
                    ["One-Hot Encoding", "Label Encoding", "자동(고유값 기준)"],
                    horizontal=True,
                    key="encode_method"
                )

            encode_cols = st.multiselect(
                "인코딩할 컬럼 선택",
                options=[c for c in cat_cols_tab4 if c != target_col_tab4],
                default=[c for c in cat_cols_tab4 if c != target_col_tab4],
                key="encode_cols"
            )

            if encode_cols:
                st.markdown("**📊 인코딩 대상 컬럼 현황**")
                encode_summary = []
                for c in encode_cols:
                    encode_summary.append({
                        "컬럼": c,
                        "고유값 수": df_tab4[c].nunique(),
                        "샘플": str(df_tab4[c].unique()[:3].tolist())
                    })
                st.dataframe(
                    pd.DataFrame(encode_summary),
                    use_container_width=True
                )

            if st.button("🟢 인코딩 실행", key="btn_encode", type="primary"):
                df_work = st.session_state.df.copy()

                try:
                    if encode_method == "One-Hot Encoding":
                        df_work = pd.get_dummies(
                            df_work,
                            columns=encode_cols,
                            drop_first=True
                        )
                        bool_cols = [
                            c for c in df_work.columns
                            if df_work[c].dtype == bool
                        ]
                        if bool_cols:
                            df_work[bool_cols] = df_work[bool_cols].astype(int)

                    elif encode_method == "Label Encoding":
                        le = LabelEncoder()
                        for c in encode_cols:
                            df_work[c] = le.fit_transform(df_work[c].astype(str))

                    else:  # 자동
                        ohe_cols = [c for c in encode_cols if df_work[c].nunique() <= 10]
                        le_cols  = [c for c in encode_cols if df_work[c].nunique() > 10]

                        if ohe_cols:
                            df_work = pd.get_dummies(
                                df_work,
                                columns=ohe_cols,
                                drop_first=True
                            )
                            bool_cols = [
                                c for c in df_work.columns
                                if df_work[c].dtype == bool
                            ]
                            if bool_cols:
                                df_work[bool_cols] = df_work[bool_cols].astype(int)

                        if le_cols:
                            le = LabelEncoder()
                            for c in le_cols:
                                df_work[c] = le.fit_transform(df_work[c].astype(str))

                    # ✅ Target 컬럼 보호 확인
                    if target_col_tab4 not in df_work.columns:
                        st.error(
                            f"❌ Target 컬럼({target_col_tab4})이 사라졌습니다! "
                            f"인코딩 대상에서 제외해주세요."
                        )
                    else:
                        # ✅ 클래스 수 확인
                        n_classes = df_work[target_col_tab4].nunique()
                        if n_classes < 2:
                            st.error(
                                f"❌ Target 클래스가 {n_classes}개입니다. "
                                f"최소 2개 이상이어야 합니다."
                            )
                        else:
                            st.session_state.df = df_work
                            st.session_state.encoded = True
                            st.success(
                                f"✅ 인코딩 완료! "
                                f"컬럼 수: {df_tab4.shape[1]} → {df_work.shape[1]}"
                            )

                            import time
                            time.sleep(0.5)
                            st.rerun()

                except Exception as e:
                    st.error(f"❌ 인코딩 오류: {e}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # Feature Selection
    # ════════════════════════════════════════════════════════
    st.markdown("### 🎯 Feature Selection")

    df_fs = st.session_state.df.copy()

    col1, col2 = st.columns(2)
    with col1:
        target_fs = st.selectbox(
            "🎯 종속변수(Target) 선택",
            options=df_fs.columns.tolist(),
            key="fs_target_main"
        )
    with col2:
        fs_method = st.radio(
            "Feature Selection 방법",
            ["상관계수", "Random Forest 중요도", "Stepwise Selection"],
            horizontal=True,
            key="fs_method"
        )

    # ── 상관계수 ──────────────────────────────────────────────
    if fs_method == "상관계수":
        threshold_corr = st.slider(
            "상관계수 임계값 (절댓값)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            key="corr_threshold"
        )

        if st.button("📊 상관계수 기반 선택", key="btn_corr", type="primary"):
            try:
                num_df = df_fs.select_dtypes(include='number')
                if target_fs in num_df.columns:
                    corr = num_df.corr()[target_fs].drop(target_fs)
                    selected = corr[corr.abs() >= threshold_corr].index.tolist()

                    corr_df = pd.DataFrame({
                        "변수": corr.index,
                        "상관계수": corr.values,
                        "절댓값": corr.abs().values,
                        "선택여부": ["✅" if c in selected else "❌" for c in corr.index]
                    }).sort_values("절댓값", ascending=False)

                    st.dataframe(corr_df, use_container_width=True)
                    st.info(f"📌 선택된 변수: {len(selected)}개")

                    if st.button("✅ 선택된 변수로 업데이트", key="btn_apply_corr"):
                        keep_cols = selected + [target_fs]
                        st.session_state.df = df_fs[keep_cols].copy()
                        st.success("✅ 데이터셋 업데이트 완료!")
                        import time
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.warning("⚠️ Target이 수치형이 아닙니다.")
            except Exception as e:
                st.error(f"❌ 오류: {e}")

    # ── Random Forest 중요도 ──────────────────────────────────
    elif fs_method == "Random Forest 중요도":
        top_n = st.slider(
            "상위 N개 변수 선택",
            min_value=1,
            max_value=min(50, df_fs.shape[1] - 1),
            value=min(20, df_fs.shape[1] - 1),
            key="rf_top_n"
        )

        if st.button("🌲 Random Forest 중요도 계산", key="btn_rf", type="primary"):
            try:
                from sklearn.ensemble import RandomForestClassifier

                feature_cols = [c for c in df_fs.columns if c != target_fs]
                X_rf = df_fs[feature_cols].select_dtypes(include='number')
                X_rf = X_rf.fillna(X_rf.median())
                y_rf = df_fs[target_fs]

                # ✅ 클래스 수 확인
                if y_rf.nunique() < 2:
                    st.error("❌ Target 클래스가 1개입니다. 데이터를 확인해주세요.")
                else:
                    with st.spinner("Random Forest 학습 중..."):
                        rf = RandomForestClassifier(
                            n_estimators=100,
                            random_state=42,
                            n_jobs=-1
                        )
                        rf.fit(X_rf, y_rf)

                    importance_df = pd.DataFrame({
                        "변수": X_rf.columns,
                        "중요도": rf.feature_importances_
                    }).sort_values("중요도", ascending=False).head(top_n)

                    st.dataframe(importance_df, use_container_width=True)

                    import plotly.express as px
                    fig = px.bar(
                        importance_df,
                        x="중요도",
                        y="변수",
                        orientation='h',
                        title=f"Top {top_n} Feature Importance"
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)

                    selected_rf = importance_df["변수"].tolist()

                    if st.button("✅ 선택된 변수로 업데이트", key="btn_apply_rf"):
                        keep_cols = selected_rf + [target_fs]
                        st.session_state.df = df_fs[keep_cols].copy()
                        st.success("✅ 데이터셋 업데이트 완료!")
                        import time
                        time.sleep(0.5)
                        st.rerun()

            except Exception as e:
                st.error(f"❌ 오류: {e}")

    # ── Stepwise Selection ────────────────────────────────────
    elif fs_method == "Stepwise Selection":
        with st.expander("🔍 Stepwise Selection 설정", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                step_direction = st.radio(
                    "탐색 방향",
                    ["Forward", "Backward", "Both"],
                    horizontal=True,
                    key="step_direction"
                )
            with col2:
                step_criterion = st.radio(
                    "선택 기준",
                    ["p-value", "AIC"],
                    horizontal=True,
                    key="step_criterion"
                )

            if step_criterion == "p-value":
                threshold_step = st.slider(
                    "p-value 임계값",
                    min_value=0.01,
                    max_value=0.10,
                    value=0.05,
                    step=0.01,
                    key="step_threshold"
                )
            else:
                threshold_step = None

            max_features_step = st.slider(
                "최대 선택 변수 수",
                min_value=1,
                max_value=min(50, df_fs.shape[1] - 1),
                value=min(20, df_fs.shape[1] - 1),
                key="step_max_features"
            )

        if st.button("🔍 Stepwise Selection 실행", key="btn_stepwise", type="primary"):
            try:
                import statsmodels.api as sm

                feature_cols = [c for c in df_fs.columns if c != target_fs]
                X_all = df_fs[feature_cols].select_dtypes(include='number')
                X_all = X_all.fillna(X_all.median())
                y_all = df_fs[target_fs].copy()

                # ✅ 클래스 수 확인
                if y_all.nunique() < 2:
                    st.error("❌ Target 클래스가 1개입니다. 데이터를 확인해주세요.")
                    st.stop()

                def stepwise_selection(X, y, direction, criterion,
                                       threshold=0.05, max_feat=20):
                    selected = []
                    best_aic = float('inf')
                    step_log = []

                    if direction in ["Forward", "Both"]:
                        remaining = list(X.columns)

                        while remaining and len(selected) < max_feat:
                            scores = {}
                            for col in remaining:
                                candidate = selected + [col]
                                try:
                                    X_cand = sm.add_constant(X[candidate])
                                    model = sm.Logit(y, X_cand).fit(disp=0)
                                    if criterion == "p-value":
                                        scores[col] = model.pvalues[col]
                                    else:
                                        scores[col] = model.aic
                                except:
                                    continue

                            if not scores:
                                break

                            best_col = min(scores, key=scores.get)
                            best_score = scores[best_col]

                            if criterion == "p-value":
                                if best_score < threshold:
                                    selected.append(best_col)
                                    remaining.remove(best_col)
                                    step_log.append(
                                        f"➕ 추가: **{best_col}** (p={best_score:.4f})"
                                    )
                                else:
                                    break
                            else:
                                if best_score < best_aic:
                                    best_aic = best_score
                                    selected.append(best_col)
                                    remaining.remove(best_col)
                                    step_log.append(
                                        f"➕ 추가: **{best_col}** (AIC={best_score:.2f})"
                                    )
                                else:
                                    break

                            # Both: Backward 검토
                            if direction == "Both" and len(selected) > 1:
                                while True:
                                    pvals = {}
                                    try:
                                        X_sel = sm.add_constant(X[selected])
                                        model = sm.Logit(y, X_sel).fit(disp=0)
                                        for col in selected:
                                            pvals[col] = model.pvalues.get(col, 0)
                                    except:
                                        break

                                    worst_col = max(pvals, key=pvals.get)
                                    if pvals[worst_col] > threshold:
                                        selected.remove(worst_col)
                                        step_log.append(
                                            f"➖ 제거: **{worst_col}** "
                                            f"(p={pvals[worst_col]:.4f})"
                                        )
                                    else:
                                        break

                    elif direction == "Backward":
                        selected = list(X.columns)[:max_feat]

                        while len(selected) > 1:
                            try:
                                X_sel = sm.add_constant(X[selected])
                                model = sm.Logit(y, X_sel).fit(disp=0)

                                if criterion == "p-value":
                                    pvals = {
                                        c: model.pvalues.get(c, 0)
                                        for c in selected
                                    }
                                    worst_col = max(pvals, key=pvals.get)
                                    if pvals[worst_col] > threshold:
                                        selected.remove(worst_col)
                                        step_log.append(
                                            f"➖ 제거: **{worst_col}** "
                                            f"(p={pvals[worst_col]:.4f})"
                                        )
                                    else:
                                        break
                                else:
                                    current_aic = model.aic
                                    aic_scores = {}
                                    for col in selected:
                                        candidate = [c for c in selected if c != col]
                                        try:
                                            X_cand = sm.add_constant(X[candidate])
                                            m = sm.Logit(y, X_cand).fit(disp=0)
                                            aic_scores[col] = m.aic
                                        except:
                                            continue

                                    if not aic_scores:
                                        break

                                    best_remove = min(aic_scores, key=aic_scores.get)
                                    if aic_scores[best_remove] < current_aic:
                                        selected.remove(best_remove)
                                        step_log.append(
                                            f"➖ 제거: **{best_remove}** "
                                            f"(AIC {current_aic:.2f}→{aic_scores[best_remove]:.2f})"
                                        )
                                    else:
                                        break
                            except:
                                break

                    return selected, step_log

                with st.spinner("Stepwise Selection 진행 중..."):
                    selected_features, step_log = stepwise_selection(
                        X_all, y_all,
                        direction=step_direction,
                        criterion=step_criterion,
                        threshold=threshold_step if threshold_step else 0.05,
                        max_feat=max_features_step
                    )

                # ── 결과 출력 ─────────────────────────────────
                st.success(f"✅ Stepwise 완료! **{len(selected_features)}개** 변수 선택됨")

                with st.expander("📋 단계별 선택 과정", expanded=False):
                    for log_msg in step_log:
                        st.markdown(log_msg)

                if selected_features:
                    result_df = pd.DataFrame({
                        "순위": range(1, len(selected_features) + 1),
                        "변수명": selected_features
                    })
                    st.dataframe(result_df, use_container_width=True)

                    # 최종 모델 통계
                    try:
                        X_final = sm.add_constant(X_all[selected_features])
                        final_model = sm.Logit(y_all, X_final).fit(disp=0)

                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("선택된 변수 수", f"{len(selected_features)}개")
                        with col_b:
                            st.metric("최종 AIC", f"{final_model.aic:.2f}")
                        with col_c:
                            st.metric("Pseudo R²", f"{final_model.prsquared:.4f}")

                        with st.expander("📊 최종 모델 요약", expanded=False):
                            st.text(final_model.summary().as_text())
                    except Exception as e:
                        st.warning(f"모델 요약 생성 실패: {e}")

                    # 데이터셋 적용
                    st.markdown("---")
                    if st.button("✅ 선택된 변수로 데이터셋 업데이트", key="btn_apply_stepwise"):
                        keep_cols = selected_features + [target_fs]
                        st.session_state.df = df_fs[keep_cols].copy()
                        st.success(
                            f"✅ 데이터셋 업데이트 완료! "
                            f"({len(keep_cols)}개 컬럼 유지)"
                        )
                        import time
                        time.sleep(0.5)
                        st.rerun()

            except ImportError:
                st.error("❌ statsmodels 패키지 필요: pip install statsmodels")
            except Exception as e:
                st.error(f"❌ Stepwise 오류: {e}")
                st.info("💡 변수가 너무 많거나 다중공선성이 있을 수 있습니다.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # 학습 데이터 분할
    # ════════════════════════════════════════════════════════
    st.markdown("### ✂️ 학습/테스트 데이터 분할")

    df_split = st.session_state.df.copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        target_split = st.selectbox(
            "🎯 Target 컬럼 선택",
            options=df_split.columns.tolist(),
            key="split_target"
        )
    with col2:
        test_size = st.slider(
            "테스트 데이터 비율",
            min_value=0.1,
            max_value=0.4,
            value=0.3,
            step=0.05,
            key="test_size"
        )
    with col3:
        random_state = st.number_input(
            "Random State",
            min_value=0,
            max_value=999,
            value=42,
            key="random_state"
        )

    if st.button("✂️ 데이터 분할 실행", key="btn_split", type="primary"):
        try:
            X = df_split.drop(columns=[target_split])
            y = df_split[target_split]

            # ✅ 클래스 수 확인
            if y.nunique() < 2:
                st.error(
                    f"❌ Target 클래스가 {y.nunique()}개입니다. "
                    f"최소 2개 이상이어야 합니다.\n\n"
                    f"현재 값: {y.unique()}"
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y  # ✅ 클래스 비율 유지
                )

                # 세션에 저장
                st.session_state.X_train = X_train
                st.session_state.X_test  = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test  = y_test
                st.session_state.target_col = target_split
                st.session_state.data_split = True

                st.success("✅ 데이터 분할 완료!")

                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Train 행 수", f"{X_train.shape[0]:,}행")
                with col_b:
                    st.metric("Test 행 수", f"{X_test.shape[0]:,}행")
                with col_c:
                    st.metric("Feature 수", f"{X_train.shape[1]:,}개")
                with col_d:
                    st.metric("Target", target_split)

                # 클래스 분포 확인
                st.markdown("**📊 클래스 분포 확인**")
                col_e, col_f = st.columns(2)
                with col_e:
                    st.write("Train 클래스 분포:")
                    st.write(y_train.value_counts())
                with col_f:
                    st.write("Test 클래스 분포:")
                    st.write(y_test.value_counts())

        except Exception as e:
            st.error(f"❌ 데이터 분할 오류: {e}")

# ══════════════════════════════════════════════════════════════════
#  PAGE 4 ── 연구 모형
# ══════════════════════════════════════════════════════════════════
elif current == "model":
    check_data()
    st.markdown("## 🤖 연구 모형")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if st.session_state.X_train is None:
        st.warning("⚠️ **데이터 전처리** 페이지에서 데이터 분할을 먼저 완료해 주세요.")
        st.stop()

    X_train = st.session_state.X_train
    X_test  = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test  = st.session_state.y_test

    # 데이터 현황
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📊 학습 데이터 현황</div>
        Train: <b>{len(X_train):,}행</b> &nbsp;|&nbsp;
        Test: <b>{len(X_test):,}행</b> &nbsp;|&nbsp;
        Feature 수: <b>{X_train.shape[1]}</b> &nbsp;|&nbsp;
        Target: <b>{st.session_state.selected_y}</b>
    </div>""", unsafe_allow_html=True)

    model_tab1, model_tab2 = st.tabs(
        ["📉 Logistic Regression", "🌳 Decision Tree"]
    )

    # ── Logistic Regression ─────────────────────────────────
    with model_tab1:
        st.markdown("#### Logistic Regression 하이퍼파라미터")
        lr_col1, lr_col2, lr_col3 = st.columns(3)
        with lr_col1:
            lr_C = st.select_slider("규제 강도 C",
                                    options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                                    value=1.0)
        with lr_col2:
            lr_max_iter = st.slider("최대 반복 횟수", 100, 2000, 1000, 100)
        with lr_col3:
            lr_solver = st.selectbox("Solver",
                                     ["lbfgs", "liblinear", "saga", "newton-cg"])

        if st.button("🚀 Logistic Regression 학습", key="btn_lr"):
            with st.spinner("학습 중..."):
                try:
                    lr = LogisticRegression(
                        C=lr_C, max_iter=lr_max_iter,
                        solver=lr_solver, random_state=42
                    )
                    lr.fit(X_train, y_train)
                    st.session_state.lr_model  = lr
                    st.session_state.lr_result = compute_metrics(lr, X_test, y_test)
                    st.success("✅ Logistic Regression 학습 완료!")
                except Exception as e:
                    st.error(f"❌ 학습 오류: {e}")

        if st.session_state.lr_result:
            r = st.session_state.lr_result
            st.markdown("##### 📊 학습 결과")
            m1, m2, m3, m4 = st.columns(4)
            for col, (lbl, val) in zip(
                [m1, m2, m3, m4],
                [("Accuracy", r["accuracy"]), ("Precision", r["precision"]),
                 ("Recall",   r["recall"]),   ("F1-Score",  r["f1"])]
            ):
                with col:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="val">{val:.4f}</div>
                        <div class="lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            # 혼동 행렬
            st.markdown("<br>", unsafe_allow_html=True)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues",
                        ax=ax_cm, linewidths=0.5)
            ax_cm.set_title("Confusion Matrix", fontweight="bold")
            ax_cm.set_xlabel("예측값"); ax_cm.set_ylabel("실제값")
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=False)
            plt.close()

    # ── Decision Tree ───────────────────────────────────────
    with model_tab2:
        st.markdown("#### Decision Tree 하이퍼파라미터")
        dt_col1, dt_col2, dt_col3 = st.columns(3)
        with dt_col1:
            dt_max_depth = st.slider("최대 깊이 (max_depth)", 1, 20, 5)
        with dt_col2:
            dt_min_samples = st.slider("최소 분할 샘플 수", 2, 50, 2)
        with dt_col3:
            dt_criterion = st.selectbox("분할 기준", ["gini", "entropy", "log_loss"])

        if st.button("🚀 Decision Tree 학습", key="btn_dt"):
            with st.spinner("학습 중..."):
                try:
                    dt = DecisionTreeClassifier(
                        max_depth=dt_max_depth,
                        min_samples_split=dt_min_samples,
                        criterion=dt_criterion,
                        random_state=42
                    )
                    dt.fit(X_train, y_train)
                    st.session_state.dt_model  = dt
                    st.session_state.dt_result = compute_metrics(dt, X_test, y_test)
                    st.success("✅ Decision Tree 학습 완료!")
                except Exception as e:
                    st.error(f"❌ 학습 오류: {e}")

        if st.session_state.dt_result:
            r = st.session_state.dt_result
            st.markdown("##### 📊 학습 결과")
            m1, m2, m3, m4 = st.columns(4)
            for col, (lbl, val) in zip(
                [m1, m2, m3, m4],
                [("Accuracy", r["accuracy"]), ("Precision", r["precision"]),
                 ("Recall",   r["recall"]),   ("F1-Score",  r["f1"])]
            ):
                with col:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="val">{val:.4f}</div>
                        <div class="lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            # Feature Importance
            st.markdown("<br>", unsafe_allow_html=True)
            fi = pd.Series(
                st.session_state.dt_model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=True).tail(15)

            fig_fi, ax_fi = plt.subplots(figsize=(7, 4))
            fi.plot(kind="barh", ax=ax_fi, color="#667eea", edgecolor="white")
            ax_fi.set_title("Feature Importance (Top 15)", fontweight="bold")
            ax_fi.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_fi, use_container_width=True)
            plt.close()

# ══════════════════════════════════════════════════════════════════
#  PAGE 5 ── 연구 결과
# ══════════════════════════════════════════════════════════════════
elif current == "result":
    check_data()
    st.markdown("## 📈 연구 결과")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    lr_r = st.session_state.lr_result
    dt_r = st.session_state.dt_result

    if lr_r is None and dt_r is None:
        st.warning("⚠️ **연구 모형** 페이지에서 모델을 먼저 학습해 주세요.")
        st.stop()

    # ── 성능 비교 테이블 ───────────────────────────────────────
    st.markdown("### 📊 모형 성능 비교")

    metrics_rows = []
    for name, r in [("Logistic Regression", lr_r), ("Decision Tree", dt_r)]:
        if r:
            metrics_rows.append({
                "모형":       name,
                "Accuracy":  f"{r['accuracy']:.4f}",
                "Precision": f"{r['precision']:.4f}",
                "Recall":    f"{r['recall']:.4f}",
                "F1-Score":  f"{r['f1']:.4f}",
                "AUC":       f"{r['auc']:.4f}" if r["auc"] else "N/A",
            })

    if metrics_rows:
        result_df = pd.DataFrame(metrics_rows).set_index("모형")
        st.dataframe(result_df.style.highlight_max(
            axis=0, color="#d4edda", subset=["Accuracy", "Precision",
                                              "Recall", "F1-Score", "AUC"]
        ), use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 메트릭 시각화 ─────────────────────────────────────────
    st.markdown("### 📉 성능 지표 시각화")

    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    lr_vals = ([lr_r["accuracy"], lr_r["precision"], lr_r["recall"], lr_r["f1"]]
               if lr_r else None)
    dt_vals = ([dt_r["accuracy"], dt_r["precision"], dt_r["recall"], dt_r["f1"]]
               if dt_r else None)

    fig_bar, ax_bar = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(metric_names))
    width = 0.35

    if lr_vals:
        bars1 = ax_bar.bar(x - width/2, lr_vals, width,
                           label="Logistic Regression",
                           color="#667eea", edgecolor="white", alpha=0.9)
        for bar in bars1:
            ax_bar.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{bar.get_height():.3f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    if dt_vals:
        bars2 = ax_bar.bar(x + width/2, dt_vals, width,
                           label="Decision Tree",
                           color="#764ba2", edgecolor="white", alpha=0.9)
        for bar in bars2:
            ax_bar.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{bar.get_height():.3f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names, fontsize=11)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_ylabel("Score", fontsize=11)
    ax_bar.set_title("모형별 성능 지표 비교", fontsize=13, fontweight="bold")
    ax_bar.legend(fontsize=10)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── ROC Curve ─────────────────────────────────────────────
    st.markdown("### 📈 ROC Curve 비교")

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Random (AUC = 0.50)")

    colors = {"Logistic Regression": "#667eea", "Decision Tree": "#764ba2"}
    for name, r in [("Logistic Regression", lr_r), ("Decision Tree", dt_r)]:
        if r and r["fpr"] is not None:
            ax_roc.plot(r["fpr"], r["tpr"],
                        color=colors[name], lw=2.5,
                        label=f"{name} (AUC = {r['auc']:.4f})")
            ax_roc.fill_between(r["fpr"], r["tpr"], alpha=0.08,
                                color=colors[name])

    ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.02])
    ax_roc.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax_roc.set_title("ROC Curve 비교", fontsize=14, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=11)
    ax_roc.spines[["top", "right"]].set_visible(False)
    ax_roc.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_roc, use_container_width=True)
    plt.close()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 혼동 행렬 비교 ─────────────────────────────────────────
    st.markdown("### 🔲 Confusion Matrix 비교")
    cm_cols = st.columns(2)
    for col, (name, r) in zip(
        cm_cols,
        [("Logistic Regression", lr_r), ("Decision Tree", dt_r)]
    ):
        if r:
            with col:
                st.markdown(f"**{name}**")
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(r["cm"], annot=True, fmt="d",
                            cmap="Blues", ax=ax_cm, linewidths=0.5,
                            cbar_kws={"shrink": 0.8})
                ax_cm.set_title(f"{name}", fontsize=11, fontweight="bold")
                ax_cm.set_xlabel("예측값"); ax_cm.set_ylabel("실제값")
                plt.tight_layout()
                st.pyplot(fig_cm, use_container_width=True)
                plt.close()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 결과 다운로드 ─────────────────────────────────────────
    st.markdown("### 💾 결과 저장")
    if metrics_rows:
        csv_result = result_df.to_csv(encoding="utf-8-sig")
        st.download_button(
            label="📥 성능 지표 CSV 다운로드",
            data=csv_result,
            file_name="model_performance.csv",
            mime="text/csv"
        )
