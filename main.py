import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------
# 기본 설정
# ---------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
CUTOFF_YEAR = 2025

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 과거 기온 데이터를 바탕으로 연평균기온의 추세를 분석하고 "
    "선택한 연도의 예상 평균기온을 확인합니다."
)


# ---------------------------------
# 데이터 불러오기 및 전처리
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "날짜", "지점", "평균기온", "최저기온", "최고기온"
    ]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"필요한 열이 없습니다: {', '.join(missing_columns)}"
        )

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 날짜와 평균기온이 유효한 관측만 사용
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 수업 기준 기간: 2025년까지
    df = df[df["날짜"].dt.year <= CUTOFF_YEAR].copy()

    df["연도"] = df["날짜"].dt.year

    # 연도별 유효 관측일 수와 연평균기온 계산
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique"),
        )
        .reset_index()
    )

    # 관측일이 300일 이상인 해만 사용
    annual = annual[annual["관측일수"] >= 300].copy()
    annual = annual.sort_values("연도").reset_index(drop=True)

    return annual


try:
    with st.spinner("서울 기온 데이터를 불러오는 중입니다..."):
        annual = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.info("인터넷 연결과 데이터 URL을 확인한 뒤 다시 실행해 주세요.")
    st.stop()

if len(annual) < 2:
    st.error("회귀분석을 수행하기 위한 유효한 연도가 부족합니다.")
    st.stop()


# ---------------------------------
# 선형회귀 및 상관계수
# ---------------------------------
x = annual["연도"].to_numpy(dtype=float)
y = annual["연평균기온"].to_numpy(dtype=float)

# 최소제곱법으로 회귀 직선 계산
slope, intercept = np.polyfit(x, y, 1)

# 피어슨 상관계수
correlation = np.corrcoef(x, y)[0, 1]

start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
year_count = len(annual)


def predict_temperature(year):
    """회귀 직선을 이용한 예상 연평균기온."""
    return slope * year + intercept


# ---------------------------------
# 회귀분석 결과
# ---------------------------------
st.subheader("📊 회귀분석 결과")

col1, col2, col3, col4 = st.columns(4)

col1.metric("회귀에 사용한 연도 수", f"{year_count:,}년")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수", f"{correlation:.4f}")

st.caption(
    f"분석 기준: 2025년까지의 데이터 중 유효 관측일이 "
    f"300일 이상인 연도. 회귀식: "
    f"연평균기온 = {slope:.5f} × 연도 + ({intercept:.3f})"
)

# ---------------------------------
# 연도 선택 및 예상 기온 표시
# ---------------------------------
st.subheader("🌡️ 미래 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
    format="%d년",
)

predicted_temp = predict_temperature(selected_year)

st.markdown(
    f"""
    <div style="
        padding: 24px;
        border-radius: 16px;
        background-color: rgba(100, 149, 237, 0.12);
        text-align: center;
        margin: 8px 0 20px 0;
    ">
        <div style="font-size: 20px; margin-bottom: 8px;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="
            font-size: 52px;
            font-weight: 700;
            line-height: 1.2;
        ">
            {predicted_temp:.2f} °C
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if selected_year > end_year:
    st.warning(
        "선택한 연도는 실제 관측 데이터의 마지막 연도 이후입니다. "
        "과거의 직선을 미래로 연장한 추정값이므로 실제 기온을 "
        "정확하게 예측한다고 보장할 수 없습니다."
    )


# ---------------------------------
# Plotly 산점도와 회귀 직선
# ---------------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

# 실제 관측값
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7,
            color="#2471A3",
            opacity=0.8,
        ),
        customdata=annual[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        ),
    )
)

# 회귀 직선은 1900~2100년 범위로 표시
line_years = np.arange(1900, 2101)
line_temps = predict_temperature(line_years)

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="선형회귀 직선",
        line=dict(
            color="#E74C3C",
            width=3,
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 추정 기온: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

# 선택한 연도 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(
            size=14,
            color="#F39C12",
            symbol="diamond",
            line=dict(color="white", width=1),
        ),
        hovertemplate=(
            f"{selected_year}년 예상 기온: "
            "%{y:.2f} °C<extra></extra>"
        ),
    )
)

# 실제 관측 연도와 미래 예측 구간 구분
if end_year < 2100:
    fig.add_vline(
        x=end_year,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"마지막 관측 연도: {end_year}",
        annotation_position="top right",
    )

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    template="plotly_white",
    height=560,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=20, r=20, t=70, b=20),
)

fig.update_xaxes(range=[1900, 2100], dtick=20)
fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# 연도별 집계 데이터
# ---------------------------------
with st.expander("연도별 분석 데이터 확인"):
    display_df = annual.copy()
    display_df["연평균기온"] = display_df["연평균기온"].round(2)
    display_df = display_df.rename(
        columns={
            "연도": "연도",
            "연평균기온": "연평균기온(°C)",
            "관측일수": "관측일수(일)",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.caption(
    "참고: 연평균기온은 각 연도의 유효한 일평균기온을 산술평균한 값입니다. "
    "선형회귀는 시간에 따른 직선형 추세만 반영하며, "
    "장기적인 기후변화나 미래의 실제 기온을 완전히 설명하지 못합니다."
)
