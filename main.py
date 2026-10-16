
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------
# 1. 기본 설정
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

st.title("🌡️ 서울 기온 예측기")
st.write("과거 서울의 연평균기온을 분석하고 연도별 예상 기온을 확인합니다.")


# ---------------------------------
# 2. 데이터 불러오기 및 연도별 집계
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required = ["날짜", "지점", "평균기온", "최저기온", "최고기온"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"CSV에 필요한 열이 없습니다: {missing}")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df = df[df["날짜"].dt.year <= CUTOFF_YEAR].copy()
    df["연도"] = df["날짜"].dt.year

    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique"),
        )
        .reset_index()
    )

    # 유효 관측일이 300일 이상인 연도만 사용
    annual = annual[annual["관측일수"] >= 300]
    annual = annual.sort_values("연도").reset_index(drop=True)

    return annual


try:
    with st.spinner("서울 기온 데이터를 불러오는 중입니다..."):
        annual = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.info("인터넷 연결과 CSV 주소를 확인해 주세요.")
    st.stop()

if len(annual) < 2:
    st.error("회귀분석에 사용할 수 있는 연도가 2개 미만입니다.")
    st.stop()


# ---------------------------------
# 3. 전체 기간 및 최근 20년 회귀분석
# ---------------------------------
x = annual["연도"].to_numpy(dtype=float)
y = annual["연평균기온"].to_numpy(dtype=float)

# 전체 기간 회귀 직선
slope, intercept = np.polyfit(x, y, 1)
correlation = np.corrcoef(x, y)[0, 1]

start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
year_count = len(annual)

# 마지막 관측 연도를 포함하는 최근 20년 범위
recent_start_year = end_year - 19
recent = annual[annual["연도"] >= recent_start_year].copy()

if len(recent) >= 2:
    recent_x = recent["연도"].to_numpy(dtype=float)
    recent_y = recent["연평균기온"].to_numpy(dtype=float)

    recent_slope, recent_intercept = np.polyfit(recent_x, recent_y, 1)
    recent_correlation = np.corrcoef(recent_x, recent_y)[0, 1]
else:
    recent_slope = np.nan
    recent_intercept = np.nan
    recent_correlation = np.nan

# 연간 기울기를 100년당 기온 변화량으로 환산
warming_100 = slope * 100
recent_warming_100 = recent_slope * 100


def predict_temperature(year):
    return slope * year + intercept


# ---------------------------------
# 4. 100년당 기온 변화 비교
# ---------------------------------
st.subheader("📈 100년 동안 기온이 얼마나 변할까?")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label=f"전체 기간 ({start_year}~{end_year})",
        value=f"{warming_100:+.2f} °C",
    )
    st.caption(
        f"회귀에 사용한 연도: {year_count}개 | "
        f"상관계수: {correlation:.3f}"
    )

with col2:
    if np.isfinite(recent_warming_100):
        st.metric(
            label=f"최근 20년 범위 ({recent_start_year}~{end_year})",
            value=f"{recent_warming_100:+.2f} °C",
            delta=(
                f"{recent_warming_100 - warming_100:+.2f} °C"
                " / 100년 (전체 기간 대비)"
            ),
        )
        st.caption(
            f"회귀에 사용한 연도: {len(recent)}개 | "
            f"상관계수: {recent_correlation:.3f}"
        )
    else:
        st.metric(label="최근 20년 회귀", value="계산 불가")
        st.caption("유효한 연도가 2개 미만입니다.")

st.info(
    "표시된 값은 회귀 직선의 기울기를 100배 한 추세입니다. "
    "실제 100년간의 기온 변화량이나 미래 기온을 보장하는 값은 아닙니다."
)

st.caption(
    f"전체 회귀식: 연평균기온 = "
    f"{slope:.5f} × 연도 + ({intercept:.3f})"
)


# ---------------------------------
# 5. 연도 선택 및 예상 기온
# ---------------------------------
st.subheader("🌡️ 연도별 예상 기온")

selected_year = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
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
        <div style="font-size: 48px; font-weight: 700;">
            {predicted_temp:.2f} °C
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if selected_year > end_year:
    st.warning(
        "실제 관측 기간 이후의 값입니다. "
        "과거 회귀 직선을 연장한 추정치이므로 실제 기온과 다를 수 있습니다."
    )


# ---------------------------------
# 6. Plotly 산점도 및 회귀 직선
# ---------------------------------
st.subheader("📊 연도별 평균기온과 회귀 직선")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7, color="#2471A3", opacity=0.8),
        customdata=annual[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        ),
    )
)

# 전체 기간 회귀 직선: 1900~2100년
line_years = np.arange(1900, 2101)
line_temps = predict_temperature(line_years)

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(color="#E74C3C", width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "전체 회귀 추정: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

# 최근 20년 회귀 직선
if np.isfinite(recent_slope):
    recent_line_years = np.arange(recent_start_year, end_year + 1)
    recent_line_temps = (
        recent_slope * recent_line_years + recent_intercept
    )

    fig.add_trace(
        go.Scatter(
            x=recent_line_years,
            y=recent_line_temps,
            mode="lines",
            name="최근 20년 회귀 직선",
            line=dict(color="#16A085", width=3, dash="dash"),
            hovertemplate=(
                "연도: %{x}년<br>"
                "최근 회귀 추정: %{y:.2f} °C"
                "<extra></extra>"
            ),
        )
    )

# 슬라이더로 선택한 연도
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

if end_year < 2100:
    fig.add_vline(
        x=end_year,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"마지막 관측 연도: {end_year}",
        annotation_position="top right",
    )

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    template="plotly_white",
    hovermode="closest",
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
# 7. 연도별 데이터 표
# ---------------------------------
with st.expander("연도별 분석 데이터 확인"):
    display_df = annual.copy()
    display_df["연평균기온"] = display_df["연평균기온"].round(2)
    display_df = display_df.rename(
        columns={
            "연평균기온": "연평균기온(°C)",
            "관측일수": "관측일수(일)",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.caption(
    "데이터 출처: 제공된 서울 기온 CSV. "
    "2025년까지의 데이터 중 유효 관측일이 300일 이상인 연도만 분석했습니다."
)
