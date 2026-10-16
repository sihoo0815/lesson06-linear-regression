# ---------------------------------
# 선형회귀 및 상관계수
# ---------------------------------
x = annual["연도"].to_numpy(dtype=float)
y = annual["연평균기온"].to_numpy(dtype=float)

# 전체 기간 회귀
slope, intercept = np.polyfit(x, y, 1)
correlation = np.corrcoef(x, y)[0, 1]

start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
year_count = len(annual)

# 최근 20년: 마지막 관측 연도를 포함해 20년 범위 설정
recent_start_year = end_year - 19
recent = annual[annual["연도"] >= recent_start_year].copy()

if len(recent) >= 2:
    recent_slope, recent_intercept = np.polyfit(
        recent["연도"].to_numpy(dtype=float),
        recent["연평균기온"].to_numpy(dtype=float),
        1,
    )
    recent_correlation = np.corrcoef(
        recent["연도"].to_numpy(dtype=float),
        recent["연평균기온"].to_numpy(dtype=float),
    )[0, 1]
else:
    recent_slope = np.nan
    recent_intercept = np.nan
    recent_correlation = np.nan


def predict_temperature(year):
    """전체 기간 회귀 직선을 이용한 예상 연평균기온."""
    return slope * year + intercept


# 기울기를 100년당 기온 변화량으로 환산
warming_per_100_years = slope * 100
recent_warming_per_100_years = recent_slope * 100
