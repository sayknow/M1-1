"""Paired circular-block bootstrap inference for contemporaneous correlations."""
from pathlib import Path
import json

from src import analyzer  # Configure Matplotlib before importing pyplot.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def statistics(a):
    centered = a - a.mean(axis=-2, keepdims=True)
    sums = (centered ** 2).sum(axis=-2)
    ex = (centered[..., 0] * centered[..., 2]).sum(axis=-1) / np.sqrt(sums[..., 0] * sums[..., 2])
    rate = (centered[..., 1] * centered[..., 2]).sum(axis=-1) / np.sqrt(sums[..., 1] * sums[..., 2])
    return np.stack([ex, rate, rate - ex], axis=-1)


def holm(p):
    order = np.argsort(p)
    result = np.empty(len(p))
    result[order] = np.minimum(1, np.maximum.accumulate(np.asarray(p)[order] * np.arange(len(p), 0, -1)))
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data/processed/economic_indicators_monthly.csv")
    data = data.loc[data.date.ge("2017-01-01"), ["date", "exchange_yoy", "base_rate_change", "cpi_yoy"]].dropna()
    a = data.iloc[:, 1:].to_numpy()
    assert len(a) == 108
    observed = statistics(a)
    assert np.allclose(observed[:2], [data.exchange_yoy.corr(data.cpi_yoy), data.base_rate_change.corr(data.cpi_yoy)])
    names = ["exchange_cpi", "rate_cpi", "rate_minus_exchange"]
    repetitions = 20000
    rows = []
    distributions = {}
    for length in [6, 12, 18, 24]:
        rng = np.random.default_rng(20261008 + length)
        starts = rng.integers(0, len(a), size=(repetitions, int(np.ceil(len(a) / length))))
        indices = ((starts[..., None] + np.arange(length)) % len(a)).reshape(repetitions, -1)[:, :len(a)]
        samples = statistics(a[indices])
        finite = np.isfinite(samples).all(axis=1)
        samples = samples[finite]
        distributions[length] = samples
        # Null-centered bootstrap approximation, not an exact randomization test.
        p = (1 + (np.abs(samples - observed) >= np.abs(observed)).sum(axis=0)) / (len(samples) + 1)
        adjusted = holm(p)
        for j, name in enumerate(names):
            low, high = np.quantile(samples[:, j], [.025, .975])
            rows.append(dict(block_months=length, statistic=name, estimate=observed[j], ci_low=low, ci_high=high,
                             approximate_p_two_sided=p[j], holm_p_three_tests=adjusted[j], valid_replicates=len(samples)))
    result = pd.DataFrame(rows)
    result.to_csv(root / "output/significance_results.csv", index=False, encoding="utf-8-sig")
    acf = {column: {str(k): float(data[column].autocorr(k)) for k in [1, 6, 12]} for column in data.columns[1:]}
    (root / "output/significance_diagnostics.json").write_text(json.dumps(acf, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for j, ax in enumerate(axes):
        part = result[result.statistic.eq(names[j])]
        for i, row in enumerate(part.itertuples()):
            ax.plot([row.ci_low, row.ci_high], [i, i], color="#176b87", lw=3)
            ax.scatter(row.estimate, i, color="#e67e22", zorder=3)
        ax.axvline(0, color="gray", ls="--")
        ax.set(title=names[j], xlabel="Estimate and percentile 95% CI", yticks=range(4), yticklabels=["6 months", "12 months", "18 months", "24 months"])
        ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(root / "output/figures/significance_intervals.png", dpi=160)
    plt.close(fig)
    lines = ["# 상관계수와 상관계수 차이의 유의성 검증", "",
             "2017-01~2025-12의 동일한 108개월을 사용한다. 분석 대상은 같은 달의 상관이다. H0는 각각 환율-CPI 상관=0, 금리-CPI 상관=0, 두 상관의 차이=0이다. 양측 5% 기준을 사용한다.",
             "월별 관측은 독립적이지 않으므로 일반 Pearson 검정의 독립 표본 가정을 그대로 적용하지 않는다. 세 변수를 같은 행 인덱스로 함께 재표집해 CPI를 공유하는 두 상관의 의존성을 보존한다.",
             "원형 블록 부트스트랩 20,000회, 고정 난수 시드를 사용했다. 연속 12개월을 주 분석 단위로 사용하고 6·18·24개월로 민감도를 확인했다. 끝과 시작을 연결하는 원형 방식이며 블록 경계에서는 연속성이 끊어진다.",
             "95% 구간은 부트스트랩 백분위 구간이다. p값은 |추정치* - 원추정치| >= |원추정치| 비율을 이용한 귀무가설 중심화 근사값이며 정확 검정이 아니다. 세 가설의 p값은 블록 길이별 Holm 보정을 제공한다. 백분위 구간과 중심화 p값은 서로 다른 근사이므로 판정이 항상 일치하지 않을 수 있다.", "",
             "| 블록(개월) | 검증 대상 | 추정치 | 95% 구간 | 근사 p | Holm 보정 p |", "|---|---|---:|---|---:|---:|"]
    for r in result.itertuples():
        lines.append(f"| {r.block_months} | {r.statistic} | {r.estimate:.4f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] | {r.approximate_p_two_sided:.4f} | {r.holm_p_three_tests:.4f} |")
    lines += ["", "## 해석 범위",
              "rate_minus_exchange는 금리-CPI 상관에서 환율-CPI 상관을 뺀 값이다. 한 상관이 유의하고 다른 상관이 유의하지 않다는 사실만으로 두 상관의 차이가 유의하다고 말할 수 없으므로 차이를 직접 검증했다.",
              "블록 길이는 결과를 보기 전에 12개월을 주 기준으로 정했으나 최적 길이를 추정한 것은 아니다. 108개월에서 24개월 블록은 약 4.5개 분량밖에 없으므로 불확실성이 크다. 부트스트랩은 대략적인 정상성과 약한 의존을 전제로 하며 경제 국면 변화가 있으면 신뢰구간의 정확성이 제한된다.",
              "같은 달의 분석은 이전 0~12개월 탐색 결과를 본 뒤 선택됐다. 따라서 탐색적 추가 검증이며, Holm 보정은 여기의 세 가설에만 적용되어 이전 모든 시차 탐색을 보정하지 않는다. 시차별 유의성·정책의 인과효과·예측력은 검증하지 않았다.",
              "p값은 귀무가설이 참일 확률이 아니다. 유의하지 않다는 결과도 상관이나 차이가 없다는 증거로 단정할 수 없다.",
              "", "## 재실행과 검증", "`python -m src.significance`", "상관계수는 pandas 계산과 대조했고 모든 재표집에서 동일 행 인덱스를 공유한다. 비유한 추정치는 제외하고 유효 반복 수를 결과표에 기록한다.",
              "", "## 방법 참고", "- [CMU: 시계열 블록 부트스트랩](https://stat.cmu.edu/~cshalizi/uADA/16/lectures/26.pdf)", "- [SciPy: Pearson 검정의 가정](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)"]
    (root / "output/significance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result.to_string(index=False))
    print("Autocorrelations:", acf)


if __name__ == "__main__":
    main()
