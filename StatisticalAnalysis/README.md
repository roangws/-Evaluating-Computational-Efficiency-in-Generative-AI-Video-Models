# Statistical Analysis — Generative AI Video Models

This module performs comprehensive statistical analysis comparing **5 generative AI video models** across **5 performance metrics**.

## Models Analyzed

| Model | Source | Samples |
|-------|--------|---------|
| CogVideoX-2b | Local GPU | 15 |
| CogVideoX-5b | Local GPU | 15 |
| CogVideoX1.5-5B | Local GPU | 15 |
| Google Veo 3.1 | API | 9 |
| HunyuanVideo-1.5 | Local GPU | 9 |

## Metrics Evaluated

1. **Inference Time** (seconds) — Generation latency
2. **Peak Memory** (GB) — GPU memory usage
3. **Compute Cost** (USD) — Estimated cost per generation
4. **CLIP Score** — Text-to-video alignment quality
5. **Frame Consistency** — Temporal coherence between frames

---

## Quick Start

```bash
cd StatisticalAnalysis
source venv/bin/activate
python statistical_analysis.py
```

Results are saved to `output/`.

---

## Key Findings

### Descriptive Statistics (Mean ± SD)

| Model | Inference Time (s) | Cost (USD) | CLIP Score | Frame Consistency |
|-------|-------------------|------------|------------|-------------------|
| CogVideoX-2b | 257.90 ± 0.43 | 0.085 ± 0.00 | 30.90 ± 1.91 | 0.54 ± 0.19 |
| CogVideoX-5b | 611.11 ± 1.76 | 0.200 ± 0.00 | 32.20 ± 2.42 | 0.56 ± 0.22 |
| CogVideoX1.5-5B | 1762.35 ± 1.67 | 0.578 ± 0.00 | 32.40 ± 2.85 | 0.42 ± 0.14 |
| Google Veo 3.1 | 61.73 ± 8.58 | 0.750 ± 0.00 | 28.34 ± 3.76 | 0.26 ± 0.10 |
| HunyuanVideo-1.5 | 696.23 ± 0.42 | 1.454 ± 0.00 | 28.98 ± 3.79 | 0.26 ± 0.16 |

### ANOVA Results

All metrics showed **statistically significant differences** across models:

| Metric | F-statistic | p-value | Significance |
|--------|-------------|---------|--------------|
| Inference Time | 506,573.85 | < 0.001 | *** |
| Peak Memory | ∞ | < 0.001 | *** |
| Compute Cost | 12,160,029.47 | < 0.001 | *** |
| CLIP Score | 4.61 | 0.003 | ** |
| Frame Consistency | 8.08 | < 0.001 | *** |

### Tukey HSD Post-Hoc Highlights

**CLIP Score significant differences:**
- CogVideoX-5b > Google Veo 3.1 (p = 0.018)
- CogVideoX1.5-5B > Google Veo 3.1 (p = 0.012)
- CogVideoX1.5-5B > HunyuanVideo-1.5 (p = 0.048)

**Frame Consistency:** CogVideoX models (2b, 5b) significantly outperformed both Google Veo 3.1 and HunyuanVideo-1.5.

### Cost-Efficiency Ranking

| Rank | Model | Efficiency Score |
|------|-------|------------------|
| 🥇 1 | CogVideoX-2b | 198.26 |
| 🥈 2 | CogVideoX-5b | 90.45 |
| 🥉 3 | CogVideoX1.5-5B | 23.76 |
| 4 | Google Veo 3.1 | 9.92 |
| 5 | HunyuanVideo-1.5 | 5.21 |

*Efficiency Score = (CLIP Score × Frame Consistency) / Compute Cost*

---

## Output Files

| File | Description |
|------|-------------|
| `descriptive_stats.csv` | Mean ± SD for all metrics by model |
| `anova_results.csv` | F-statistics, p-values, degrees of freedom |
| `tukey_posthoc.csv` | Pairwise comparisons (50 pairs) |
| `cohens_d_matrix.csv` | Effect sizes with interpretations |
| `cost_efficiency_ranking.csv` | Composite rankings |

---

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
statsmodels>=0.14.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Methods

- **Descriptive Statistics:** Sample mean and standard deviation per model
- **ANOVA:** One-way analysis of variance (α = 0.05)
- **Post-Hoc:** Tukey's Honest Significant Difference with family-wise error correction
- **Effect Size:** Cohen's d with pooled standard deviation
- **Cost-Efficiency:** Composite quality/cost ratio

## Limitations

- Small sample sizes (n = 9–15 per model)
- Google Veo 3.1 memory reported as 0 (API-based model)
- Cost estimates based on compute time × GPU hourly rate
