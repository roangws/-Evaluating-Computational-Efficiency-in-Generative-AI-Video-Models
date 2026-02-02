# Statistical Analysis — Generative AI Video Models

This module performs comprehensive statistical analysis comparing **5 generative AI video models** across **5 performance metrics** and includes a **user perception study** with 32 participants evaluating subjective quality dimensions.

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

Most metrics showed **statistically significant differences** across models:

| Metric | F-statistic | p-value | Significance | Notes |
|--------|-------------|---------|--------------|-------|
| Inference Time | 506,573.85 | < 0.001 | *** | |
| Peak Memory | — | — | degenerate | No within-group variance; deterministic differences reported directly |
| Compute Cost | 12,160,029.47 | < 0.001 | *** | |
| CLIP Score | 4.61 | 0.003 | ** | |
| Frame Consistency | 8.08 | < 0.001 | *** | |

**Note on Peak Memory:** ANOVA is not applicable because memory values show zero within-group variance (each model has deterministic memory usage). For open-source models, differences are reported directly in descriptive statistics.

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

## � Visualizations

### Figure 1: Quality Metrics Comparison

![Quality Metrics Comparison](Visualization/Quality%20metrics%20comparison.png)

**Figure 1.** Quality metrics comparison: CLIP score (blue) and frame consistency (orange), normalized 0-100 scale. CogVideoX-2b and CogVideoX-5b achieve balanced performance across both dimensions.

This visualization decomposes the quality metrics underlying cost-efficiency calculations, revealing the dual-dimensional nature of video generation quality assessment.

### Figure 2: Inference Time Distributions

![Inference Time Distributions](Visualization/Inference%20time%20distributions.png)

**Figure 2.** Inference time distributions (violin plots). Google Veo 3.1 shows highest variability (σ=8.58s) due to API latency. Local models demonstrate consistent performance (σ<2s).

This provides a distributional perspective on inference time variability, complementing the mean values presented in the descriptive statistics.

### Figure 3: Cost-Quality Tradeoff with Pareto Frontier

![Cost-Quality Tradeoff](Visualization/Cost-quality%20tradeoff%20with%20Pareto%20frontie.png)

**Figure 3.** Cost-quality tradeoff with Pareto frontier. CogVideoX models on frontier offer optimal efficiency. Bubble size represents video count (n=9 or n=15).

To examine the quality-cost relationship underlying these efficiency scores, this plot shows CLIP score against computational cost, with a Pareto frontier identifying models that maximize quality per dollar spent.

### Figure 4: Cost-Efficiency Rankings

![Cost-Efficiency Rankings](Visualization/Cost-efficiency%20rankings%20using%20composite%20metric.png)

**Figure 4.** Cost-efficiency rankings using composite metric (CLIP Score × Frame Consistency / Cost). CogVideoX-2b achieves 198.26, representing 20× higher efficiency than Google Veo 3.1.

The cost-efficiency rankings employ a composite metric combining CLIP score, frame consistency, and computational cost to identify optimal models for budget-constrained deployments.

---

## �👥 User Perception Study

### Study Overview

A comprehensive **user research study** with **32 participants** evaluated all 5 models across 10 subjective quality dimensions plus preference rankings.

**Key Demographics:**
- **Age**: 18-55 years (diverse age range)
- **Technical Background**: 47% Technical Professionals, 34% Some Technical Knowledge, 13% No Technical Background, 6% AI/ML Specialists
- **Education**: 53% Bachelor's, 28% Master's, 16% PhD, 3% High School

### User Preference Rankings

| Rank | Model | Mean Rank | First Place Votes | First Place % |
|------|-------|-----------|-------------------|---------------|
| 🥇 1 | **Google Veo 3.1** | 1.81 ± 0.97 | 14 | 43.8% |
| 🥈 2 | **HunyuanVideo-1.5** | 1.94 ± 1.22 | 14 | 43.8% |
| 🥉 3 | **CogVideoX1.5-5B** | 3.50 ± 1.02 | 1 | 3.1% |
| 4 | **CogVideoX-5b** | 3.62 ± 1.04 | 1 | 3.1% |
| 5 | **CogVideoX-2b** | 3.97 ± 1.06 | 2 | 6.2% |

### Top User Perception Metrics (5-point scale)

**Overall Satisfaction:**
- Google Veo 3.1: **4.38 ± 0.55** ⭐⭐⭐⭐
- HunyuanVideo-1.5: **4.25 ± 0.76** ⭐⭐⭐⭐
- CogVideoX models: 2.81-2.94 (Moderate)

**Realism:**
- HunyuanVideo-1.5: **4.72 ± 0.46** ⭐⭐⭐⭐⭐ (Highest)
- Google Veo 3.1: **4.31 ± 0.78** ⭐⭐⭐⭐
- CogVideoX models: 2.62-3.09 (Moderate)

**Ease of Use:**
- Google Veo 3.1: **4.66 ± 0.48** ⭐⭐⭐⭐⭐ (Dominant)
- CogVideoX models: 2.06-2.94 (Fair to Moderate)
- HunyuanVideo-1.5: 2.69 (Moderate)

**Artifacts Severity (lower is better):**
- HunyuanVideo-1.5: **1.72 ± 0.73** ✅ Minimal
- Google Veo 3.1: **1.72 ± 0.63** ✅ Minimal
- CogVideoX1.5-5B: **3.66 ± 0.65** ❌ Noticeable

### Critical Insights

**1. User Preference ≠ Technical Performance**
- **User preference leaders**: Google Veo 3.1 & HunyuanVideo-1.5
- **Technical performance leaders**: CogVideoX-5b & CogVideoX1.5-5B (CLIP scores)
- **Implication**: Users prioritize perceptual quality (realism, ease of use) over algorithmic metrics

**2. Ease of Use is a Major Differentiator**
- Google Veo 3.1's ease of use (4.66) significantly outperforms all local models (2.06-2.94)
- API-based solutions have strong UX advantage over local GPU deployment

**3. Realism Drives Professional Adoption**
- HunyuanVideo-1.5 leads in realism (4.72) and professional suitability (4.31)
- Photorealistic output critical for user acceptance

**4. Artifact Intolerance**
- CogVideoX1.5-5B's high artifact severity (3.66) correlates with poor professional suitability (2.03)
- Users highly sensitive to visual artifacts

**5. Willingness to Pay Aligns with Satisfaction**
- Google Veo 3.1: 3.91 (highest)
- HunyuanVideo-1.5: 3.31
- CogVideoX models: 2.03-2.50

📁 **Detailed user study findings**: See [`UserResearch/README.md`](UserResearch/README.md)

---

## Output Files

### Technical Performance Analysis

| File | Description |
|------|-------------|
| `descriptive_stats.csv` | Mean ± SD for all metrics by model |
| `anova_results.csv` | F-statistics, p-values, degrees of freedom |
| `tukey_posthoc.csv` | Pairwise comparisons (50 pairs) |
| `cohens_d_matrix.csv` | Effect sizes with interpretations |
| `cost_efficiency_ranking.csv` | Composite rankings |

### User Research Study

| File | Description |
|------|-------------|
| `UserResearch/user_perception_study.csv` | Raw participant responses (32 participants × 55 metrics) |
| `UserResearch/user_study_descriptive_stats.csv` | Mean ± SD for all perception metrics by model |
| `UserResearch/user_study_preference_rankings.csv` | Overall preference rankings and first-place votes |
| `UserResearch/README.md` | Comprehensive user study insights and analysis |

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
  - Degenerate cases (zero within-group variance) reported separately
- **Post-Hoc:** Tukey's Honest Significant Difference with family-wise error correction
- **Effect Size:** 
  - Cohen's d for metrics with meaningful variance (CLIP Score, Frame Consistency)
  - Mean differences for near-deterministic metrics (time, cost)
- **Cost-Efficiency:** Composite quality/cost ratio
- **Missing Data Handling:** API-based models (Veo) report memory as N/A (not observable)

## Statistical Considerations

### Peak Memory Analysis
Peak memory ANOVA shows F→∞ (degenerate) because:
- **Zero within-group variance:** Each model has deterministic memory usage
- **Google Veo 3.1:** API-based; memory not observable (reported as N/A)
- **Open-source models:** Memory comparisons valid only among CogVideoX and HunyuanVideo

**Recommendation:** Report deterministic memory differences directly rather than ANOVA results.

### Cohen's d Effect Sizes
For metrics with **near-zero variance** (inference time, compute cost):
- Cohen's d values can be unrealistically large (>100)
- **Solution:** Report **mean differences** instead of Cohen's d for these metrics
- **Cohen's d retained for:** CLIP Score and Frame Consistency (meaningful variance)

### Sample Sizes
- CogVideoX models: n = 15 per model
- Google Veo 3.1: n = 9
- HunyuanVideo-1.5: n = 9

### Cost Estimates
- Based on compute time × GPU hourly rate ($7.52/hour for A100)
- Google Veo 3.1: Fixed API pricing
