#!/usr/bin/env python3
"""
Statistical Analysis for Generative AI Video Models
Calculates descriptive statistics, ANOVA, Tukey HSD, Cohen's d, and cost-efficiency rankings.
"""

import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings('ignore')

# Paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

CSV_PATHS = [
    os.path.join(BASE_DIR, "HunyuanVideo", "results_hunyuan.csv"),
    os.path.join(BASE_DIR, "GoogleVeo", "results_veo.csv"),
    os.path.join(BASE_DIR, "CogVideoX", "results.csv"),
]

METRICS = [
    "inference_time_seconds",
    "peak_memory_gb",
    "compute_cost_usd",
    "clip_score",
    "frame_consistency",
]

MODEL_NAME_MAP = {
    "THUDM/CogVideoX-5b": "CogVideoX-5b",
    "THUDM/CogVideoX-2b": "CogVideoX-2b",
    "THUDM/CogVideoX1.5-5B": "CogVideoX1.5-5B",
    "HunyuanVideo-1.5": "HunyuanVideo-1.5",
    "Google Veo 3.1": "Google Veo 3.1",
}


def load_and_merge_data() -> pd.DataFrame:
    """Load all CSV files and merge into a single DataFrame."""
    dfs = []
    for path in CSV_PATHS:
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"Loaded: {os.path.basename(path)} ({len(df)} rows)")
        else:
            print(f"Warning: {path} not found")
    
    merged = pd.concat(dfs, ignore_index=True)
    merged["model_name"] = merged["model_name"].map(MODEL_NAME_MAP).fillna(merged["model_name"])
    
    # Replace 0.0 memory with NaN for API-based models (Veo)
    merged.loc[merged["peak_memory_gb"] == 0.0, "peak_memory_gb"] = np.nan
    
    print(f"\nTotal samples: {len(merged)}")
    print(f"Models: {merged['model_name'].unique().tolist()}")
    return merged


def calculate_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean ± SD for each metric grouped by model."""
    results = []
    
    for model in df["model_name"].unique():
        model_data = df[df["model_name"] == model]
        row = {"model": model, "n": len(model_data)}
        
        for metric in METRICS:
            mean = model_data[metric].mean()
            std = model_data[metric].std()
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = std
            row[f"{metric}_formatted"] = f"{mean:.4f} ± {std:.4f}"
        
        results.append(row)
    
    return pd.DataFrame(results)


def run_anova_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run one-way ANOVA for each metric across all models."""
    results = []
    models = df["model_name"].unique()
    
    for metric in METRICS:
        # Skip ANOVA for peak_memory_gb if it has no variance within groups
        if metric == "peak_memory_gb":
            # Check if there's within-group variance
            has_variance = False
            for m in models:
                group_data = df[df["model_name"] == m][metric].dropna()
                if len(group_data) > 1 and group_data.std() > 1e-10:
                    has_variance = True
                    break
            
            if not has_variance:
                results.append({
                    "metric": metric,
                    "f_statistic": np.nan,
                    "p_value": np.nan,
                    "df_between": len(models) - 1,
                    "df_within": len(df) - len(models),
                    "significance": "degenerate",
                    "note": "No within-group variance; deterministic differences"
                })
                continue
        
        groups = [df[df["model_name"] == m][metric].dropna().values for m in models]
        groups = [g for g in groups if len(g) > 0]  # Remove empty groups
        
        if len(groups) < 2:
            results.append({
                "metric": metric,
                "f_statistic": np.nan,
                "p_value": np.nan,
                "df_between": np.nan,
                "df_within": np.nan,
                "significance": "insufficient data",
                "note": "Not enough groups for comparison"
            })
            continue
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Calculate degrees of freedom
        df_between = len(groups) - 1
        df_within = sum(len(g) for g in groups) - len(groups)
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        results.append({
            "metric": metric,
            "f_statistic": f_stat,
            "p_value": p_value,
            "df_between": df_between,
            "df_within": df_within,
            "significance": significance,
            "note": ""
        })
    
    return pd.DataFrame(results)


def tukey_hsd_posthoc(df: pd.DataFrame) -> pd.DataFrame:
    """Perform Tukey HSD post-hoc comparisons for each metric."""
    all_results = []
    
    for metric in METRICS:
        tukey = pairwise_tukeyhsd(
            endog=df[metric],
            groups=df["model_name"],
            alpha=0.05
        )
        
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], 
                                 columns=tukey._results_table.data[0])
        tukey_df["metric"] = metric
        tukey_df["significant"] = tukey_df["reject"].apply(lambda x: "Yes" if x else "No")
        all_results.append(tukey_df)
    
    return pd.concat(all_results, ignore_index=True)


def calculate_cohens_d(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Cohen's d effect size for all model pairs."""
    models = df["model_name"].unique()
    results = []
    
    # Metrics where Cohen's d is meaningful (those with reasonable variance)
    meaningful_metrics = ["clip_score", "frame_consistency"]
    
    for metric in METRICS:
        for m1, m2 in combinations(models, 2):
            g1 = df[df["model_name"] == m1][metric].dropna().values
            g2 = df[df["model_name"] == m2][metric].dropna().values
            
            if len(g1) == 0 or len(g2) == 0:
                results.append({
                    "metric": metric,
                    "model_1": m1,
                    "model_2": m2,
                    "cohens_d": np.nan,
                    "abs_cohens_d": np.nan,
                    "interpretation": "insufficient data",
                    "mean_diff": np.nan,
                })
                continue
            
            # Pooled standard deviation
            n1, n2 = len(g1), len(g2)
            var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            
            mean_diff = g1.mean() - g2.mean()
            
            # For metrics with very low variance, report mean difference instead of Cohen's d
            if metric not in meaningful_metrics:
                if pooled_std < 1e-6 or (abs(mean_diff) / pooled_std) > 100:
                    results.append({
                        "metric": metric,
                        "model_1": m1,
                        "model_2": m2,
                        "cohens_d": "not applicable",
                        "abs_cohens_d": "not applicable",
                        "interpretation": "near-zero variance",
                        "mean_diff": mean_diff,
                    })
                    continue
            
            # Cohen's d
            d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            # Interpretation
            abs_d = abs(d)
            if abs_d < 0.2:
                interpretation = "negligible"
            elif abs_d < 0.5:
                interpretation = "small"
            elif abs_d < 0.8:
                interpretation = "medium"
            else:
                interpretation = "large"
            
            results.append({
                "metric": metric,
                "model_1": m1,
                "model_2": m2,
                "cohens_d": d,
                "abs_cohens_d": abs_d,
                "interpretation": interpretation,
                "mean_diff": mean_diff,
            })
    
    return pd.DataFrame(results)


def generate_cost_efficiency_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Generate cost-efficiency rankings based on quality/cost ratio."""
    stats_df = calculate_descriptive_stats(df)
    
    rankings = []
    for _, row in stats_df.iterrows():
        model = row["model"]
        
        # Quality metrics (higher is better)
        clip = row["clip_score_mean"]
        consistency = row["frame_consistency_mean"]
        
        # Cost metrics (lower is better)
        time = row["inference_time_seconds_mean"]
        memory = row["peak_memory_gb_mean"]
        cost = row["compute_cost_usd_mean"]
        
        # Composite scores
        quality_score = clip * consistency
        
        # Cost efficiency: quality per dollar
        cost_efficiency = quality_score / cost if cost > 0 else 0
        
        # Time efficiency: quality per second
        time_efficiency = quality_score / time if time > 0 else 0
        
        # Memory efficiency: quality per GB (handle 0 memory for API models)
        memory_efficiency = quality_score / memory if memory > 0 else float('inf')
        
        rankings.append({
            "model": model,
            "clip_score": clip,
            "frame_consistency": consistency,
            "quality_score": quality_score,
            "compute_cost_usd": cost,
            "inference_time_s": time,
            "peak_memory_gb": memory,
            "cost_efficiency": cost_efficiency,
            "time_efficiency": time_efficiency,
            "memory_efficiency": memory_efficiency if memory > 0 else "N/A (API)",
        })
    
    ranking_df = pd.DataFrame(rankings)
    
    # Add ranks
    ranking_df["cost_efficiency_rank"] = ranking_df["cost_efficiency"].rank(ascending=False).astype(int)
    ranking_df["time_efficiency_rank"] = ranking_df["time_efficiency"].rank(ascending=False).astype(int)
    
    return ranking_df.sort_values("cost_efficiency_rank")


def print_summary(desc_stats: pd.DataFrame, anova: pd.DataFrame, ranking: pd.DataFrame):
    """Print a summary of key findings."""
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("\n## Descriptive Statistics (Mean ± SD)")
    print("-" * 50)
    for _, row in desc_stats.iterrows():
        print(f"\n{row['model']} (n={row['n']}):")
        for metric in METRICS:
            print(f"  {metric}: {row[f'{metric}_formatted']}")
    
    print("\n## ANOVA Results")
    print("-" * 50)
    for _, row in anova.iterrows():
        if pd.isna(row['f_statistic']):
            print(f"{row['metric']}: {row['significance']} - {row.get('note', '')}")
        else:
            print(f"{row['metric']}: F({row['df_between']},{row['df_within']}) = {row['f_statistic']:.3f}, "
                  f"p = {row['p_value']:.6f} {row['significance']}")
    
    print("\n## Cost-Efficiency Ranking")
    print("-" * 50)
    for _, row in ranking.iterrows():
        print(f"#{row['cost_efficiency_rank']}: {row['model']} "
              f"(efficiency score: {row['cost_efficiency']:.4f})")


def main():
    """Main execution function."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading and merging data...")
    df = load_and_merge_data()
    
    print("\nCalculating descriptive statistics...")
    desc_stats = calculate_descriptive_stats(df)
    desc_stats.to_csv(os.path.join(OUTPUT_DIR, "descriptive_stats.csv"), index=False)
    
    print("Running ANOVA tests...")
    anova_results = run_anova_tests(df)
    anova_results.to_csv(os.path.join(OUTPUT_DIR, "anova_results.csv"), index=False)
    
    print("Performing Tukey HSD post-hoc comparisons...")
    tukey_results = tukey_hsd_posthoc(df)
    tukey_results.to_csv(os.path.join(OUTPUT_DIR, "tukey_posthoc.csv"), index=False)
    
    print("Calculating Cohen's d effect sizes...")
    cohens_d = calculate_cohens_d(df)
    cohens_d.to_csv(os.path.join(OUTPUT_DIR, "cohens_d_matrix.csv"), index=False)
    
    print("Generating cost-efficiency rankings...")
    ranking = generate_cost_efficiency_ranking(df)
    ranking.to_csv(os.path.join(OUTPUT_DIR, "cost_efficiency_ranking.csv"), index=False)
    
    # Print summary
    print_summary(desc_stats, anova_results, ranking)
    
    print("\n" + "=" * 70)
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
