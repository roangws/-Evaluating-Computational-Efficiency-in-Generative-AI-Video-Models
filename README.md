# Cost-Efficiency Metrics: Evaluating Computational and Resource Efficiency in Generative AI Video Models

**Author**: Roan Guilherme Weigert Salgueiro  
**Date**: January 2026

A comprehensive benchmark study comparing **5 generative AI video models** across local GPU-based (CogVideoX, HunyuanVideo) and cloud API-based (Google Veo 3.1) generation, with rigorous statistical analysis measuring cost, speed, quality, and scalability.

---

## 🎯 Research Objective

Provide data-driven insights for choosing between local and cloud-based AI video generation, evaluating:
- **Cost efficiency**: GPU compute costs vs API pricing
- **Performance**: Inference speed and throughput
- **Quality**: CLIP score (text-video alignment) and frame consistency
- **Scalability**: Hardware requirements vs cloud availability
- **Statistical significance**: ANOVA, Tukey HSD, Cohen's d effect sizes

---

## 📊 Key Research Findings

### Comprehensive Model Comparison (5 Models)

| Model | Avg Time (s) | Cost/Video | CLIP Score | Frame Consistency | Memory (GB) | Source |
|-------|-------------|------------|------------|-------------------|-------------|--------|
| **CogVideoX-2b** | 257.90 ± 0.43 | $0.085 | 30.90 ± 1.91 | 0.54 ± 0.19 | 12.24 | Local GPU |
| **CogVideoX-5b** | 611.11 ± 1.76 | $0.200 | 32.20 ± 2.42 | 0.56 ± 0.22 | 12.00 | Local GPU |
| **CogVideoX1.5-5B** | 1762.35 ± 1.67 | $0.578 | 32.40 ± 2.85 | 0.42 ± 0.14 | 12.49 | Local GPU |
| **Google Veo 3.1** | 61.73 ± 8.58 | $0.750 | 28.34 ± 3.76 | 0.26 ± 0.10 | 0.00 | Cloud API |
| **HunyuanVideo-1.5** | 696.23 ± 0.42 | $1.454 | 28.98 ± 3.79 | 0.26 ± 0.16 | 39.08 | Local GPU |

### 🏆 Winner by Category

| Category | Winner | Reason |
|----------|--------|--------|
| **⚡ Fastest** | Google Veo 3.1 | 61.73s (4.2× faster than fastest local) |
| **💰 Cheapest** | CogVideoX-2b | $0.085/video (17× cheaper than HunyuanVideo) |
| **🎨 Best Quality** | CogVideoX1.5-5B | CLIP 32.40, statistically significant |
| **🏅 Most Consistent** | CogVideoX-5b | Frame consistency 0.56 |
| **🚀 Easiest Setup** | Google Veo 3.1 | No GPU required, just API key |
| **📈 Best Cost-Efficiency** | CogVideoX-2b | Efficiency score: 198.26 |

### 📊 Statistical Significance (ANOVA)

All metrics showed **statistically significant differences** across models (p < 0.01):

| Metric | F-statistic | p-value | Significance |
|--------|-------------|---------|--------------|
| Inference Time | 506,573.85 | < 0.001 | *** |
| Compute Cost | 12,160,029.47 | < 0.001 | *** |
| CLIP Score | 4.61 | 0.003 | ** |
| Frame Consistency | 8.08 | < 0.001 | *** |

**Key Post-Hoc Findings (Tukey HSD)**:
- CogVideoX-5b significantly outperformed Google Veo 3.1 in CLIP score (p = 0.018)
- CogVideoX1.5-5B significantly outperformed HunyuanVideo-1.5 (p = 0.048)
- CogVideoX models showed significantly better frame consistency than cloud/HunyuanVideo

---

## 🔬 Experiments Conducted

### Experiment 1: CogVideoX (Local GPU)
- **Models Tested**: CogVideoX-2b, CogVideoX-5b, CogVideoX1.5-5B
- **Videos Generated**: 45 (3 models × 3 prompts × 5 runs)
- **Success Rate**: 100% (45/45)
- **Total Cost**: $12.93
- **Hardware**: NVIDIA GPU with 12GB+ VRAM

### Experiment 2: Google Veo 3.1 (Cloud API)
- **Model Tested**: veo-3.1-generate-preview
- **Videos Generated**: 9 (1 model × 3 prompts × 3 runs)
- **Success Rate**: 100% (9/9)
- **Total Cost**: $6.75
- **Hardware**: None (cloud-based)

### Experiment 3: HunyuanVideo (Local GPU)
- **Model Tested**: HunyuanVideo-1.5 (720p text-to-video)
- **Videos Generated**: 9 (1 model × 3 prompts × 3 runs)
- **Success Rate**: 100% (9/9)
- **Total Cost**: $13.09
- **Hardware**: NVIDIA A100-SXM4-80GB (39GB VRAM required)

### Experiment 4: Statistical Analysis
- **Models Analyzed**: All 5 models (63 total videos)
- **Methods**: ANOVA, Tukey HSD, Cohen's d effect sizes
- **Outputs**: Descriptive stats, pairwise comparisons, cost-efficiency rankings
- **Tools**: Python (pandas, scipy, statsmodels)

### Standardized Test Prompts
1. "A person walking in a park on a sunny day"
2. "A car driving on a highway with trees in background"
3. "A cat playing with a ball in a living room"

---

## 🎬 Sample Results

### CogVideoX-5b (Local GPU)
<table>
  <tr>
    <td align="center">
      <a href="CogVideoX/output_videos/CogVideoX-5b/prompt_1/run_1.mp4">
        <img src="CogVideoX/output_frames/CogVideoX-5b/prompt_1/run_1_middle_frame.jpg" width="200px"/>
      </a><br/>
      <b>Person in Park</b><br/>
      CLIP: 33.43 | 617s
    </td>
    <td align="center">
      <a href="CogVideoX/output_videos/CogVideoX-5b/prompt_2/run_1.mp4">
        <img src="CogVideoX/output_frames/CogVideoX-5b/prompt_2/run_1_middle_frame.jpg" width="200px"/>
      </a><br/>
      <b>Car on Highway</b><br/>
      CLIP: 28.74 | 610s
    </td>
    <td align="center">
      <a href="CogVideoX/output_videos/CogVideoX-5b/prompt_3/run_1.mp4">
        <img src="CogVideoX/output_frames/CogVideoX-5b/prompt_3/run_1_middle_frame.jpg" width="200px"/>
      </a><br/>
      <b>Cat with Ball</b><br/>
      CLIP: 30.97 | 611s
    </td>
  </tr>
</table>

### Google Veo 3.1 (Cloud API)
<table>
  <tr>
    <td align="center">
      <a href="GoogleVeo/output_videos/veo_prompt_1/veo_prompt_1_run_1.mp4">
        <img src="GoogleVeo/output_frames/veo_prompt_1/run_1_middle_frame.jpg" width="200px"/>
      </a><br/>
      <b>Person in Park</b><br/>
      CLIP: 26.08 | Consistency: 0.409 | 74s
    </td>
    <td align="center">
      <a href="GoogleVeo/output_videos/veo_prompt_2/veo_prompt_2_run_1.mp4">
        <img src="GoogleVeo/output_frames/veo_prompt_2/run_1_middle_frame.jpg" width="200px"/>
      </a><br/>
      <b>Car on Highway</b><br/>
      CLIP: 24.13 | Consistency: 0.334 | 64s
    </td>
    <td align="center">
      <a href="GoogleVeo/output_videos/veo_prompt_3/veo_prompt_3_run_1.mp4">
        <img src="GoogleVeo/output_frames/veo_prompt_3/run_1_middle_frame.jpg" width="200px"/>
      </a><br/>
      <b>Cat with Ball</b><br/>
      CLIP: 32.81 | Consistency: 0.187 | 54s
    </td>
  </tr>
</table>

> Click any image to view the full video

---

## 💡 Recommendations

### Choose **Local GPU (CogVideoX)** when:
- ✅ Processing large batches (100+ videos)
- ✅ Cost is primary concern ($0.08-0.20 vs $0.75)
- ✅ Quality metrics (CLIP score) are critical
- ✅ You have GPU hardware available
- ✅ Data privacy is important (no cloud upload)

### Choose **Cloud API (Google Veo)** when:
- ✅ Speed is critical (4-10× faster)
- ✅ No GPU hardware available
- ✅ Quick prototyping or small batches
- ✅ Easier deployment (just API key)
- ✅ Scalability without hardware limits

### Cost Break-Even Analysis
```
Break-even point: ~100 videos
- Under 100 videos: Cloud API may be more practical (no setup)
- Over 100 videos: Local GPU becomes significantly cheaper
- 1000 videos: Local saves ~$670 ($80 vs $750)
```

---

## 📁 Project Structure

```
├── README.md                 # This file (project overview)
├── CogVideoX/                # Local GPU experiments (3 models)
│   ├── README.md             # Detailed CogVideoX documentation
│   ├── experiment.py         # Benchmarking script
│   ├── results.csv           # 45 video results
│   ├── output_videos/        # Generated videos (45)
│   └── output_frames/        # Preview frames (27)
├── GoogleVeo/                # Cloud API experiments
│   ├── README.md             # Detailed Veo documentation
│   ├── experiment_veo.py     # API integration script
│   ├── results_veo.csv       # 9 video results
│   ├── output_videos/        # Generated videos (9)
│   └── output_frames/        # Preview frames (9)
├── HunyuanVideo/             # Local GPU experiments (HunyuanVideo-1.5)
│   ├── README.md             # Detailed HunyuanVideo documentation
│   ├── experiment_hunyuan.py # Benchmarking script
│   ├── results_hunyuan.csv   # 9 video results
│   ├── output_videos/        # Generated videos (9)
│   └── output_frames/        # Preview frames (9)
└── StatisticalAnalysis/      # Comprehensive statistical analysis
    ├── README.md             # Analysis documentation
    ├── statistical_analysis.py # Main analysis script
    ├── requirements.txt      # Python dependencies
    └── output/               # Analysis results
        ├── descriptive_stats.csv
        ├── anova_results.csv
        ├── tukey_posthoc.csv
        ├── cohens_d_matrix.csv
        └── cost_efficiency_ranking.csv
```

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Local Models** | CogVideoX (2B, 5B, 1.5-5B), HunyuanVideo-1.5, PyTorch, Diffusers, CUDA |
| **Cloud API** | Google Gemini API, Veo 3.1, google-genai |
| **Quality Metrics** | CLIP (OpenAI), Frame Consistency Analysis |
| **Statistical Analysis** | Pandas, NumPy, SciPy, Statsmodels (ANOVA, Tukey HSD) |
| **Video Processing** | OpenCV, FFmpeg |
| **Language** | Python 3.8+ |

---

## 📈 Detailed Results

### All Models Comparison

| Model | Parameters | Avg Time | CLIP Score | Consistency | Cost/Video | Memory |
|-------|------------|----------|------------|-------------|------------|--------|
| CogVideoX-2b | 2B | 257.90s | 30.90 ± 1.91 | 0.542 ± 0.193 | $0.085 | 12.24GB |
| CogVideoX-5b | 5B | 611.11s | 32.20 ± 2.42 | 0.563 ± 0.218 | $0.200 | 12.00GB |
| CogVideoX1.5-5B | 5B+ | 1762.35s | 32.40 ± 2.85 | 0.424 ± 0.135 | $0.578 | 12.49GB |
| Google Veo 3.1 | N/A | 61.73s | 28.34 ± 3.76 | 0.262 ± 0.103 | $0.750 | 0.00GB (API) |
| HunyuanVideo-1.5 | N/A | 696.23s | 28.98 ± 3.79 | 0.262 ± 0.155 | $1.454 | 39.08GB |

### Cost-Efficiency Rankings

*Efficiency Score = (CLIP Score × Frame Consistency) / Compute Cost*

| Rank | Model | Efficiency Score | Quality Score | Cost |
|------|-------|------------------|---------------|------|
| 🥇 1 | CogVideoX-2b | 198.26 | 16.76 | $0.085 |
| 🥈 2 | CogVideoX-5b | 90.45 | 18.12 | $0.200 |
| 🥉 3 | CogVideoX1.5-5B | 23.76 | 13.74 | $0.578 |
| 4 | Google Veo 3.1 | 9.92 | 7.43 | $0.750 |
| 5 | HunyuanVideo-1.5 | 5.21 | 7.58 | $1.454 |

---

## 🚀 Quick Start

### Run CogVideoX Experiment (Local GPU)
```bash
cd CogVideoX
pip install -r requirements.txt
python experiment.py
```

### Run Google Veo Experiment (Cloud API)
```bash
cd GoogleVeo
pip install -r requirements.txt
export GEMINI_API_KEY='your-api-key'
python experiment_veo.py
```

### Run HunyuanVideo Experiment (Local GPU)
```bash
cd HunyuanVideo
pip install -r requirements.txt
python experiment_hunyuan.py
```

### Run Statistical Analysis
```bash
cd StatisticalAnalysis
source venv/bin/activate
python statistical_analysis.py
```

---

## 📧 Contact

**Author**: Roan Guilherme Weigert Salgueiro  
**LinkedIn**: [\[Your LinkedIn\]  ](https://www.linkedin.com/in/-roan/)
**GitHub**: [\[Your GitHub\]](https://github.com/roangws/)

---

## 📄 Citation

If you use this research in your work, please cite:

```bibtex
@misc{salgueiro2026costefficiency,
  author = {Salgueiro, Roan Guilherme Weigert},
  title = {Cost-Efficiency Metrics: Evaluating Computational and Resource Efficiency in Generative AI Video Models},
  year = {2026},
  publisher = {GitHub},
  note = {Benchmark comparing CogVideoX (local) vs Google Veo 3.1 (cloud)}
}
```

---

## 📜 License

This project is provided for academic and research purposes. Please respect the licenses of underlying models and APIs.

---

**Last Updated**: January 21, 2026  
**Total Videos Generated**: 63 (45 CogVideoX + 9 Veo + 9 HunyuanVideo)  
**Experiment Status**: ✅ Complete  
**Statistical Analysis**: ✅ Complete (ANOVA, Tukey HSD, Cohen's d)
