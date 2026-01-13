# Cost-Efficiency Metrics: Evaluating Computational and Resource Efficiency in Generative AI Video Models

**Author**: Roan Guilherme Weigert Salgueiro  
**Date**: January 2026

A comprehensive benchmark study comparing **local GPU-based** video generation (CogVideoX) versus **cloud API-based** generation (Google Veo 3.1), measuring cost, speed, quality, and scalability.

---

## 🎯 Research Objective

Provide data-driven insights for choosing between local and cloud-based AI video generation, evaluating:
- **Cost efficiency**: GPU compute costs vs API pricing
- **Performance**: Inference speed and throughput
- **Quality**: CLIP score (text-video alignment) and frame consistency
- **Scalability**: Hardware requirements vs cloud availability

---

## 📊 Key Research Findings

### Head-to-Head Comparison

| Metric | CogVideoX-2b (Local) | CogVideoX-5b (Local) | Google Veo 3.1 (Cloud) |
|--------|---------------------|---------------------|------------------------|
| **Avg Time/Video** | 257.90s | 611.11s | 59.61s |
| **Cost/Video** | $0.08 | $0.20 | $0.75 |
| **CLIP Score** | 30.90 | 32.20 | N/A* |
| **Frame Consistency** | 0.542 | 0.563 | 0.217 |
| **Hardware Required** | 12GB+ GPU | 12GB+ GPU | None (API) |

*CLIP score not available for Veo due to transformers dependency

### 🏆 Winner by Category

| Category | Winner | Reason |
|----------|--------|--------|
| **⚡ Fastest** | Google Veo 3.1 | 59.61s vs 257.90s (4.3× faster than fastest local) |
| **💰 Cheapest** | CogVideoX-2b | $0.08/video vs $0.75 (9.4× cheaper than cloud) |
| **🎨 Best Quality** | CogVideoX-5b | CLIP 32.20, Consistency 0.563 |
| **🚀 Easiest Setup** | Google Veo 3.1 | No GPU required, just API key |
| **📈 Best for Batch** | CogVideoX-2b | $0.08 × 1000 = $80 vs $750 cloud |

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
      Consistency: 0.165 | 64s
    </td>
    <td align="center">
      <a href="GoogleVeo/output_videos/veo_prompt_2/veo_prompt_2_run_1.mp4">
        <img src="GoogleVeo/output_frames/veo_prompt_2/run_1_middle_frame.jpg" width="200px"/>
      </a><br/>
      <b>Car on Highway</b><br/>
      Consistency: 0.152 | 54s
    </td>
    <td align="center">
      <a href="GoogleVeo/output_videos/veo_prompt_3/veo_prompt_3_run_1.mp4">
        <img src="GoogleVeo/output_frames/veo_prompt_3/run_1_middle_frame.jpg" width="200px"/>
      </a><br/>
      <b>Cat with Ball</b><br/>
      Consistency: 0.364 | 54s
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
├── CogVideoX/                # Local GPU experiments
│   ├── README.md             # Detailed CogVideoX documentation
│   ├── experiment.py         # Benchmarking script
│   ├── results.csv           # 45 video results
│   ├── output_videos/        # Generated videos (45)
│   └── output_frames/        # Preview frames (27)
└── GoogleVeo/                # Cloud API experiments
    ├── README.md             # Detailed Veo documentation
    ├── experiment_veo.py     # API integration script
    ├── results_veo.csv       # 9 video results
    ├── output_videos/        # Generated videos (9)
    └── output_frames/        # Preview frames (9)
```

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Local Models** | CogVideoX (2B, 5B, 1.5-5B), PyTorch, Diffusers, CUDA |
| **Cloud API** | Google Gemini API, Veo 3.1, google-genai |
| **Quality Metrics** | CLIP (OpenAI), Frame Consistency Analysis |
| **Data Analysis** | Pandas, NumPy, SciPy |
| **Video Processing** | OpenCV, FFmpeg |
| **Language** | Python 3.8+ |

---

## 📈 Detailed Results

### CogVideoX Models Comparison

| Model | Parameters | Avg Time | CLIP Score | Consistency | Cost/Video |
|-------|------------|----------|------------|-------------|------------|
| CogVideoX-2b | 2B | 257.90s | 30.90 ± 1.91 | 0.542 ± 0.193 | $0.08 |
| CogVideoX-5b | 5B | 611.11s | 32.20 ± 2.42 | 0.563 ± 0.218 | $0.20 |
| CogVideoX1.5-5B | 5B+ | 1762.35s | 32.40 ± 2.85 | 0.424 ± 0.135 | $0.58 |

### Google Veo 3.1 Results

| Metric | Value |
|--------|-------|
| Average Inference Time | 59.61 ± 7.54 seconds |
| Frame Consistency | 0.217 ± 0.078 |
| Cost per Video | $0.75 (5 seconds × $0.15/sec) |
| Video Duration | 5 seconds at 8 FPS |

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

---

## 📧 Contact

**Author**: Roan Guilherme Weigert Salgueiro  
**Email**: [Your email]  
**LinkedIn**: [Your LinkedIn]  
**GitHub**: [Your GitHub]

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

**Last Updated**: January 13, 2026  
**Total Videos Generated**: 54 (45 CogVideoX + 9 Veo)  
**Experiment Status**: ✅ Complete
