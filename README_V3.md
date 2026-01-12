# CogVideoX V3 Experiment - Quality Metrics Evaluation

## Overview
This experiment benchmarks 3 CogVideoX models with **quality metrics** (CLIP Score + Frame Consistency) in addition to computational efficiency metrics.

## Models Tested
1. **CogVideoX-5b** - 49 frames, 480×720
2. **CogVideoX-2b** - 49 frames, 480×720  
3. **CogVideoX1.5-5B** - 81 frames, 768×1360

## Quality Metrics

### CLIP Score (0-100)
- Measures text-video alignment
- Uses middle frame vs prompt text
- Higher = better match to prompt
- Expected range: 50-90

### Frame Consistency (0-1)
- Measures motion smoothness
- Analyzes frame-to-frame differences
- Higher = smoother motion
- Expected range: 0.6-0.9

## Cost-Efficiency Formula
```
Cost-Efficiency = (CLIP Score × Frame Consistency) / (Time × Cost)
```
Higher quality + lower cost = better efficiency

## Setup

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Run Experiment
```bash
python experiment_cogvideox_v3.py
```

## Output Files
- `results_v3.csv` - All measurements with quality metrics
- `progress_log_v3.txt` - Real-time progress
- `checkpoint_v3.json` - Resume capability
- `summary_v3.txt` - Periodic summaries
- `errors_v3.log` - Error tracking
- `output_videos_cogvideox_v3/` - Generated videos

## CSV Columns
- `model_name`
- `inference_time_seconds`
- `peak_memory_gb`
- `compute_cost_usd`
- `clip_score` ⭐ NEW
- `frame_consistency` ⭐ NEW
- `status`

## Statistical Analysis
- Mean ± std for all metrics per model
- Pairwise t-tests for 3 model combinations
- Tests: Time, Memory, CLIP Score, Frame Consistency
- Significance level: p < 0.05

## Total Videos
**45 videos** (3 models × 3 prompts × 5 runs)

## Requirements
- CUDA-enabled GPU
- Python 3.8+
- ~40GB GPU memory (for CogVideoX1.5-5B)
- torchmetrics library
