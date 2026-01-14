# HunyuanVideo-1.5 Experiment on Google Colab

## Prerequisites
- Google Colab Pro account ($9/100 compute units)
- A100 GPU with High RAM runtime

## Setup Instructions

### 1. Configure Runtime
1. Go to **Runtime** → **Change runtime type**
2. Select **A100 GPU**
3. Select **High-RAM**
4. Click **Save**

### 2. Clone Repository
```bash
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO/HunyuanVideo
```

### 3. Install Dependencies
```bash
!pip install -r requirements_hunyuan.txt
```

### 4. Run Experiment
```bash
!python experiment_hunyuan.py
```

The script will:
- Run a **sanity check** first (Prompt 1, Run 1)
- If CLIP score > 0 and status is SUCCESS, proceed to full 9 runs
- If sanity check fails, stop and report error

### 5. Monitor Progress
- Check `progress_log.txt` for real-time updates
- Check `errors.log` if any failures occur
- Results are saved incrementally to `results_hunyuan.csv`

### 6. Download Results
```python
from google.colab import files

# Download CSV results
files.download('results_hunyuan.csv')

# Download all generated videos
!zip -r output_videos.zip output_videos/
files.download('output_videos.zip')

# Download logs
files.download('progress_log.txt')
files.download('errors.log')
```

## Expected Output Structure
```
HunyuanVideo/
├── results_hunyuan.csv          # Main results file
├── output_videos/
│   ├── hunyuan_prompt_1/
│   │   ├── hunyuan_prompt_1_run_1.mp4
│   │   ├── hunyuan_prompt_1_run_2.mp4
│   │   └── hunyuan_prompt_1_run_3.mp4
│   ├── hunyuan_prompt_2/
│   │   ├── hunyuan_prompt_2_run_1.mp4
│   │   ├── hunyuan_prompt_2_run_2.mp4
│   │   └── hunyuan_prompt_2_run_3.mp4
│   └── hunyuan_prompt_3/
│       ├── hunyuan_prompt_3_run_1.mp4
│       ├── hunyuan_prompt_3_run_2.mp4
│       └── hunyuan_prompt_3_run_3.mp4
├── progress_log.txt
├── errors.log
└── checkpoint.json
```

## CSV Schema
The `results_hunyuan.csv` file contains the following columns:
- `model_name`: "HunyuanVideo-1.5"
- `run_number`: 1-3
- `prompt_index`: 1-3
- `prompt_text`: The text prompt used
- `inference_time_seconds`: Time taken for generation
- `peak_memory_gb`: Peak GPU memory usage
- `compute_cost_usd`: Cost based on A100 hourly rate
- `clip_score`: CLIP score for text-video alignment
- `frame_consistency`: Motion smoothness score
- `status`: "SUCCESS", "FAILED", "CLIP_ERROR:...", etc.

## Troubleshooting

### Out of Memory (OOM)
- The script uses `enable_model_cpu_offload()` and `vae.enable_tiling()` to optimize memory
- If OOM still occurs, the script will log it and continue with remaining runs

### CLIP Score Issues
- If CLIP initialization fails, the script continues but CLIP scores will be 0
- Sanity check will fail if CLIP score is 0

### Resuming from Checkpoint
- If the notebook disconnects, simply re-run the experiment script
- It will automatically resume from the last completed video using `checkpoint.json`

## Compute Usage Estimate
- **Expected time per video**: ~5-10 minutes (depending on A100 performance)
- **Total experiment time**: ~45-90 minutes for 9 videos
- **Estimated compute units**: ~6-11 units (~$0.54-$0.99)
