# HunyuanVideo Video Generation Benchmark

**Author**: Roan Guilherme Weigert Salgueiro

This experiment evaluates the HunyuanVideo-1.5 model for local GPU-based video generation, measuring computational efficiency, cost, and quality metrics for comparison with cloud-based alternatives.

## 🎯 Experiment Completed Successfully

**Status**: ✅ All 9 videos generated successfully  
**Date**: January 2026  
**Total Videos**: 9 (1 model × 3 prompts × 3 runs)  
**Success Rate**: 100% (9/9 videos)

## 📊 Key Results

### Performance Metrics by Model

| Model | Avg Time (s) | CLIP Score | Consistency | Total Cost | Videos |
|-------|-------------|------------|-------------|------------|--------|
| **HunyuanVideo-1.5** | 696.23 ± 0.42 | 29.0 ± 3.8 | 0.262 ± 0.155 | $13.09 | 9 |

### Key Findings
- **Average Generation Time**: 696.23s (~11.6 minutes per video)
- **Quality Score**: CLIP: 29.0 (text-video alignment)
- **Frame Consistency**: 0.262 (motion smoothness)
- **Cost per Video**: $1.45 per video
- **Memory Usage**: 39.08GB consistent across all runs

## Overview

- **Model**: HunyuanVideo-1.5 (720p text-to-video)
- **Hardware**: Local GPU (NVIDIA A100-SXM4-80GB)
- **Prompts**: 3 standardized prompts
- **Runs per prompt**: 3 (N=3 for statistical validity)
- **Total videos**: 9
- **Video specs**: 75 frames @ 24fps, 50 inference steps
- **GPU Cost**: $1.18/hour (based on power consumption)

## 🎬 Generated Videos Showcase

Click on any image to view the full video file. Showing all 3 runs for each prompt.

### HunyuanVideo-1.5 (1.5 Model - 720p)

#### Prompt 1: "A person walking in a park on a sunny day"

<table>
  <tr>
    <td align="center">
      <a href="output_videos/hunyuan_prompt_1/hunyuan_prompt_1_run_1.mp4">
        <img src="output_frames/hunyuan_prompt_1/hunyuan_prompt_1_run_1_middle_frame.jpg" width="250px" alt="Run 1"/>
      </a><br/>
      <b>Run 1</b><br/>
      CLIP: 28.25 | Consistency: 0.361
    </td>
    <td align="center">
      <a href="output_videos/hunyuan_prompt_1/hunyuan_prompt_1_run_2.mp4">
        <img src="output_frames/hunyuan_prompt_1/hunyuan_prompt_1_run_2_middle_frame.jpg" width="250px" alt="Run 2"/>
      </a><br/>
      <b>Run 2</b><br/>
      CLIP: 30.30 | Consistency: 0.488
    </td>
    <td align="center">
      <a href="output_videos/hunyuan_prompt_1/hunyuan_prompt_1_run_3.mp4">
        <img src="output_frames/hunyuan_prompt_1/hunyuan_prompt_1_run_3_middle_frame.jpg" width="250px" alt="Run 3"/>
      </a><br/>
      <b>Run 3</b><br/>
      CLIP: 32.18 | Consistency: 0.467
    </td>
  </tr>
</table>

#### Prompt 2: "A car driving on a highway with trees in background"

<table>
  <tr>
    <td align="center">
      <a href="output_videos/hunyuan_prompt_2/hunyuan_prompt_2_run_1.mp4">
        <img src="output_frames/hunyuan_prompt_2/hunyuan_prompt_2_run_1_middle_frame.jpg" width="250px" alt="Run 1"/>
      </a><br/>
      <b>Run 1</b><br/>
      CLIP: 23.32 | Consistency: 0.308
    </td>
    <td align="center">
      <a href="output_videos/hunyuan_prompt_2/hunyuan_prompt_2_run_2.mp4">
        <img src="output_frames/hunyuan_prompt_2/hunyuan_prompt_2_run_2_middle_frame.jpg" width="250px" alt="Run 2"/>
      </a><br/>
      <b>Run 2</b><br/>
      CLIP: 27.24 | Consistency: 0.097
    </td>
    <td align="center">
      <a href="output_videos/hunyuan_prompt_2/hunyuan_prompt_2_run_3.mp4">
        <img src="output_frames/hunyuan_prompt_2/hunyuan_prompt_2_run_3_middle_frame.jpg" width="250px" alt="Run 3"/>
      </a><br/>
      <b>Run 3</b><br/>
      CLIP: 23.68 | Consistency: 0.189
    </td>
  </tr>
</table>

#### Prompt 3: "A cat playing with a ball in a living room"

<table>
  <tr>
    <td align="center">
      <a href="output_videos/hunyuan_prompt_3/hunyuan_prompt_3_run_1.mp4">
        <img src="output_frames/hunyuan_prompt_3/hunyuan_prompt_3_run_1_middle_frame.jpg" width="250px" alt="Run 1"/>
      </a><br/>
      <b>Run 1</b><br/>
      CLIP: 30.16 | Consistency: 0.093
    </td>
    <td align="center">
      <a href="output_videos/hunyuan_prompt_3/hunyuan_prompt_3_run_2.mp4">
        <img src="output_frames/hunyuan_prompt_3/hunyuan_prompt_3_run_2_middle_frame.jpg" width="250px" alt="Run 2"/>
      </a><br/>
      <b>Run 2</b><br/>
      CLIP: 34.84 | Consistency: 0.250
    </td>
    <td align="center">
      <a href="output_videos/hunyuan_prompt_3/hunyuan_prompt_3_run_3.mp4">
        <img src="output_frames/hunyuan_prompt_3/hunyuan_prompt_3_run_3_middle_frame.jpg" width="250px" alt="Run 3"/>
      </a><br/>
      <b>Run 3</b><br/>
      CLIP: 30.83 | Consistency: 0.101
    </td>
  </tr>
</table>

> **Note**: Videos are 75 frames at 24fps. Images show the middle frame from each video. Click any image to download and view the full video.

## Project Structure

```
├── experiment_hunyuan.py     # Main benchmarking script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── checkpoint.json           # Experiment checkpoint (auto-generated)
├── results_hunyuan.csv       # Detailed results (auto-generated)
├── Log_HunyuanVideo.txt      # Complete execution log (auto-generated)
├── output_videos/            # Generated video outputs
│   ├── hunyuan_prompt_1/
│   ├── hunyuan_prompt_2/
│   └── hunyuan_prompt_3/
└── output_frames/            # Extracted thumbnail frames
    ├── hunyuan_prompt_1/
    ├── hunyuan_prompt_2/
    └── hunyuan_prompt_3/
```

## How the Benchmark Works

### 1. Experimental Design

The benchmark runs the model through a standardized test suite:
- **Prompts**: 3 diverse text prompts testing different scenarios
- **Repetitions**: 3 runs per prompt for statistical significance
- **Total Videos**: 9 videos per complete experiment (1 model × 3 prompts × 3 runs)

### 2. Metrics Calculated

#### Performance Metrics
- **Inference Time**: Seconds per video generation
- **Peak Memory**: Maximum GPU memory usage in GB
- **Compute Cost**: Dollar cost based on GPU hourly rate ($1.18/hour)

#### Quality Metrics
- **CLIP Score**: Measures text-video alignment (0-100 scale)
- **Frame Consistency**: Evaluates motion smoothness (0-1 scale)

#### Cost-Efficiency Formula
```
Cost-Efficiency = (CLIP Score × Frame Consistency) / (Time × Cost)
```
Higher values indicate better quality per computational cost.

### 3. Statistical Analysis

The framework provides:
- **Descriptive Statistics**: Mean ± standard deviation for all metrics
- **Success Rate**: Percentage of successful video generations
- **Per-Video Cost Analysis**: Based on actual GPU usage

## Installation and Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended: 40GB+ VRAM for HunyuanVideo-1.5)
- Linux or macOS environment

### Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `torch` (with CUDA support)
- `diffusers` (for HunyuanVideo models)
- `transformers` (for CLIP scoring)
- `pandas`, `numpy`, `scipy` (data analysis)
- `opencv-python` (video processing)

## Running the Benchmark

### Basic Usage

```bash
python experiment_hunyuan.py
```

### What Happens During Execution

1. **Initialization**
   - Checks CUDA availability
   - Initializes CLIP metric for quality evaluation
   - Creates output directories

2. **Model Loading**
   - Loads HunyuanVideo-1.5 model with memory optimizations
   - Applies CPU offloading and VAE tiling for memory efficiency

3. **Video Generation**
   - Generates videos for each prompt and repetition
   - Measures inference time and memory usage
   - Saves videos to organized directory structure

4. **Quality Evaluation**
   - Calculates CLIP score using middle frame
   - Computes frame consistency across all frames
   - Logs all metrics to CSV file

5. **Progress Tracking**
   - Real-time progress updates with ETA
   - Checkpoint saving every video (resumable)
   - Detailed logging to Log_HunyuanVideo.txt

6. **Statistical Analysis**
   - Generates comprehensive statistics
   - Calculates total costs and averages

## Configuration

### Key Parameters (modifiable in `experiment_hunyuan.py`)

```python
# Model to test
MODEL = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"

# Test prompts
PROMPTS = [
    "A person walking in a park on a sunny day",
    "A car driving on a highway with trees in background",
    "A cat playing with a ball in a living room"
]

# Generation parameters
NUM_RUNS_PER_PROMPT = 3
NUM_INFERENCE_STEPS = 50
VIDEO_LENGTH = 75  # frames
FPS = 24

# Cost calculation
POWER_CONSUMPTION_WATTS = 200
GPU_HOURLY_COST = 1.18  # Adjust based on your GPU
```

### Model-Specific Configurations

HunyuanVideo-1.5 settings:
- **Frames**: 75 frames @ 24fps
- **Inference Steps**: 50
- **Memory Optimizations**: CPU offload + VAE tiling enabled

## Output Files

### Results CSV (`results_hunyuan.csv`)
Contains detailed metrics for each video generation:
- Model name, prompt text, run number
- Inference time, memory usage, cost
- CLIP score, frame consistency
- Success/failure status

### Execution Log (`Log_HunyuanVideo.txt`)
Complete execution log with:
- Model loading progress
- Generation progress for each video
- Real-time metrics and timestamps
- Statistical summary

### Generated Videos
Saved in `output_videos/` with structure:
```
output_videos/
├── hunyuan_prompt_1/
│   ├── hunyuan_prompt_1_run_1.mp4
│   ├── hunyuan_prompt_1_run_2.mp4
│   └── hunyuan_prompt_1_run_3.mp4
├── hunyuan_prompt_2/
│   ├── hunyuan_prompt_2_run_1.mp4
│   ├── hunyuan_prompt_2_run_2.mp4
│   └── hunyuan_prompt_2_run_3.mp4
└── hunyuan_prompt_3/
    ├── hunyuan_prompt_3_run_1.mp4
    ├── hunyuan_prompt_3_run_2.mp4
    └── hunyuan_prompt_3_run_3.mp4
```

## Resuming Interrupted Experiments

The framework automatically saves checkpoints. If interrupted, simply rerun:

```bash
python experiment_hunyuan.py
```

It will detect the checkpoint and resume from the last completed video.

## Interpreting Results

### Key Metrics to Watch

1. **Cost-Efficiency Score**: Primary metric for overall performance
2. **Inference Time**: Critical for real-time applications
3. **Memory Usage**: Determines hardware requirements
4. **CLIP Score**: Higher = better text-video alignment
5. **Frame Consistency**: Higher = smoother motion

### Performance Observations

- **Consistent Timing**: Very stable generation time (~696s per video, ±0.42s std dev)
- **High Memory Usage**: Requires 39GB VRAM consistently
- **Variable Quality**: CLIP scores range from 23.3 to 34.8 depending on prompt
- **Consistency Challenges**: Frame consistency varies significantly (0.097 to 0.488)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - HunyuanVideo-1.5 requires significant VRAM (39GB+)
   - CPU offloading and VAE tiling are already enabled
   - Consider using a GPU with at least 40GB VRAM

2. **Model Download Failures**
   - Ensure internet connection for first-time downloads
   - Models cache locally after first download
   - Large model size may require stable connection

3. **CLIP Score Initialization Errors**
   - Experiment continues without CLIP scoring if initialization fails
   - Check transformers library installation

### Performance Optimization

- Use `torch.compile()` for PyTorch 2.0+ (add in model loading)
- Enable mixed precision with `torch.bfloat16` (already enabled)
- Ensure adequate cooling for long generation sessions

## Contributing

To extend the benchmark:

1. **Add New Models**: Update model path and configurations
2. **New Metrics**: Implement measurement functions and update results
3. **Custom Prompts**: Modify `PROMPTS` array for domain-specific testing

## License

This project is provided for research and evaluation purposes. Please respect the licenses of the underlying models and libraries.

## 🔗 Links & Resources

### Official Documentation
- **HunyuanVideo GitHub**: https://github.com/Tencent/HunyuanVideo
- **Hugging Face Models**: https://huggingface.co/hunyuanvideo-community
- **Diffusers Documentation**: https://huggingface.co/docs/diffusers
- **PyTorch**: https://pytorch.org

### Repository
- **GitHub**: [Cost-Efficiency Metrics for Generative AI Video Models](https://github.com/yourusername/video-generation-benchmark)
- **CogVideoX Comparison**: See `../CogVideoX/` directory for alternative local models
- **Google Veo Comparison**: See `../GoogleVeo/` directory for cloud API results

### Related Work
- **Research Paper**: "Cost-Efficiency Metrics: Evaluating Computational and Resource Efficiency in Generative AI Video Models"
- **Author**: Roan Guilherme Weigert Salgueiro

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **PyTorch**: Deep learning framework with CUDA support
- **Diffusers**: Hugging Face library for HunyuanVideo models
- **Transformers**: CLIP model for quality evaluation
- **OpenCV**: Video processing and frame extraction
- **Pandas & NumPy**: Data analysis and numerical computations
- **SciPy**: Statistical analysis

## 📈 Technical Achievements

1. ✅ **HunyuanVideo-1.5 Benchmarking**: Successfully tested 720p text-to-video model
2. ✅ **9 Videos Generated**: 100% success rate across all prompts
3. ✅ **Quality Metrics**: CLIP score and frame consistency evaluation
4. ✅ **Cost Tracking**: Accurate GPU cost calculation based on power consumption
5. ✅ **Statistical Analysis**: Comprehensive metrics with standard deviations
6. ✅ **Checkpoint System**: Resumable experiments with automatic state saving
7. ✅ **Memory Optimization**: CPU offloading and VAE tiling for large models

## 📝 Key Findings

### Performance Analysis
- **Generation Time**: Highly consistent at 696.23s (±0.42s) per video
- **Memory Requirements**: Stable 39.08GB VRAM usage
- **Cost per Video**: $1.45 per video generation
- **Total Experiment Cost**: $13.09 for 9 videos

### Quality Observations
- **Best CLIP Score**: Prompt 3 Run 2 (cat with ball) achieved 34.84
- **Best Consistency**: Prompt 1 Run 2 (person in park) achieved 0.488
- **Prompt Variability**: Highway scene (Prompt 2) showed lower scores overall
- **Quality Trade-offs**: Higher CLIP scores don't always correlate with higher consistency

### Cost Analysis
- **Cost per Video**: $1.45 (significantly higher than CogVideoX models)
- **Time per Video**: ~11.6 minutes (slower than CogVideoX-2b, faster than CogVideoX1.5-5B)
- **Memory Efficiency**: Requires high-end GPU (A100 80GB used in testing)

## 📊 Integration with Research Paper

This benchmark provides direct comparison data for:
- **Cost-efficiency**: Local GPU ($1.45/video) vs CogVideoX ($0.08-$0.58/video) vs Cloud API
- **Quality**: CLIP scores (29.0 avg) vs other models
- **Speed**: 696s (local) vs other local/cloud alternatives
- **Hardware Requirements**: 39GB VRAM vs other models

### Analysis Ready
Use `results_hunyuan.csv` alongside CogVideoX and Google Veo results for:
- ✅ Statistical comparison (t-tests, ANOVA)
- ✅ Cost-benefit analysis
- ✅ Quality-efficiency tradeoffs
- ✅ Model selection recommendations
- ✅ Hardware requirement planning

## 🎓 Academic Context

This benchmark is part of a comprehensive study comparing:
1. **Local Models**: CogVideoX (2B, 5B, 1.5-5B variants) + HunyuanVideo-1.5 - this experiment
2. **Cloud APIs**: Google Veo 3.1
3. **Metrics**: Cost, speed, quality, scalability, hardware requirements
4. **Goal**: Provide data-driven recommendations for video generation deployment

## 📧 Contact & Support

**Author**: Roan Guilherme Weigert Salgueiro  
**Email**: [Your email]  
**LinkedIn**: [Your LinkedIn]  
**GitHub**: [Your GitHub]

For issues with:
- **HunyuanVideo Models**: [Tencent HunyuanVideo GitHub Issues](https://github.com/Tencent/HunyuanVideo/issues)
- **Experiment code**: Check `Log_HunyuanVideo.txt` for detailed errors
- **CUDA/GPU issues**: Ensure PyTorch is installed with CUDA support

## 📄 License

This project is part of academic research. Please cite if used in publications.

## 🙏 Acknowledgments

- Tencent team for HunyuanVideo models
- Hugging Face for Diffusers library
- OpenAI for CLIP model used in quality evaluation

---

**Last Updated**: January 2026  
**Experiment Status**: ✅ Complete  
**Results**: 9/9 videos successfully generated

## Citation

If you use this benchmark in your research, please cite:

```
Cost-Efficiency Metrics: Evaluating Computational and Resource Efficiency in Generative AI Video Models
Author: Roan Guilherme Weigert Salgueiro
Benchmark framework for HunyuanVideo-1.5 model
```
