# Cost-Efficiency Metrics: Evaluating Computational and Resource Efficiency in Generative AI Video Models

This project provides a comprehensive benchmarking framework for evaluating the cost-efficiency of generative AI video models, specifically focusing on the CogVideoX family of models from THUDM.

## Overview

The benchmark evaluates multiple video generation models across key performance metrics:
- **Computational Efficiency**: Inference time, memory usage, and GPU utilization
- **Cost Efficiency**: Dollar cost per video generation based on GPU runtime
- **Quality Metrics**: CLIP score (text-video alignment) and frame consistency (motion smoothness)
- **Statistical Analysis**: Pairwise comparisons and significance testing

## Models Tested

The framework currently supports:
- **THUDM/CogVideoX-5b**: 5-billion parameter model
- **THUDM/CogVideoX-2b**: 2-billion parameter model  
- **THUDM/CogVideoX1.5-5B**: Enhanced 5-billion parameter model

## Project Structure

```
├── experiment.py              # Main benchmarking script
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── checkpoint.json           # Experiment checkpoint (auto-generated)
├── results.csv               # Detailed results (auto-generated)
├── summary.txt               # Summary statistics (auto-generated)
├── progress_log.txt          # Real-time progress log (auto-generated)
├── errors.log                # Error log (auto-generated)
└── output_videos/            # Generated video outputs
    ├── CogVideoX-5b/
    ├── CogVideoX-2b/
    └── CogVideoX1.5-5B/
```

## How the Benchmark Works

### 1. Experimental Design

The benchmark runs each model through a standardized test suite:
- **Prompts**: 3 diverse text prompts testing different scenarios
- **Repetitions**: 5 runs per prompt for statistical significance
- **Total Videos**: 45 videos per complete experiment (3 models × 3 prompts × 5 runs)

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
- **Pairwise T-Tests**: Statistical significance between models (p < 0.05)
- **Success Rate**: Percentage of successful video generations
- **Best Model Identification**: Based on cost-efficiency ratio

## Installation and Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- Linux or macOS environment

### Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `torch` (with CUDA support)
- `diffusers` (for CogVideoX models)
- `transformers` (for CLIP scoring)
- `pandas`, `numpy`, `scipy` (data analysis)
- `opencv-python` (video processing)

## Running the Benchmark

### Basic Usage

```bash
python experiment.py
```

### What Happens During Execution

1. **Initialization**
   - Checks CUDA availability
   - Initializes CLIP metric for quality evaluation
   - Creates output directories

2. **Model Loading**
   - Loads each model sequentially with memory optimizations
   - Applies CPU offloading and attention slicing where available

3. **Video Generation**
   - Generates videos for each prompt and repetition
   - Measures inference time and memory usage
   - Applies post-processing (denoising, color inversion)
   - Saves videos to organized directory structure

4. **Quality Evaluation**
   - Calculates CLIP score using middle frame
   - Computes frame consistency across all frames
   - Logs all metrics to CSV file

5. **Progress Tracking**
   - Real-time progress updates with ETA
   - Checkpoint saving every video (resumable)
   - Periodic summaries every 3 videos

6. **Statistical Analysis**
   - Generates comprehensive statistics
   - Performs pairwise model comparisons
   - Identifies best performing model

## Configuration

### Key Parameters (modifiable in `experiment.py`)

```python
# Models to test
MODELS = [
    "THUDM/CogVideoX-5b",
    "THUDM/CogVideoX-2b", 
    "THUDM/CogVideoX1.5-5B"
]

# Test prompts
PROMPTS = [
    "A person walking in a park on a sunny day",
    "A car driving on a highway with trees in background",
    "A cat playing with a ball in a living room"
]

# Generation parameters
NUM_RUNS_PER_PROMPT = 5
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 6.5

# Cost calculation
POWER_CONSUMPTION_WATTS = 200
GPU_HOURLY_COST = 1.18  # Adjust based on your GPU
```

### Model-Specific Configurations

Each model has optimized settings:
- **CogVideoX-5b/2b**: 49 frames at 720×480
- **CogVideoX1.5-5B**: 81 frames at 1360×768

## Output Files

### Results CSV (`results.csv`)
Contains detailed metrics for each video generation:
- Model name, prompt text, run number
- Inference time, memory usage, cost
- CLIP score, frame consistency
- Success/failure status

### Summary Report (`summary.txt`)
Human-readable summary with:
- Overall statistics
- Model comparison tables
- Best model recommendations

### Progress Log (`progress_log.txt`)
Real-time execution log with timestamps and ETAs.

### Error Log (`errors.log`)
Detailed error information for failed generations.

### Generated Videos
Saved in `output_videos/` with structure:
```
output_videos/
├── CogVideoX-5b/
│   ├── prompt_1/
│   │   ├── run_1.mp4
│   │   ├── run_2.mp4
│   │   └── ...
│   └── ...
```

## Resuming Interrupted Experiments

The framework automatically saves checkpoints. If interrupted, simply rerun:

```bash
python experiment.py
```

It will detect the checkpoint and resume from the last completed video.

## Interpreting Results

### Key Metrics to Watch

1. **Cost-Efficiency Score**: Primary metric for overall performance
2. **Inference Time**: Critical for real-time applications
3. **Memory Usage**: Determines hardware requirements
4. **CLIP Score**: Higher = better text-video alignment
5. **Frame Consistency**: Higher = smoother motion

### Statistical Significance

Look for asterisks (*) in pairwise comparisons, indicating p < 0.05 significance.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Framework automatically reduces inference steps from 50 to 30
   - Consider using a smaller model or reducing batch size

2. **Model Download Failures**
   - Ensure internet connection for first-time downloads
   - Models cache locally after first download

3. **CLIP Score Initialization Errors**
   - Experiment continues without CLIP scoring if initialization fails
   - Check transformers library installation

### Performance Optimization

- Use `torch.compile()` for PyTorch 2.0+ (add in model loading)
- Enable mixed precision with `torch.bfloat16` (already enabled)
- Consider gradient checkpointing for memory-constrained systems

## Contributing

To extend the benchmark:

1. **Add New Models**: Update `MODELS` list and `MODEL_CONFIGS`
2. **New Metrics**: Implement measurement functions and update results
3. **Custom Prompts**: Modify `PROMPTS` array for domain-specific testing

## License

This project is provided for research and evaluation purposes. Please respect the licenses of the underlying models and libraries.

## Citation

If you use this benchmark in your research, please cite:

```
Cost-Efficiency Metrics: Evaluating Computational and Resource Efficiency in Generative AI Video Models
Benchmark framework for CogVideoX family of models
```
