# Cost-Efficiency Metrics: Evaluating Computational and Resource Efficiency in Generative AI Video Models

## Overview

This repository focuses on developing and evaluating cost-efficiency metrics for generative AI video models. The project aims to provide comprehensive benchmarks and analysis tools for measuring computational and resource efficiency in video generation systems.

## Author

**Roan Guilherme Weigert Salgueiro**

## Objectives

- Develop standardized metrics for evaluating computational efficiency in generative AI video models
- Analyze resource utilization patterns across different video generation architectures
- Create benchmarking tools for performance-cost trade-offs
- Provide insights into optimization strategies for video generation systems

## Key Areas

- **Computational Efficiency**: FLOPs, inference time, memory usage
- **Resource Efficiency**: GPU utilization, energy consumption, carbon footprint
- **Cost Analysis**: Training costs, inference costs, operational expenses
- **Performance Metrics**: Video quality vs. computational cost ratios

## Repository Structure

```
├── README.md                 # This file
├── metrics/                  # Efficiency metric implementations
├── benchmarks/              # Benchmarking tools and datasets
├── analysis/                 # Analysis scripts and results
├── models/                   # Model implementations and wrappers
└── docs/                     # Documentation and reports
```

## Getting Started

### CogVideoX Benchmarking Experiment

This repository includes a comprehensive benchmarking experiment for evaluating the cost-efficiency of CogVideoX video generation models.

#### Prerequisites

- CUDA-capable GPU with at least 16GB VRAM (24GB+ recommended for 5B model)
- Python 3.8 or higher
- CUDA toolkit installed

#### Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

#### Running the Experiment

Execute the benchmarking script:
```bash
python experiment.py
```

The script will:
- Download and test two models: CogVideoX-2B and CogVideoX-5B
- Generate 30 videos total (15 per model)
- Measure inference time and GPU memory for each generation
- Save detailed results to `results.csv`
- Display statistical analysis and cost-efficiency metrics

#### What Gets Measured

For each model, the experiment generates videos using 3 different prompts, with 5 runs per prompt:

**Prompts:**
1. "A person walking in a park on a sunny day"
2. "A car driving on a highway with trees in background"
3. "A cat playing with a ball in a living room"

**Metrics Captured:**
- **Inference Time**: Time in seconds to generate each video
- **Peak GPU Memory**: Maximum GPU memory used during generation (in GB)
- **Cost-Efficiency**: Calculated as `1 / (time × memory × power)` where power = 200W

**Video Settings:**
- 16 frames per video
- 480×720 resolution
- 50 diffusion steps (may reduce to 30 for 5B model if memory constrained)
- FP16 precision

#### Expected Output

The experiment produces:

1. **results.csv**: Raw data with all measurements
   - Columns: model_name, run_number, prompt_index, prompt_text, inference_time_seconds, peak_memory_gb

2. **Console Output**: Real-time progress updates showing:
   - Model loading status
   - Per-video generation progress with time and memory
   - Statistical summary with mean ± standard deviation
   - Cost-efficiency comparison table
   - T-test results comparing the two models

3. **Summary Table**: Final comparison showing:
   - Average inference time per model
   - Average memory usage per model
   - Cost-efficiency metric for each model

#### Experiment Duration

- **CogVideoX-2B**: ~5-10 minutes per video (15 videos ≈ 75-150 minutes)
- **CogVideoX-5B**: ~10-20 minutes per video (15 videos ≈ 150-300 minutes)
- **Total**: Approximately 4-8 hours depending on GPU

#### Troubleshooting

**Out of Memory Errors:**
- The script automatically reduces inference steps from 50 to 30 for the 5B model if memory errors occur
- Ensure no other GPU processes are running
- Close unnecessary applications

**Model Download Issues:**
- Models are downloaded from HuggingFace on first run (~2-5GB each)
- Ensure stable internet connection
- Check HuggingFace authentication if needed

**CUDA Not Available:**
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Update GPU drivers if necessary

## Contributing

[Contributing guidelines will be added as the project develops]

## License

[License information will be added as the project develops]

---

*This repository is part of ongoing research into cost-efficient generative AI video models.*
