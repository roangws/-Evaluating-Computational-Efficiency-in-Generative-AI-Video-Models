import torch
import time
import gc
import pandas as pd
from scipy import stats
from diffusers import CogVideoXPipeline
import warnings

warnings.filterwarnings('ignore')

# Configuration
MODELS = [
    "THUDM/CogVideoX-2b",
    "THUDM/CogVideoX-5b"
]

PROMPTS = [
    "A person walking in a park on a sunny day",
    "A car driving on a highway with trees in background",
    "A cat playing with a ball in a living room"
]

NUM_RUNS_PER_PROMPT = 5
NUM_FRAMES = 16
HEIGHT = 480
WIDTH = 720
NUM_INFERENCE_STEPS = 50
POWER_CONSUMPTION_WATTS = 200

# Results storage
results = []

def measure_inference(pipeline, prompt, model_name, run_number, prompt_index, num_steps=50):
    """
    Measure inference time and peak GPU memory for a single video generation.
    
    Args:
        pipeline: The loaded CogVideoX pipeline
        prompt: Text prompt for video generation
        model_name: Name of the model being tested
        run_number: Current run number (1-5)
        prompt_index: Index of the prompt (1-3)
        num_steps: Number of diffusion steps
    
    Returns:
        dict: Dictionary containing measurement results
    """
    # Clear GPU cache and reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        try:
            _video = pipeline(
                prompt=prompt,
                num_frames=NUM_FRAMES,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=num_steps,
            ).frames[0]
        except Exception as e:
            print(f"Error during inference: {e}")
            raise
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Measure peak GPU memory
    peak_memory_bytes = torch.cuda.max_memory_allocated()
    peak_memory_gb = peak_memory_bytes / (1024 ** 3)
    
    result = {
        'model_name': model_name,
        'run_number': run_number,
        'prompt_index': prompt_index,
        'prompt_text': prompt,
        'inference_time_seconds': inference_time,
        'peak_memory_gb': peak_memory_gb
    }
    
    print(f"✓ {model_name} | Prompt {prompt_index} | Run {run_number} | "
          f"Time: {inference_time:.2f}s | Memory: {peak_memory_gb:.2f}GB")
    
    return result

def load_model(model_name):
    """
    Load a CogVideoX model with error handling.
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        CogVideoXPipeline: Loaded pipeline
    """
    try:
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name}")
        print(f"{'='*60}")
        
        pipeline = CogVideoXPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
        pipeline = pipeline.to("cuda")
        
        print("✓ Model loaded successfully\n")
        return pipeline
        
    except Exception as e:
        print(f"✗ Failed to load model {model_name}: {e}")
        raise

def cleanup_model(pipeline):
    """
    Clean up model from memory and GPU.
    
    Args:
        pipeline: The pipeline to clean up
    """
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    print("\n✓ Model cleaned up from memory\n")

def run_experiments():
    """
    Run the complete benchmarking experiment for all models.
    """
    print("="*60)
    print("COGVIDEOX COST-EFFICIENCY BENCHMARKING EXPERIMENT")
    print("="*60)
    print(f"Models to test: {len(MODELS)}")
    print(f"Prompts per model: {len(PROMPTS)}")
    print(f"Runs per prompt: {NUM_RUNS_PER_PROMPT}")
    print(f"Total videos to generate: {len(MODELS) * len(PROMPTS) * NUM_RUNS_PER_PROMPT}")
    print(f"Video settings: {NUM_FRAMES} frames, {HEIGHT}x{WIDTH}, {NUM_INFERENCE_STEPS} steps")
    print("="*60)
    
    for model_idx, model_name in enumerate(MODELS):
        # Determine number of inference steps (reduce for 5B model if needed)
        num_steps = NUM_INFERENCE_STEPS
        if "5b" in model_name.lower():
            # Start with default, will reduce if memory error occurs
            num_steps = NUM_INFERENCE_STEPS
        
        try:
            # Load model
            pipeline = load_model(model_name)
            
            # Run experiments for each prompt
            for prompt_idx, prompt in enumerate(PROMPTS):
                print(f"\nPrompt {prompt_idx + 1}/{len(PROMPTS)}: \"{prompt}\"")
                print("-" * 60)
                
                for run in range(1, NUM_RUNS_PER_PROMPT + 1):
                    try:
                        result = measure_inference(
                            pipeline=pipeline,
                            prompt=prompt,
                            model_name=model_name,
                            run_number=run,
                            prompt_index=prompt_idx + 1,
                            num_steps=num_steps
                        )
                        results.append(result)
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() and "5b" in model_name.lower():
                            print("\n⚠ Memory error detected. Reducing steps to 30 and retrying...")
                            num_steps = 30
                            torch.cuda.empty_cache()
                            gc.collect()
                            
                            result = measure_inference(
                                pipeline=pipeline,
                                prompt=prompt,
                                model_name=model_name,
                                run_number=run,
                                prompt_index=prompt_idx + 1,
                                num_steps=num_steps
                            )
                            results.append(result)
                        else:
                            raise
            
            # Clean up model before loading next one
            cleanup_model(pipeline)
            
        except Exception as e:
            print(f"\n✗ Error with model {model_name}: {e}")
            print("Continuing with next model...\n")
            continue

def calculate_statistics():
    """
    Calculate and display statistics for all models.
    """
    df = pd.DataFrame(results)
    
    # Save raw results to CSV
    df.to_csv('results.csv', index=False)
    print("\n" + "="*60)
    print("✓ Results saved to results.csv")
    print("="*60)
    
    # Calculate statistics per model
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    stats_data = []
    
    for model_name in df['model_name'].unique():
        model_df = df[df['model_name'] == model_name]
        
        avg_time = model_df['inference_time_seconds'].mean()
        std_time = model_df['inference_time_seconds'].std()
        
        avg_memory = model_df['peak_memory_gb'].mean()
        std_memory = model_df['peak_memory_gb'].std()
        
        # Calculate cost-efficiency: 1 / (time × memory × power)
        cost_efficiency = 1 / (avg_time * avg_memory * POWER_CONSUMPTION_WATTS)
        
        stats_data.append({
            'Model': model_name,
            'Avg Time (s)': f"{avg_time:.2f} ± {std_time:.2f}",
            'Avg Memory (GB)': f"{avg_memory:.2f} ± {std_memory:.2f}",
            'Cost-Efficiency': f"{cost_efficiency:.6f}"
        })
        
        print(f"\n{model_name}:")
        print(f"  Average Time: {avg_time:.2f} ± {std_time:.2f} seconds")
        print(f"  Average Memory: {avg_memory:.2f} ± {std_memory:.2f} GB")
        print(f"  Cost-Efficiency: {cost_efficiency:.6f}")
    
    # Create summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    
    # Perform t-test if we have data for both models
    if len(df['model_name'].unique()) == 2:
        print("\n" + "="*60)
        print("STATISTICAL COMPARISON (T-TEST)")
        print("="*60)
        
        model1_name = MODELS[0]
        model2_name = MODELS[1]
        
        model1_times = df[df['model_name'] == model1_name]['inference_time_seconds']
        model2_times = df[df['model_name'] == model2_name]['inference_time_seconds']
        
        model1_memory = df[df['model_name'] == model1_name]['peak_memory_gb']
        model2_memory = df[df['model_name'] == model2_name]['peak_memory_gb']
        
        # T-test for inference time
        t_stat_time, p_value_time = stats.ttest_ind(model1_times, model2_times)
        print(f"\nInference Time Comparison:")
        print(f"  t-statistic: {t_stat_time:.4f}")
        print(f"  p-value: {p_value_time:.6f}")
        
        if p_value_time < 0.05:
            print("  Result: Statistically significant difference (p < 0.05)")
        else:
            print("  Result: No statistically significant difference (p >= 0.05)")
        
        # T-test for memory usage
        t_stat_memory, p_value_memory = stats.ttest_ind(model1_memory, model2_memory)
        print("\nMemory Usage Comparison:")
        print(f"  t-statistic: {t_stat_memory:.4f}")
        print(f"  p-value: {p_value_memory:.6f}")
        
        if p_value_memory < 0.05:
            print("  Result: Statistically significant difference (p < 0.05)")
        else:
            print("  Result: No statistically significant difference (p >= 0.05)")
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)

def main():
    """
    Main entry point for the experiment.
    """
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This experiment requires a GPU.")
        return
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n")
    
    # Run experiments
    run_experiments()
    
    # Calculate and display statistics
    if len(results) > 0:
        calculate_statistics()
    else:
        print("\n✗ No results collected. Experiment failed.")

if __name__ == "__main__":
    main()
