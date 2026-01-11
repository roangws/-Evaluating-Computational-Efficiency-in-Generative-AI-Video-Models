import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import time
import gc
import pandas as pd
from scipy import stats
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import warnings
import json
from datetime import datetime
import numpy as np

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
NUM_FRAMES = 24
HEIGHT = 720
WIDTH = 1280
NUM_INFERENCE_STEPS = 50
POWER_CONSUMPTION_WATTS = 200
GPU_HOURLY_COST = 1.18  # USD/hour - A100 High-RAM (11.77 compute units @ $0.10)

# File paths for checkpointing and logging
CHECKPOINT_FILE = "checkpoint.json"
PROGRESS_LOG_FILE = "progress_log.txt"
RESULTS_CSV_FILE = "results.csv"
SUMMARY_FILE = "summary.txt"
ERROR_LOG_FILE = "errors.log"
OUTPUT_VIDEO_DIR = "output_videos"

# Global tracking variables
results = []
total_videos = len(MODELS) * len(PROMPTS) * NUM_RUNS_PER_PROMPT
videos_completed = 0
experiment_start_time = None

def save_checkpoint(model_name, model_idx, prompt_idx, run_number):
    """
    Save current progress to checkpoint file.
    """
    checkpoint_data = {
        'current_model': model_name,
        'model_idx': model_idx,
        'prompt_idx': prompt_idx,
        'run_number': run_number,
        'videos_completed': videos_completed,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def load_checkpoint():
    """
    Load checkpoint if exists, return None if not.
    """
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None

def log_progress(model_name, run_number, prompt_index, inference_time, peak_memory_gb, quality=0.0):
    """
    Append progress line to progress_log.txt after each video.
    """
    global videos_completed, experiment_start_time
    
    # Calculate ETA
    elapsed_time = time.time() - experiment_start_time
    videos_remaining = total_videos - videos_completed
    
    if videos_completed > 0:
        avg_time_per_video = elapsed_time / videos_completed
        eta_seconds = avg_time_per_video * videos_remaining
        eta_hours = eta_seconds / 3600
    else:
        eta_hours = 0
    
    # Format timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Calculate progress percentage
    progress_pct = (videos_completed / total_videos) * 100
    
    # Create log line
    log_line = (f"[{timestamp}] [{progress_pct:.1f}%] Model: {model_name} | Run: {run_number}/{NUM_RUNS_PER_PROMPT} | "
                f"Prompt: {prompt_index}/{len(PROMPTS)} | Time: {inference_time:.2f}s | "
                f"Memory: {peak_memory_gb:.2f}GB | Quality: {quality:.2f} | ETA: {eta_hours:.1f}h\n")
    
    # Append to progress log
    with open(PROGRESS_LOG_FILE, 'a') as f:
        f.write(log_line)
    
    print(log_line.strip())

def log_error(error_message, model_name, run_number, prompt_index):
    """
    Log error to errors.log with timestamp.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_entry = (f"[{timestamp}] Model: {model_name} | Run: {run_number} | "
                   f"Prompt: {prompt_index} | Error: {error_message}\n")
    
    with open(ERROR_LOG_FILE, 'a') as f:
        f.write(error_entry)
    
    print(f"✗ ERROR LOGGED: {error_message}")

def append_to_csv(result):
    """
    Immediately append result to CSV file (real-time updates).
    """
    df = pd.DataFrame([result])
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(RESULTS_CSV_FILE)
    
    # Append to CSV
    df.to_csv(RESULTS_CSV_FILE, mode='a', header=not file_exists, index=False)

def print_periodic_summary():
    """
    Print and save summary every 3 videos.
    """
    global videos_completed, experiment_start_time
    
    elapsed_time = time.time() - experiment_start_time
    elapsed_hours = elapsed_time / 3600
    elapsed_minutes = (elapsed_time % 3600) / 60
    
    # Calculate ETA
    if videos_completed > 0:
        avg_time_per_video = elapsed_time / videos_completed
        eta_seconds = avg_time_per_video * (total_videos - videos_completed)
        eta_hours = int(eta_seconds // 3600)
        eta_minutes = int((eta_seconds % 3600) // 60)
    else:
        eta_hours = 0
        eta_minutes = 0
    
    # Calculate success rate and costs
    if os.path.exists(RESULTS_CSV_FILE):
        df = pd.read_csv(RESULTS_CSV_FILE)
        success_count = len(df[df['status'] == 'SUCCESS'])
        success_rate = (success_count / videos_completed * 100) if videos_completed > 0 else 0
        
        # Calculate costs
        successful_results = df[df['status'] == 'SUCCESS']
        if len(successful_results) > 0:
            total_cost = successful_results['compute_cost_usd'].sum()
            avg_cost_per_video = total_cost / len(successful_results)
        else:
            total_cost = 0
            avg_cost_per_video = 0
        
        # Find best model so far (filter out failed runs)
        if len(successful_results) > 0:
            successful_results['cost_efficiency'] = 1 / (successful_results['inference_time_seconds'] * successful_results['peak_memory_gb'] * POWER_CONSUMPTION_WATTS)
            best_model = successful_results.groupby('model_name')['cost_efficiency'].mean().idxmax()
            best_efficiency = successful_results.groupby('model_name')['cost_efficiency'].mean().max()
        else:
            best_model = "N/A"
            best_efficiency = 0
    else:
        success_rate = 100
        best_model = "N/A"
        best_efficiency = 0
        total_cost = 0
        avg_cost_per_video = 0
    
    summary = f"""
{'='*60}
PERIODIC SUMMARY (Every 3 videos)
{'='*60}
Total videos completed: {videos_completed}/{total_videos}
Time spent so far: {int(elapsed_hours)}h {int(elapsed_minutes)}m
Estimated time remaining: {eta_hours}h {eta_minutes}m
Success rate: {success_rate:.1f}%
Total Compute Cost: ${total_cost:.2f}
Avg Cost/Video: ${avg_cost_per_video:.4f}
Best model so far: {best_model} (Cost-Efficiency: {best_efficiency:.6f})
{'='*60}
"""
    
    print(summary)
    
    # Save to summary.txt
    with open(SUMMARY_FILE, 'a') as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
        f.write(summary)

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
            video = pipeline(
                prompt=prompt,
                num_frames=NUM_FRAMES,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=num_steps,
            ).frames[0]
        except Exception as e:
            print(f"Error during inference: {e}")
            raise
    
    elapsed = time.time() - start_time
    inference_time = elapsed
    compute_cost_usd = (elapsed / 3600) * GPU_HOURLY_COST
    
    # Measure peak GPU memory
    peak_memory_bytes = torch.cuda.max_memory_allocated()
    peak_memory_gb = peak_memory_bytes / (1024 ** 3)
    
    # Move video frames to CPU and convert to uint8 to free GPU memory before export
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()

    if isinstance(video, list):
        converted_frames = []
        for frame in video:
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            else:
                frame = np.asarray(frame)

            if frame.dtype != np.uint8:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)

            converted_frames.append(frame)
        video = converted_frames
    else:
        # Ensure proper dtype (uint8) for video export
        if video.dtype != np.uint8:
            video = (video * 255).clip(0, 255).astype(np.uint8)
    
    # Clear GPU memory immediately after moving frames to CPU
    torch.cuda.empty_cache()
    
    # Save video to output folder
    model_short_name = model_name.split('/')[-1]
    output_path = f"{OUTPUT_VIDEO_DIR}/{model_short_name}/prompt_{prompt_index}/run_{run_number}.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    export_to_video(video, output_path, fps=8)
    
    result = {
        'model_name': model_name,
        'run_number': run_number,
        'prompt_index': prompt_index,
        'prompt_text': prompt,
        'inference_time_seconds': inference_time,
        'peak_memory_gb': peak_memory_gb,
        'compute_cost_usd': compute_cost_usd,
        'status': 'SUCCESS'
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
        if hasattr(pipeline, "enable_sequential_cpu_offload"):
            pipeline.enable_sequential_cpu_offload()
        elif hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to("cuda")

        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()

        if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
        
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
    Run the complete benchmarking experiment for all models with checkpoint support.
    """
    global videos_completed, experiment_start_time, results
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    
    # Load existing results from CSV if resuming
    if checkpoint and os.path.exists(RESULTS_CSV_FILE):
        print("="*60)
        print("RESUMING FROM CHECKPOINT")
        print("="*60)
        print(f"Last completed: Model {checkpoint['current_model']}")
        print(f"Videos completed: {checkpoint['videos_completed']}/{total_videos}")
        print(f"Resuming from video {checkpoint['videos_completed'] + 1}")
        print("="*60)
        
        # Load existing results
        df = pd.read_csv(RESULTS_CSV_FILE)
        results = df.to_dict('records')
        videos_completed = checkpoint['videos_completed']
        
        # Resume from checkpoint position
        start_model_idx = checkpoint['model_idx']
        start_prompt_idx = checkpoint['prompt_idx']
        start_run = checkpoint['run_number'] + 1
    else:
        print("="*60)
        print("COGVIDEOX COST-EFFICIENCY BENCHMARKING EXPERIMENT")
        print("="*60)
        print(f"Models to test: {len(MODELS)}")
        print(f"Prompts per model: {len(PROMPTS)}")
        print(f"Runs per prompt: {NUM_RUNS_PER_PROMPT}")
        print(f"Total videos to generate: {total_videos}")
        print(f"Video settings: {NUM_FRAMES} frames, {HEIGHT}x{WIDTH}, {NUM_INFERENCE_STEPS} steps")
        print("="*60)
        
        start_model_idx = 0
        start_prompt_idx = 0
        start_run = 1
    
    # Start experiment timer
    experiment_start_time = time.time()
    
    # Iterate through models
    for model_idx, model_name in enumerate(MODELS):
        # Skip models already completed
        if model_idx < start_model_idx:
            continue
        
        # Determine number of inference steps
        num_steps = NUM_INFERENCE_STEPS
        
        try:
            # Load model
            pipeline = load_model(model_name)
            
            # Run experiments for each prompt
            for prompt_idx, prompt in enumerate(PROMPTS):
                # Skip prompts already completed
                if model_idx == start_model_idx and prompt_idx < start_prompt_idx:
                    continue
                
                print(f"\nPrompt {prompt_idx + 1}/{len(PROMPTS)}: \"{prompt}\"")
                print("-" * 60)
                
                # Determine starting run number
                run_start = start_run if (model_idx == start_model_idx and prompt_idx == start_prompt_idx) else 1
                
                for run in range(run_start, NUM_RUNS_PER_PROMPT + 1):
                    try:
                        # Measure inference
                        result = measure_inference(
                            pipeline=pipeline,
                            prompt=prompt,
                            model_name=model_name,
                            run_number=run,
                            prompt_index=prompt_idx + 1,
                            num_steps=num_steps
                        )
                        
                        # Save result immediately
                        results.append(result)
                        append_to_csv(result)
                        
                        # Update progress tracking
                        videos_completed += 1
                        
                        # Log progress
                        log_progress(model_name, run, prompt_idx + 1, 
                                   result['inference_time_seconds'], 
                                   result['peak_memory_gb'])
                        
                        # Save checkpoint after every video
                        save_checkpoint(model_name, model_idx, prompt_idx, run)
                        
                        # Print periodic summary every 3 videos
                        if videos_completed % 3 == 0:
                            print_periodic_summary()
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            # Log error
                            log_error(f"OOM Error: {str(e)}", model_name, run, prompt_idx + 1)
                            
                            # Reduce steps and retry
                            print("\n⚠ Memory error detected. Reducing steps to 30 and retrying...")
                            num_steps = 30
                            cleanup_model(pipeline)
                            torch.cuda.empty_cache()
                            if hasattr(torch.cuda, "ipc_collect"):
                                torch.cuda.ipc_collect()
                            gc.collect()
                            pipeline = load_model(model_name)
                            
                            try:
                                result = measure_inference(
                                    pipeline=pipeline,
                                    prompt=prompt,
                                    model_name=model_name,
                                    run_number=run,
                                    prompt_index=prompt_idx + 1,
                                    num_steps=num_steps
                                )
                                
                                # Save result
                                results.append(result)
                                append_to_csv(result)
                                videos_completed += 1
                                
                                log_progress(model_name, run, prompt_idx + 1,
                                           result['inference_time_seconds'],
                                           result['peak_memory_gb'])
                                
                                save_checkpoint(model_name, model_idx, prompt_idx, run)
                                
                                if videos_completed % 3 == 0:
                                    print_periodic_summary()
                                    
                            except Exception as retry_error:
                                # Log failure and skip this video
                                log_error(f"Retry failed: {str(retry_error)}", model_name, run, prompt_idx + 1)
                                
                                # Mark as FAILED in CSV
                                failed_result = {
                                    'model_name': model_name,
                                    'run_number': run,
                                    'prompt_index': prompt_idx + 1,
                                    'prompt_text': prompt,
                                    'inference_time_seconds': 0,
                                    'peak_memory_gb': 0,
                                    'compute_cost_usd': 0,
                                    'status': 'FAILED'
                                }
                                append_to_csv(failed_result)
                                videos_completed += 1
                                save_checkpoint(model_name, model_idx, prompt_idx, run)
                                
                                print(f"✗ Skipping video {videos_completed}/{total_videos} due to error")
                                continue
                        else:
                            # Log other runtime errors and skip
                            log_error(f"Runtime Error: {str(e)}", model_name, run, prompt_idx + 1)
                            
                            failed_result = {
                                'model_name': model_name,
                                'run_number': run,
                                'prompt_index': prompt_idx + 1,
                                'prompt_text': prompt,
                                'inference_time_seconds': 0,
                                'peak_memory_gb': 0,
                                'compute_cost_usd': 0,
                                'status': 'FAILED'
                            }
                            append_to_csv(failed_result)
                            videos_completed += 1
                            save_checkpoint(model_name, model_idx, prompt_idx, run)
                            
                            print(f"✗ Skipping video {videos_completed}/{total_videos} due to error")
                            continue
                    
                    except Exception as e:
                        # Log general errors and skip
                        log_error(f"General Error: {str(e)}", model_name, run, prompt_idx + 1)
                        
                        failed_result = {
                            'model_name': model_name,
                            'run_number': run,
                            'prompt_index': prompt_idx + 1,
                            'prompt_text': prompt,
                            'inference_time_seconds': 0,
                            'peak_memory_gb': 0,
                            'compute_cost_usd': 0,
                            'status': 'FAILED'
                        }
                        append_to_csv(failed_result)
                        videos_completed += 1
                        save_checkpoint(model_name, model_idx, prompt_idx, run)
                        
                        print(f"✗ Skipping video {videos_completed}/{total_videos} due to error")
                        continue
            
            # Clean up model before loading next one
            cleanup_model(pipeline)
            
        except Exception as e:
            log_error(f"Model loading error: {str(e)}", model_name, 0, 0)
            print(f"\n✗ Error with model {model_name}: {e}")
            print("Continuing with next model...\n")
            continue

def calculate_statistics():
    """
    Calculate and display statistics for all models.
    """
    # Load results from CSV (in case of resume)
    if os.path.exists(RESULTS_CSV_FILE):
        df = pd.read_csv(RESULTS_CSV_FILE)
    else:
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_CSV_FILE, index=False)
    
    # Filter out failed videos for statistics
    df_success = df[df['status'] == 'SUCCESS']
    
    print("\n" + "="*60)
    print("✓ Results saved to results.csv")
    print(f"Total videos: {len(df)} | Successful: {len(df_success)} | Failed: {len(df) - len(df_success)}")
    print("="*60)
    
    # Calculate statistics per model
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    stats_data = []
    
    for model_name in df_success['model_name'].unique():
        model_df = df_success[df_success['model_name'] == model_name]
        
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
    if len(df_success['model_name'].unique()) == 2:
        print("\n" + "="*60)
        print("STATISTICAL COMPARISON (T-TEST)")
        print("="*60)
        
        model1_name = MODELS[0]
        model2_name = MODELS[1]
        
        model1_times = df_success[df_success['model_name'] == model1_name]['inference_time_seconds']
        model2_times = df_success[df_success['model_name'] == model2_name]['inference_time_seconds']
        
        model1_memory = df_success[df_success['model_name'] == model1_name]['peak_memory_gb']
        model2_memory = df_success[df_success['model_name'] == model2_name]['peak_memory_gb']
        
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
