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
from torchmetrics.multimodal.clip_score import CLIPScore

warnings.filterwarnings('ignore')

# Configuration
MODELS = [
    "THUDM/CogVideoX-5b",
    "THUDM/CogVideoX-2b",
    "THUDM/CogVideoX1.5-5B"
]

# Model-specific video settings (frames, height, width)
MODEL_CONFIGS = {
    "THUDM/CogVideoX-5b": {"num_frames": 49, "height": 480, "width": 720},
    "THUDM/CogVideoX-2b": {"num_frames": 49, "height": 480, "width": 720},
    "THUDM/CogVideoX1.5-5B": {"num_frames": 81, "height": 768, "width": 1360},
}

PROMPTS = [
    "A person walking in a park on a sunny day",
    "A car driving on a highway with trees in background",
    "A cat playing with a ball in a living room"
]

NUM_RUNS_PER_PROMPT = 5
NUM_FRAMES_DEFAULT = 49
HEIGHT_DEFAULT = 480
WIDTH_DEFAULT = 720
NUM_INFERENCE_STEPS = 50
POWER_CONSUMPTION_WATTS = 200
GPU_HOURLY_COST = 1.18

# File paths for checkpointing and logging
CHECKPOINT_FILE = "checkpoint_v4.json"
PROGRESS_LOG_FILE = "progress_log_v4.txt"
RESULTS_CSV_FILE = "results_v4.csv"
SUMMARY_FILE = "summary_v4.txt"
ERROR_LOG_FILE = "errors_v4.log"
OUTPUT_VIDEO_DIR = "output_videos_cogvideox_v4"

# Global tracking variables
results = []
total_videos = len(MODELS) * len(PROMPTS) * NUM_RUNS_PER_PROMPT
videos_completed = 0
experiment_start_time = None
clip_metric = None

def calculate_frame_consistency(frames):
    """
    Calculate frame consistency (motion smoothness) score.
    Higher score = smoother motion, lower = jittery/flickering.
    
    Args:
        frames: List of numpy arrays (video frames)
    
    Returns:
        float: Consistency score between 0 and 1
    """
    if len(frames) < 2:
        return 1.0
    
    frame_diffs = []
    for i in range(len(frames) - 1):
        diff = np.mean(np.abs(frames[i+1].astype(np.float32) - frames[i].astype(np.float32)))
        frame_diffs.append(diff)
    
    std_diff = np.std(frame_diffs)
    consistency = 1.0 / (1.0 + std_diff)
    return float(consistency)

def calculate_clip_score(frame, prompt_text):
    """
    Calculate CLIP score for text-video alignment.
    
    Args:
        frame: numpy array (single video frame, HxWxC, uint8)
        prompt_text: str (the text prompt)
    
    Returns:
        float: CLIP score (0-100 scale)
    """
    global clip_metric
    
    if clip_metric is None:
        return 0.0
    
    try:
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
        else:
            frame_tensor = frame
        
        frame_tensor = frame_tensor.unsqueeze(0)
        
        score = clip_metric(frame_tensor, [prompt_text])
        return float(score.detach().cpu().item())
    except Exception as e:
        print(f"Warning: CLIP score calculation failed: {e}")
        return 0.0

def save_checkpoint(model_name, model_idx, prompt_idx, run_number):
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
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None

def log_progress(model_name, run_number, prompt_index, inference_time, peak_memory_gb, clip_score=0.0, frame_consistency=0.0):
    global videos_completed, experiment_start_time
    
    elapsed_time = time.time() - experiment_start_time
    videos_remaining = total_videos - videos_completed
    
    if videos_completed > 0:
        avg_time_per_video = elapsed_time / videos_completed
        eta_seconds = avg_time_per_video * videos_remaining
        eta_hours = eta_seconds / 3600
    else:
        eta_hours = 0
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    progress_pct = (videos_completed / total_videos) * 100
    
    log_line = (f"[{timestamp}] [{progress_pct:.1f}%] Model: {model_name} | Run: {run_number}/{NUM_RUNS_PER_PROMPT} | "
                f"Prompt: {prompt_index}/{len(PROMPTS)} | Time: {inference_time:.2f}s | "
                f"Memory: {peak_memory_gb:.2f}GB | CLIP: {clip_score:.1f} | Consist: {frame_consistency:.3f} | ETA: {eta_hours:.1f}h\n")
    
    with open(PROGRESS_LOG_FILE, 'a') as f:
        f.write(log_line)
    
    print(log_line.strip())

def log_error(error_message, model_name, run_number, prompt_index):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_entry = (f"[{timestamp}] Model: {model_name} | Run: {run_number} | "
                   f"Prompt: {prompt_index} | Error: {error_message}\n")
    
    with open(ERROR_LOG_FILE, 'a') as f:
        f.write(error_entry)
    
    print(f"✗ ERROR LOGGED: {error_message}")

def append_to_csv(result):
    df = pd.DataFrame([result])
    
    file_exists = os.path.exists(RESULTS_CSV_FILE)
    
    df.to_csv(RESULTS_CSV_FILE, mode='a', header=not file_exists, index=False)

def print_periodic_summary():
    global videos_completed, experiment_start_time
    
    elapsed_time = time.time() - experiment_start_time
    elapsed_hours = elapsed_time / 3600
    elapsed_minutes = (elapsed_time % 3600) / 60
    
    if videos_completed > 0:
        avg_time_per_video = elapsed_time / videos_completed
        eta_seconds = avg_time_per_video * (total_videos - videos_completed)
        eta_hours = int(eta_seconds // 3600)
        eta_minutes = int((eta_seconds % 3600) // 60)
    else:
        eta_hours = 0
        eta_minutes = 0
    
    if os.path.exists(RESULTS_CSV_FILE):
        df = pd.read_csv(RESULTS_CSV_FILE)
        success_count = len(df[df['status'] == 'SUCCESS'])
        success_rate = (success_count / videos_completed * 100) if videos_completed > 0 else 0
        
        successful_results = df[df['status'] == 'SUCCESS']
        if len(successful_results) > 0:
            total_cost = successful_results['compute_cost_usd'].sum()
            avg_cost_per_video = total_cost / len(successful_results)
        else:
            total_cost = 0
            avg_cost_per_video = 0
        
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
    
    with open(SUMMARY_FILE, 'a') as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
        f.write(summary)

def measure_inference(pipeline, prompt, model_name, run_number, prompt_index, num_steps=50):
    model_config = MODEL_CONFIGS.get(model_name, {
        "num_frames": NUM_FRAMES_DEFAULT,
        "height": HEIGHT_DEFAULT,
        "width": WIDTH_DEFAULT
    })
    num_frames = model_config["num_frames"]
    height = model_config["height"]
    width = model_config["width"]
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    start_time = time.time()
    
    with torch.no_grad():
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            video = pipeline(
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_steps,
            ).frames[0]
        except Exception as e:
            print(f"Error during inference: {e}")
            raise
    
    elapsed = time.time() - start_time
    inference_time = elapsed
    compute_cost_usd = (elapsed / 3600) * GPU_HOURLY_COST
    
    peak_memory_bytes = torch.cuda.max_memory_allocated()
    peak_memory_gb = peak_memory_bytes / (1024 ** 3)
    
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
                if frame.max() <= 1.0:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                else:
                    frame = frame.clip(0, 255).astype(np.uint8)

            converted_frames.append(frame)
        video = converted_frames
    else:
        if video.dtype != np.uint8:
            if video.max() <= 1.0:
                video = (video * 255).clip(0, 255).astype(np.uint8)
            else:
                video = video.clip(0, 255).astype(np.uint8)
    
    torch.cuda.empty_cache()
    
    # Calculate quality metrics (after inference timing)
    if isinstance(video, list):
        frames_for_quality = video
    else:
        frames_for_quality = [video[i] for i in range(video.shape[0])]
    
    # CLIP Score: use middle frame
    middle_idx = len(frames_for_quality) // 2
    middle_frame = frames_for_quality[middle_idx]
    clip_score = calculate_clip_score(middle_frame, prompt)
    
    # Frame Consistency: use all frames
    frame_consistency = calculate_frame_consistency(frames_for_quality)
    
    # Determine model short name for output path
    if "1.5" in model_name:
        model_short_name = "CogVideoX1.5-5B"
    elif "5b" in model_name.lower():
        model_short_name = "CogVideoX-5b"
    elif "2b" in model_name.lower():
        model_short_name = "CogVideoX-2b"
    else:
        model_short_name = "CogVideoX"
    
    output_path = f"{OUTPUT_VIDEO_DIR}/{model_short_name}/prompt_{prompt_index}/run_{run_number}.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        export_to_video(video, output_path, fps=8, video_codec='libx264', pix_fmt='yuv420p')
    except TypeError:
        export_to_video(video, output_path, fps=8)
    
    result = {
        'model_name': model_name,
        'run_number': run_number,
        'prompt_index': prompt_index,
        'prompt_text': prompt,
        'inference_time_seconds': inference_time,
        'peak_memory_gb': peak_memory_gb,
        'compute_cost_usd': compute_cost_usd,
        'clip_score': clip_score,
        'frame_consistency': frame_consistency,
        'status': 'SUCCESS'
    }
    
    print(f"✓ {model_name} | Prompt {prompt_index} | Run {run_number} | "
          f"Time: {inference_time:.2f}s | Memory: {peak_memory_gb:.2f}GB | "
          f"CLIP: {clip_score:.1f} | Consistency: {frame_consistency:.3f}")
    
    return result

def load_model(model_name):
    try:
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name}")
        print(f"{'='*60}")
        
        pipeline = CogVideoXPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        )
        if hasattr(pipeline, "enable_sequential_cpu_offload"):
            pipeline.enable_sequential_cpu_offload()
        elif hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to("cuda")

        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()

        if hasattr(pipeline, "vae"):
            if hasattr(pipeline.vae, 'enable_tiling'):
                pipeline.vae.enable_tiling()
        
        print("✓ Model loaded successfully\n")
        return pipeline
        
    except Exception as e:
        print(f"✗ Failed to load model {model_name}: {e}")
        raise

def cleanup_model(pipeline):
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    print("\n✓ Model cleaned up from memory\n")

def run_experiments():
    global videos_completed, experiment_start_time, results
    
    checkpoint = load_checkpoint()
    
    if checkpoint and os.path.exists(RESULTS_CSV_FILE):
        print("="*60)
        print("RESUMING FROM CHECKPOINT")
        print("="*60)
        print(f"Last completed: Model {checkpoint['current_model']}")
        print(f"Videos completed: {checkpoint['videos_completed']}/{total_videos}")
        print(f"Resuming from video {checkpoint['videos_completed'] + 1}")
        print("="*60)
        
        df = pd.read_csv(RESULTS_CSV_FILE)
        results = df.to_dict('records')
        videos_completed = checkpoint['videos_completed']
        
        start_model_idx = checkpoint['model_idx']
        start_prompt_idx = checkpoint['prompt_idx']
        start_run = checkpoint['run_number'] + 1
    else:
        print("="*60)
        print("COGVIDEOX V4 COST-EFFICIENCY BENCHMARKING EXPERIMENT")
        print("="*60)
        print(f"Models to test: {len(MODELS)}")
        print(f"Prompts per model: {len(PROMPTS)}")
        print(f"Runs per prompt: {NUM_RUNS_PER_PROMPT}")
        print(f"Total videos to generate: {total_videos}")
        print(f"Video settings: Model-specific resolution, {NUM_INFERENCE_STEPS} steps")
        print("Quality metrics: CLIP Score + Frame Consistency enabled")
        print("="*60)
        
        start_model_idx = 0
        start_prompt_idx = 0
        start_run = 1
    
    experiment_start_time = time.time()
    
    for model_idx, model_name in enumerate(MODELS):
        if model_idx < start_model_idx:
            continue
        
        num_steps = NUM_INFERENCE_STEPS
        
        try:
            pipeline = load_model(model_name)
            
            for prompt_idx, prompt in enumerate(PROMPTS):
                if model_idx == start_model_idx and prompt_idx < start_prompt_idx:
                    continue
                
                print(f"\nPrompt {prompt_idx + 1}/{len(PROMPTS)}: \"{prompt}\"")
                print("-" * 60)
                
                run_start = start_run if (model_idx == start_model_idx and prompt_idx == start_prompt_idx) else 1
                
                for run in range(run_start, NUM_RUNS_PER_PROMPT + 1):
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
                        append_to_csv(result)
                        
                        videos_completed += 1
                        
                        log_progress(model_name, run, prompt_idx + 1, 
                                   result['inference_time_seconds'], 
                                   result['peak_memory_gb'],
                                   result['clip_score'],
                                   result['frame_consistency'])
                        
                        save_checkpoint(model_name, model_idx, prompt_idx, run)
                        
                        if videos_completed % 3 == 0:
                            print_periodic_summary()
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            log_error(f"OOM Error: {str(e)}", model_name, run, prompt_idx + 1)
                            
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
                                
                                results.append(result)
                                append_to_csv(result)
                                videos_completed += 1
                                
                                log_progress(model_name, run, prompt_idx + 1,
                                           result['inference_time_seconds'],
                                           result['peak_memory_gb'],
                                           result['clip_score'],
                                           result['frame_consistency'])
                                
                                save_checkpoint(model_name, model_idx, prompt_idx, run)
                                
                                if videos_completed % 3 == 0:
                                    print_periodic_summary()
                                    
                            except Exception as retry_error:
                                log_error(f"Retry failed: {str(retry_error)}", model_name, run, prompt_idx + 1)
                                
                                failed_result = {
                                    'model_name': model_name,
                                    'run_number': run,
                                    'prompt_index': prompt_idx + 1,
                                    'prompt_text': prompt,
                                    'inference_time_seconds': 0,
                                    'peak_memory_gb': 0,
                                    'compute_cost_usd': 0,
                                    'clip_score': 0,
                                    'frame_consistency': 0,
                                    'status': 'FAILED'
                                }
                                append_to_csv(failed_result)
                                videos_completed += 1
                                save_checkpoint(model_name, model_idx, prompt_idx, run)
                                
                                print(f"✗ Skipping video {videos_completed}/{total_videos} due to error")
                                continue
                        else:
                            log_error(f"Runtime Error: {str(e)}", model_name, run, prompt_idx + 1)
                            
                            failed_result = {
                                'model_name': model_name,
                                'run_number': run,
                                'prompt_index': prompt_idx + 1,
                                'prompt_text': prompt,
                                'inference_time_seconds': 0,
                                'peak_memory_gb': 0,
                                'compute_cost_usd': 0,
                                'clip_score': 0,
                                'frame_consistency': 0,
                                'status': 'FAILED'
                            }
                            append_to_csv(failed_result)
                            videos_completed += 1
                            save_checkpoint(model_name, model_idx, prompt_idx, run)
                            
                            print(f"✗ Skipping video {videos_completed}/{total_videos} due to error")
                            continue
                    
                    except Exception as e:
                        log_error(f"General Error: {str(e)}", model_name, run, prompt_idx + 1)
                        
                        failed_result = {
                            'model_name': model_name,
                            'run_number': run,
                            'prompt_index': prompt_idx + 1,
                            'prompt_text': prompt,
                            'inference_time_seconds': 0,
                            'peak_memory_gb': 0,
                            'compute_cost_usd': 0,
                            'clip_score': 0,
                            'frame_consistency': 0,
                            'status': 'FAILED'
                        }
                        append_to_csv(failed_result)
                        videos_completed += 1
                        save_checkpoint(model_name, model_idx, prompt_idx, run)
                        
                        print(f"✗ Skipping video {videos_completed}/{total_videos} due to error")
                        continue
            
            cleanup_model(pipeline)
            
        except Exception as e:
            log_error(f"Model loading error: {str(e)}", model_name, 0, 0)
            print(f"\n✗ Error with model {model_name}: {e}")
            print("Continuing with next model...\n")
            continue

def calculate_statistics():
    if os.path.exists(RESULTS_CSV_FILE):
        df = pd.read_csv(RESULTS_CSV_FILE)
    else:
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_CSV_FILE, index=False)
    
    df_success = df[df['status'] == 'SUCCESS']
    
    print("\n" + "="*60)
    print("✓ Results saved to results_v4.csv")
    print(f"Total videos: {len(df)} | Successful: {len(df_success)} | Failed: {len(df) - len(df_success)}")
    print("="*60)
    
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
        
        avg_clip = model_df['clip_score'].mean()
        std_clip = model_df['clip_score'].std()
        
        avg_consistency = model_df['frame_consistency'].mean()
        std_consistency = model_df['frame_consistency'].std()
        
        avg_cost = model_df['compute_cost_usd'].mean()
        
        # New cost-efficiency formula: (CLIP × Consistency) / (Time × Cost)
        # Higher quality + lower cost = better efficiency
        if avg_time > 0 and avg_cost > 0:
            cost_efficiency = (avg_clip * avg_consistency) / (avg_time * avg_cost)
        else:
            cost_efficiency = 0
        
        stats_data.append({
            'Model': model_name,
            'Avg Time (s)': f"{avg_time:.2f} ± {std_time:.2f}",
            'Avg Memory (GB)': f"{avg_memory:.2f} ± {std_memory:.2f}",
            'CLIP Score': f"{avg_clip:.1f} ± {std_clip:.1f}",
            'Frame Consist.': f"{avg_consistency:.3f} ± {std_consistency:.3f}",
            'Cost-Efficiency': f"{cost_efficiency:.4f}"
        })
        
        print(f"\n{model_name}:")
        print(f"  Average Time: {avg_time:.2f} ± {std_time:.2f} seconds")
        print(f"  Average Memory: {avg_memory:.2f} ± {std_memory:.2f} GB")
        print(f"  CLIP Score: {avg_clip:.1f} ± {std_clip:.1f}")
        print(f"  Frame Consistency: {avg_consistency:.3f} ± {std_consistency:.3f}")
        print(f"  Avg Cost/Video: ${avg_cost:.4f}")
        print(f"  Cost-Efficiency (CLIP×Consist)/(Time×Cost): {cost_efficiency:.4f}")
    
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    
    # Pairwise t-tests for all model combinations
    unique_models = df_success['model_name'].unique()
    if len(unique_models) >= 2:
        print("\n" + "="*60)
        print("STATISTICAL COMPARISON (PAIRWISE T-TESTS)")
        print("="*60)
        
        from itertools import combinations
        
        for model1_name, model2_name in combinations(unique_models, 2):
            print(f"\n--- {model1_name} vs {model2_name} ---")
            
            model1_data = df_success[df_success['model_name'] == model1_name]
            model2_data = df_success[df_success['model_name'] == model2_name]
            
            # Inference Time
            t_stat, p_val = stats.ttest_ind(model1_data['inference_time_seconds'], model2_data['inference_time_seconds'])
            sig = "*" if p_val < 0.05 else ""
            print(f"  Inference Time: t={t_stat:.3f}, p={p_val:.4f} {sig}")
            
            # Memory
            t_stat, p_val = stats.ttest_ind(model1_data['peak_memory_gb'], model2_data['peak_memory_gb'])
            sig = "*" if p_val < 0.05 else ""
            print(f"  Memory Usage: t={t_stat:.3f}, p={p_val:.4f} {sig}")
            
            # CLIP Score
            t_stat, p_val = stats.ttest_ind(model1_data['clip_score'], model2_data['clip_score'])
            sig = "*" if p_val < 0.05 else ""
            print(f"  CLIP Score: t={t_stat:.3f}, p={p_val:.4f} {sig}")
            
            # Frame Consistency
            t_stat, p_val = stats.ttest_ind(model1_data['frame_consistency'], model2_data['frame_consistency'])
            sig = "*" if p_val < 0.05 else ""
            print(f"  Frame Consistency: t={t_stat:.3f}, p={p_val:.4f} {sig}")
        
        print("\n  (* = statistically significant at p < 0.05)")
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)

def main():
    global clip_metric
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This experiment requires a GPU.")
        return
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n")
    
    # Initialize CLIP metric for quality evaluation
    print("Initializing CLIP metric for quality evaluation...")
    try:
        clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")
        clip_metric = clip_metric.to("cpu")  # Keep on CPU to save GPU memory
        print("✓ CLIP metric initialized successfully\n")
    except Exception as e:
        print(f"⚠ Warning: Failed to initialize CLIP metric: {e}")
        print("Continuing without CLIP score evaluation...\n")
        clip_metric = None
    
    run_experiments()
    
    if len(results) > 0:
        calculate_statistics()
    else:
        print("\n✗ No results collected. Experiment failed.")

if __name__ == "__main__":
    main()
