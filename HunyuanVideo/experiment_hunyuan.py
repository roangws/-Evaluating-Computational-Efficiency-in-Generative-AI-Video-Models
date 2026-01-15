import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import time
import gc
import pandas as pd
from diffusers import HunyuanVideo15Pipeline
from diffusers.utils import export_to_video
import warnings
import json
from datetime import datetime
import numpy as np
import cv2
from torchmetrics.multimodal.clip_score import CLIPScore

warnings.filterwarnings('ignore')

# Configuration
MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"

PROMPTS = [
    "A person walking in a park on a sunny day",
    "A car driving on a highway with trees in background",
    "A cat playing with a ball in a living room"
]

NUM_RUNS_PER_PROMPT = 3
NUM_FRAMES = 49
FPS = 24
NUM_INFERENCE_STEPS = 50
BASE_SEED = 1000
GPU_HOURLY_COST = 7.52
HEIGHT = 480
WIDTH = 720
GUIDANCE_SCALE = 6.0

# File paths
CHECKPOINT_FILE = "checkpoint.json"
PROGRESS_LOG_FILE = "progress_log.txt"
RESULTS_CSV_FILE = "results_hunyuan.csv"
OUTPUT_VIDEO_DIR = "output_videos"
ERROR_LOG_FILE = "errors.log"

# Global tracking
results = []
total_videos = len(PROMPTS) * NUM_RUNS_PER_PROMPT
videos_completed = 0
experiment_start_time = None
clip_metric = None
SANITY_ONLY = True

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

def save_checkpoint(prompt_idx, run_number):
    checkpoint_data = {
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

def log_progress(run_number, prompt_index, inference_time, peak_memory_gb, clip_score=0.0, frame_consistency=0.0):
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
    
    log_line = (f"[{timestamp}] [{progress_pct:.1f}%] HunyuanVideo-1.5 | Run: {run_number}/{NUM_RUNS_PER_PROMPT} | "
                f"Prompt: {prompt_index}/{len(PROMPTS)} | Time: {inference_time:.2f}s | "
                f"Memory: {peak_memory_gb:.2f}GB | CLIP: {clip_score:.1f} | Consist: {frame_consistency:.3f} | ETA: {eta_hours:.1f}h\n")
    
    with open(PROGRESS_LOG_FILE, 'a') as f:
        f.write(log_line)
    
    print(log_line.strip())

def log_error(error_message, run_number, prompt_index):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_entry = (f"[{timestamp}] HunyuanVideo-1.5 | Run: {run_number} | "
                   f"Prompt: {prompt_index} | Error: {error_message}\n")
    
    with open(ERROR_LOG_FILE, 'a') as f:
        f.write(error_entry)
    
    print(f"✗ ERROR LOGGED: {error_message}")

def append_to_csv(result):
    df = pd.DataFrame([result])
    
    file_exists = os.path.exists(RESULTS_CSV_FILE)
    
    df.to_csv(RESULTS_CSV_FILE, mode='a', header=not file_exists, index=False)

def measure_inference(pipeline, prompt, run_number, prompt_index):
    # Aggressive memory cleanup to prevent fragmentation
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.synchronize()
    
    start_time = time.time()
    
    with torch.no_grad():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            
            seed = BASE_SEED * prompt_index + run_number
            generator = torch.Generator(device="cuda").manual_seed(seed)
            
            video = pipeline(
                prompt=prompt,
                num_frames=NUM_FRAMES,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=NUM_INFERENCE_STEPS,
                generator=generator,
            ).frames[0]
            
            # DEBUG: Check pipeline output range
            if isinstance(video, torch.Tensor):
                print(f"[DEBUG-COLOR-RANGE] Pipeline output: dtype={video.dtype}, min={video.min():.4f}, max={video.max():.4f}")
            elif isinstance(video, list) and len(video) > 0:
                first_frame = video[0]
                if isinstance(first_frame, torch.Tensor):
                    print(f"[DEBUG-COLOR-RANGE] Pipeline output (first frame): dtype={first_frame.dtype}, min={first_frame.min():.4f}, max={first_frame.max():.4f}")
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
        for idx, frame in enumerate(video):
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            else:
                frame = np.asarray(frame)
            
            # DEBUG: After numpy conversion (only first frame)
            if idx == 0:
                print(f"[DEBUG-COLOR-RANGE] After .numpy(): dtype={frame.dtype}, min={frame.min():.4f}, max={frame.max():.4f}")

            if frame.dtype != np.uint8:
                if frame.min() < 0 or frame.max() <= 1.0:
                    # VAE outputs [-1, 1], map to [0, 1] then to [0, 255]
                    scaled_frame = (frame + 1) / 2 * 255
                    if idx == 0:
                        print(f"[DEBUG-COLOR-RANGE] Scaling result: min={scaled_frame.min():.4f}, max={scaled_frame.max():.4f}")
                    frame = scaled_frame.clip(0, 255).astype(np.uint8)
                else:
                    # Already in [0, 255] range
                    frame = frame.clip(0, 255).astype(np.uint8)
                
                # DEBUG: After clip+uint8 (only first frame)
                if idx == 0:
                    print(f"[DEBUG-COLOR-RANGE] After clip+uint8: dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")
            
            # Convert BGR to RGB if needed
            if frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # DEBUG: After cvtColor (only first frame)
                if idx == 0:
                    print(f"[DEBUG-COLOR-RANGE] After cvtColor: min={frame.min()}, max={frame.max()}")
            
            # Invert pixel values to fix inverted colors from pipeline
            #line removed for fix the issue with the colors
            #frame = 255 - frame

            converted_frames.append(frame)
        video = converted_frames
    else:
        if video.dtype != np.uint8:
            if video.min() < 0 or video.max() <= 1.0:
                # VAE outputs [-1, 1], map to [0, 1] then to [0, 255]
                video = ((video + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            else:
                # Already in [0, 255] range
                video = video.clip(0, 255).astype(np.uint8)
    
    # Post-inference cleanup with sync
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    
    # Calculate quality metrics
    if isinstance(video, list):
        frames_for_quality = video
    else:
        frames_for_quality = [video[i] for i in range(video.shape[0])]
    
    # CLIP Score: use middle frame
    middle_idx = len(frames_for_quality) // 2
    middle_frame = frames_for_quality[middle_idx]
    
    try:
        clip_score = calculate_clip_score(middle_frame, prompt)
        if clip_score == 0.0:
            status = "CLIP_ERROR: Score is zero"
        else:
            status = "SUCCESS"
    except Exception as e:
        clip_score = 0.0
        status = f"CLIP_ERROR: {str(e)}"
    
    # Frame Consistency: use all frames
    frame_consistency = calculate_frame_consistency(frames_for_quality)
    
    output_path = f"{OUTPUT_VIDEO_DIR}/hunyuan_prompt_{prompt_index}/hunyuan_prompt_{prompt_index}_run_{run_number}.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        export_to_video(video, output_path, fps=FPS)
    except Exception as e:
        print(f"Warning: Video export failed: {e}")
        status = f"VIDEO_EXPORT_ERROR: {str(e)}"
    
    result = {
        'model_name': 'HunyuanVideo-1.5',
        'run_number': run_number,
        'prompt_index': prompt_index,
        'prompt_text': prompt,
        'inference_time_seconds': inference_time,
        'peak_memory_gb': peak_memory_gb,
        'compute_cost_usd': compute_cost_usd,
        'clip_score': clip_score,
        'frame_consistency': frame_consistency,
        'status': status
    }
    
    print(f"✓ HunyuanVideo-1.5 | Prompt {prompt_index} | Run {run_number} | "
          f"Time: {inference_time:.2f}s | Memory: {peak_memory_gb:.2f}GB | "
          f"CLIP: {clip_score:.1f} | Consistency: {frame_consistency:.3f} | Status: {status}")
    
    return result

def load_model():
    try:
        print(f"\n{'='*60}")
        print(f"Loading model: {MODEL_ID}")
        print(f"{'='*60}")
        
        pipeline = HunyuanVideo15Pipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16
        )
        
        pipeline.enable_model_cpu_offload()
        
        if hasattr(pipeline, "vae"):
            if hasattr(pipeline.vae, 'enable_tiling'):
                pipeline.vae.enable_tiling()
        
        # Configure guider for v1.5 guidance control
        pipeline.guider = pipeline.guider.new(guidance_scale=6.0)
        
        print("✓ Model loaded successfully with CPU offload and VAE tiling enabled\n")
        return pipeline
        
    except Exception as e:
        print(f"✗ Failed to load model {MODEL_ID}: {e}")
        raise

def cleanup_model(pipeline):
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    print("\n✓ Model cleaned up from memory\n")

def run_sanity_check(pipeline):
    """Run sanity check: generate only prompt 1 run 1"""
    global videos_completed
    
    print("\n" + "="*60)
    print("RUNNING SANITY CHECK (Prompt 1, Run 1)")
    print("="*60)
    
    prompt = PROMPTS[0]
    prompt_idx = 1
    run = 1
    
    try:
        result = measure_inference(
            pipeline=pipeline,
            prompt=prompt,
            run_number=run,
            prompt_index=prompt_idx
        )
        
        results.append(result)
        append_to_csv(result)
        videos_completed += 1
        
        log_progress(run, prompt_idx, 
                   result['inference_time_seconds'], 
                   result['peak_memory_gb'],
                   result['clip_score'],
                   result['frame_consistency'])
        
        save_checkpoint(prompt_idx - 1, run)
        
        # Check if sanity passed
        if result['status'] == 'SUCCESS' and result['clip_score'] > 0:
            print("\n" + "="*60)
            print("✓ SANITY CHECK PASSED")
            print(f"  CLIP Score: {result['clip_score']:.1f} (> 0)")
            print(f"  Status: {result['status']}")
            print("  Proceeding to full 9 runs...")
            print("="*60 + "\n")
            return True
        else:
            print("\n" + "="*60)
            print("✗ SANITY CHECK FAILED")
            print(f"  CLIP Score: {result['clip_score']:.1f}")
            print(f"  Status: {result['status']}")
            print("  Stopping experiment.")
            print("="*60 + "\n")
            return False
            
    except Exception as e:
        log_error(f"Sanity check error: {str(e)}", run, prompt_idx)
        print(f"\n✗ SANITY CHECK FAILED WITH ERROR: {e}\n")
        return False

def run_experiments():
    global videos_completed, experiment_start_time, results, SANITY_ONLY
    
    checkpoint = load_checkpoint()
    
    if checkpoint and os.path.exists(RESULTS_CSV_FILE):
        print("="*60)
        print("RESUMING FROM CHECKPOINT")
        print("="*60)
        print(f"Videos completed: {checkpoint['videos_completed']}/{total_videos}")
        print(f"Resuming from video {checkpoint['videos_completed'] + 1}")
        print("="*60)
        
        df = pd.read_csv(RESULTS_CSV_FILE)
        results = df.to_dict('records')
        videos_completed = checkpoint['videos_completed']
        
        start_prompt_idx = checkpoint['prompt_idx']
        start_run = checkpoint['run_number'] + 1
        SANITY_ONLY = False
    else:
        print("="*60)
        print("HUNYUANVIDEO-1.5 COST-EFFICIENCY BENCHMARKING EXPERIMENT")
        print("="*60)
        print(f"Model: {MODEL_ID}")
        print(f"Prompts: {len(PROMPTS)}")
        print(f"Runs per prompt: {NUM_RUNS_PER_PROMPT}")
        print(f"Total videos to generate: {total_videos}")
        print(f"Video settings: {NUM_FRAMES} frames @ {FPS}fps, {NUM_INFERENCE_STEPS} steps")
        print("Quality metrics: CLIP Score + Frame Consistency enabled")
        print(f"Sanity check mode: {SANITY_ONLY}")
        print("="*60)
        
        start_prompt_idx = 0
        start_run = 1
    
    experiment_start_time = time.time()
    
    try:
        pipeline = load_model()
        
        # Run sanity check first if starting fresh
        if SANITY_ONLY and videos_completed == 0:
            sanity_passed = run_sanity_check(pipeline)
            if not sanity_passed:
                print("\n✗ Experiment stopped due to failed sanity check.")
                return
            # If passed, continue from prompt 1 run 2 (sanity was run 1)
            start_prompt_idx = 0
            start_run = 2
        
        for prompt_idx, prompt in enumerate(PROMPTS):
            if prompt_idx < start_prompt_idx:
                continue
            
            print(f"\nPrompt {prompt_idx + 1}/{len(PROMPTS)}: \"{prompt}\"")
            print("-" * 60)
            
            run_start = start_run if (prompt_idx == start_prompt_idx) else 1
            
            for run in range(run_start, NUM_RUNS_PER_PROMPT + 1):
                try:
                    result = measure_inference(
                        pipeline=pipeline,
                        prompt=prompt,
                        run_number=run,
                        prompt_index=prompt_idx + 1
                    )
                    
                    results.append(result)
                    append_to_csv(result)
                    
                    videos_completed += 1
                    
                    log_progress(run, prompt_idx + 1, 
                               result['inference_time_seconds'], 
                               result['peak_memory_gb'],
                               result['clip_score'],
                               result['frame_consistency'])
                    
                    save_checkpoint(prompt_idx, run)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        log_error(f"OOM Error: {str(e)}", run, prompt_idx + 1)
                        
                        failed_result = {
                            'model_name': 'HunyuanVideo-1.5',
                            'run_number': run,
                            'prompt_index': prompt_idx + 1,
                            'prompt_text': prompt,
                            'inference_time_seconds': 0,
                            'peak_memory_gb': 0,
                            'compute_cost_usd': 0,
                            'clip_score': 0,
                            'frame_consistency': 0,
                            'status': 'OOM_FAILED'
                        }
                        append_to_csv(failed_result)
                        videos_completed += 1
                        save_checkpoint(prompt_idx, run)
                        
                        print(f"✗ Skipping video {videos_completed}/{total_videos} due to OOM")
                        continue
                    else:
                        log_error(f"Runtime Error: {str(e)}", run, prompt_idx + 1)
                        
                        failed_result = {
                            'model_name': 'HunyuanVideo-1.5',
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
                        save_checkpoint(prompt_idx, run)
                        
                        print(f"✗ Skipping video {videos_completed}/{total_videos} due to error")
                        continue
                
                except Exception as e:
                    log_error(f"General Error: {str(e)}", run, prompt_idx + 1)
                    
                    failed_result = {
                        'model_name': 'HunyuanVideo-1.5',
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
                    save_checkpoint(prompt_idx, run)
                    
                    print(f"✗ Skipping video {videos_completed}/{total_videos} due to error")
                    continue
        
        cleanup_model(pipeline)
        
    except Exception as e:
        log_error(f"Model loading error: {str(e)}", 0, 0)
        print(f"\n✗ Error with model {MODEL_ID}: {e}")

def calculate_statistics():
    if os.path.exists(RESULTS_CSV_FILE):
        df = pd.read_csv(RESULTS_CSV_FILE)
    else:
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_CSV_FILE, index=False)
    
    df_success = df[df['status'] == 'SUCCESS']
    
    print("\n" + "="*60)
    print("✓ Results saved to results_hunyuan.csv")
    print(f"Total videos: {len(df)} | Successful: {len(df_success)} | Failed: {len(df) - len(df_success)}")
    print("="*60)
    
    if len(df_success) > 0:
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS - HUNYUANVIDEO-1.5")
        print("="*60)
        
        avg_time = df_success['inference_time_seconds'].mean()
        std_time = df_success['inference_time_seconds'].std()
        
        avg_memory = df_success['peak_memory_gb'].mean()
        std_memory = df_success['peak_memory_gb'].std()
        
        avg_clip = df_success['clip_score'].mean()
        std_clip = df_success['clip_score'].std()
        
        avg_consistency = df_success['frame_consistency'].mean()
        std_consistency = df_success['frame_consistency'].std()
        
        avg_cost = df_success['compute_cost_usd'].mean()
        total_cost = df_success['compute_cost_usd'].sum()
        
        print("\nHunyuanVideo-1.5:")
        print(f"  Average Time: {avg_time:.2f} ± {std_time:.2f} seconds")
        print(f"  Average Memory: {avg_memory:.2f} ± {std_memory:.2f} GB")
        print(f"  CLIP Score: {avg_clip:.1f} ± {std_clip:.1f}")
        print(f"  Frame Consistency: {avg_consistency:.3f} ± {std_consistency:.3f}")
        print(f"  Avg Cost/Video: ${avg_cost:.4f}")
        print(f"  Total Cost: ${total_cost:.2f}")
    
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
        clip_metric = clip_metric.to("cpu")
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
