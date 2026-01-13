import os
import time
import warnings
import requests
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from google import genai

warnings.filterwarnings('ignore')

# Configuration
PROMPTS = [
    "A person walking in a park on a sunny day",
    "A car driving on a highway with trees in background",
    "A cat playing with a ball in a living room"
]

NUM_RUNS_PER_PROMPT = 3
VEO_MODEL = "veo-3.1-generate-preview"
VIDEO_DURATION_SECONDS = 5
POLLING_INTERVAL_SECONDS = 10
MAX_POLLING_TIME_SECONDS = 300
COST_PER_SECOND_USD = 0.15
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = [2, 4, 8]

# File paths
RESULTS_CSV_FILE = "results_veo.csv"
SUMMARY_FILE = "summary_veo.txt"
LOG_FILE = "veo_experiment.log"
OUTPUT_VIDEO_DIR = "output_videos"
OUTPUT_FRAMES_DIR = "output_frames"

# Pricing (as of Jan 2025 - Gemini API Veo pricing)
# Veo 3.1 Fast: $0.15 per second (720p/1080p)

# Global variables
clip_metric = None
total_videos = len(PROMPTS) * NUM_RUNS_PER_PROMPT
videos_completed = 0
experiment_start_time = None


def log_message(message, level="INFO"):
    """Log message to file and console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}\n"
    
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry)
    
    print(log_entry.strip())


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
        log_message(f"CLIP score calculation failed: {e}", "WARNING")
        return 0.0


def extract_frames_from_video(video_path):
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to video file
    
    Returns:
        list: List of numpy arrays (frames)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        return frames
    except Exception as e:
        log_message(f"Error extracting frames from {video_path}: {e}", "ERROR")
        return []


def download_video_from_url(url, output_path, max_retries=3):
    """
    Download video from URL with retry logic.
    
    Args:
        url: Video URL
        output_path: Local path to save video
        max_retries: Maximum number of retry attempts
    
    Returns:
        bool: True if successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            log_message(f"Video downloaded successfully to {output_path}")
            return True
        except Exception as e:
            log_message(f"Download attempt {attempt + 1} failed: {e}", "WARNING")
            if attempt < max_retries - 1:
                time.sleep(RETRY_BACKOFF_SECONDS[attempt])
            else:
                log_message(f"Failed to download video after {max_retries} attempts", "ERROR")
                return False
    
    return False


def generate_veo_video(prompt_text, run_number, prompt_index):
    """
    Generate video using Google Veo 3.1 via Gemini API.
    
    API Model: gemini-2.0-flash-exp (or latest video-capable model)
    Parameters: 
        - prompt: Text description
        - duration: ~10 seconds (target)
        - resolution: 1280x720 or closest available
    
    Cost Calculation: Based on Google Cloud Gemini pricing
        - Current placeholder: $0.05 per video
        - Update with actual token-based pricing from API response
    
    Args:
        prompt_text: Text prompt for video generation
        run_number: Run number (1-3)
        prompt_index: Prompt index (1-3)
    
    Returns:
        dict: Result dictionary with metrics and status
    """
    log_message(f"Starting video generation - Prompt {prompt_index}, Run {run_number}")
    log_message(f"Prompt: '{prompt_text}'")
    
    # Create output directory
    output_dir = f"{OUTPUT_VIDEO_DIR}/veo_prompt_{prompt_index}"
    os.makedirs(output_dir, exist_ok=True)
    
    video_filename = f"veo_prompt_{prompt_index}_run_{run_number}.mp4"
    video_path = os.path.join(output_dir, video_filename)
    
    start_time = time.time()
    status = "SUCCESS"
    error_message = None
    
    try:
        # Generate video using Gemini API with Veo 3.1 Fast
        log_message(f"Calling Gemini API with model: {VEO_MODEL}")
        
        # Initialize Gemini client (reads GEMINI_API_KEY automatically)
        client = genai.Client()
        
        # Make API call with retry logic
        operation = None
        for attempt in range(MAX_RETRIES):
            try:
                log_message(f"Calling generate_videos (attempt {attempt + 1})")
                
                # Call generate_videos
                operation = client.models.generate_videos(
                    model=VEO_MODEL,
                    prompt=prompt_text
                )
                
                # Store operation name for polling
                operation_name = operation.name
                log_message(f"Operation started: {operation_name}")
                break
                
            except Exception as api_error:
                log_message(f"API call attempt {attempt + 1} failed: {api_error}", "WARNING")
                if "quota" in str(api_error).lower():
                    status = "QUOTA_EXCEEDED"
                    error_message = str(api_error)
                    break
                elif "timeout" in str(api_error).lower():
                    status = "TIMEOUT"
                    error_message = str(api_error)
                elif attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_SECONDS[attempt])
                else:
                    status = "API_ERROR"
                    error_message = str(api_error)
        
        if operation is None or status != "SUCCESS":
            raise Exception(f"API call failed: {error_message}")
        
        # Poll operation until complete
        log_message("Polling operation for completion...")
        poll_start = time.time()
        while not operation.done:
            if time.time() - poll_start > MAX_POLLING_TIME_SECONDS:
                raise Exception("Operation polling timeout")
            
            time.sleep(POLLING_INTERVAL_SECONDS)
            operation = client.operations.get(operation)
            log_message(f"Operation status: {'done' if operation.done else 'running'}")
        
        log_message("Operation completed successfully")
        
        # Extract generated video
        if not hasattr(operation.response, 'generated_videos') or len(operation.response.generated_videos) == 0:
            raise Exception("No generated videos in operation response")
        
        generated_video = operation.response.generated_videos[0]
        
        # Download video
        log_message(f"Downloading video to {video_path}")
        video_bytes = client.files.download(file=generated_video.video)
        
        # Write to file
        with open(video_path, 'wb') as f:
            f.write(video_bytes)
        
        log_message(f"Video saved to {video_path}")
        
        inference_time = time.time() - start_time
        
        # Calculate cost based on Gemini API Veo pricing
        # $0.15 per second for Veo 3.1 Fast
        # Try to get actual duration from response, otherwise estimate
        if hasattr(generated_video, 'duration_seconds'):
            video_duration = generated_video.duration_seconds
        elif hasattr(operation.response, 'video_properties') and hasattr(operation.response.video_properties, 'duration'):
            video_duration = operation.response.video_properties.duration
        else:
            video_duration = VIDEO_DURATION_SECONDS  # Fallback estimate
        
        compute_cost_usd = video_duration * COST_PER_SECOND_USD
        
        # Extract frames for quality metrics
        frames = extract_frames_from_video(video_path)
        
        if len(frames) == 0:
            raise Exception("No frames extracted from generated video")
        
        log_message(f"Extracted {len(frames)} frames from video")
        
        # Calculate CLIP score (middle frame)
        middle_idx = len(frames) // 2
        middle_frame = frames[middle_idx]
        clip_score = calculate_clip_score(middle_frame, prompt_text)
        
        # Calculate frame consistency
        frame_consistency = calculate_frame_consistency(frames)
        
        # Save sample frame
        frames_dir = f"{OUTPUT_FRAMES_DIR}/veo_prompt_{prompt_index}"
        os.makedirs(frames_dir, exist_ok=True)
        frame_path = os.path.join(frames_dir, f"run_{run_number}_middle_frame.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(middle_frame, cv2.COLOR_RGB2BGR))
        
        log_message(f"✓ Video generated successfully - Time: {inference_time:.2f}s, "
                   f"CLIP: {clip_score:.1f}, Consistency: {frame_consistency:.3f}")
        
        result = {
            'model_name': 'Google Veo 3.1',
            'run_number': run_number,
            'prompt_index': prompt_index,
            'prompt_text': prompt_text,
            'inference_time_seconds': inference_time,
            'peak_memory_gb': 0.0,  # Cloud-hosted
            'compute_cost_usd': compute_cost_usd,
            'clip_score': clip_score,
            'frame_consistency': frame_consistency,
            'status': status
        }
        
    except Exception as e:
        inference_time = time.time() - start_time
        log_message(f"✗ Video generation failed: {e}", "ERROR")
        
        # Determine error type
        if status == "SUCCESS":  # Error occurred after API call
            if "quota" in str(e).lower():
                status = "QUOTA_EXCEEDED"
            elif "timeout" in str(e).lower():
                status = "TIMEOUT"
            elif "download" in str(e).lower():
                status = "DOWNLOAD_ERROR"
            else:
                status = "PROCESSING_ERROR"
        
        result = {
            'model_name': 'Google Veo 3.1',
            'run_number': run_number,
            'prompt_index': prompt_index,
            'prompt_text': prompt_text,
            'inference_time_seconds': inference_time,
            'peak_memory_gb': 0.0,
            'compute_cost_usd': 0.0,
            'clip_score': 0.0,
            'frame_consistency': 0.0,
            'status': status
        }
    
    return result


def append_to_csv(result):
    """Append result to CSV file."""
    df = pd.DataFrame([result])
    file_exists = os.path.exists(RESULTS_CSV_FILE)
    df.to_csv(RESULTS_CSV_FILE, mode='a', header=not file_exists, index=False)


def run_experiments():
    """Run all video generation experiments."""
    global videos_completed, experiment_start_time
    
    log_message("="*60)
    log_message("GOOGLE VEO 3.1 VIDEO GENERATION BENCHMARK")
    log_message("="*60)
    log_message(f"Model: {VEO_MODEL}")
    log_message("API: Gemini API")
    log_message(f"Prompts: {len(PROMPTS)}")
    log_message(f"Runs per prompt: {NUM_RUNS_PER_PROMPT}")
    log_message(f"Total videos to generate: {total_videos}")
    log_message(f"Target video duration: {VIDEO_DURATION_SECONDS} seconds")
    log_message("="*60)
    
    experiment_start_time = time.time()
    results = []
    
    for prompt_idx, prompt in enumerate(PROMPTS):
        prompt_index = prompt_idx + 1
        log_message(f"\nPrompt {prompt_index}/{len(PROMPTS)}: \"{prompt}\"")
        log_message("-" * 60)
        
        for run in range(1, NUM_RUNS_PER_PROMPT + 1):
            result = generate_veo_video(prompt, run, prompt_index)
            results.append(result)
            append_to_csv(result)
            
            videos_completed += 1
            
            # Progress update
            progress_pct = (videos_completed / total_videos) * 100
            elapsed_time = time.time() - experiment_start_time
            avg_time = elapsed_time / videos_completed
            eta_seconds = avg_time * (total_videos - videos_completed)
            
            log_message(f"Progress: {videos_completed}/{total_videos} ({progress_pct:.1f}%) | "
                       f"ETA: {eta_seconds/60:.1f} minutes")
            
            # Check if we should stop due to quota issues
            if result['status'] == 'QUOTA_EXCEEDED':
                log_message("Quota exceeded - stopping experiment", "ERROR")
                log_message(f"Completed {videos_completed}/{total_videos} videos before quota limit")
                break
        
        # Stop outer loop if quota exceeded
        if videos_completed < (prompt_index * NUM_RUNS_PER_PROMPT):
            break
    
    total_time = time.time() - experiment_start_time
    log_message("="*60)
    log_message("EXPERIMENT COMPLETED")
    log_message(f"Total time: {total_time/60:.1f} minutes")
    log_message(f"Videos completed: {videos_completed}/{total_videos}")
    log_message("="*60)
    
    return results


def calculate_statistics():
    """Calculate and display statistics from results."""
    if not os.path.exists(RESULTS_CSV_FILE):
        log_message("No results file found", "ERROR")
        return
    
    df = pd.read_csv(RESULTS_CSV_FILE)
    df_success = df[df['status'] == 'SUCCESS']
    
    log_message("\n" + "="*60)
    log_message("STATISTICAL ANALYSIS")
    log_message("="*60)
    log_message(f"Total videos: {len(df)}")
    log_message(f"Successful: {len(df_success)}")
    log_message(f"Failed: {len(df) - len(df_success)}")
    
    if len(df_success) == 0:
        log_message("No successful videos to analyze", "WARNING")
        return
    
    # Calculate statistics
    avg_time = df_success['inference_time_seconds'].mean()
    std_time = df_success['inference_time_seconds'].std()
    
    avg_clip = df_success['clip_score'].mean()
    std_clip = df_success['clip_score'].std()
    
    avg_consistency = df_success['frame_consistency'].mean()
    std_consistency = df_success['frame_consistency'].std()
    
    total_cost = df_success['compute_cost_usd'].sum()
    avg_cost = df_success['compute_cost_usd'].mean()
    
    # Summary
    summary = f"""
{'='*60}
GOOGLE VEO 3.1 RESULTS SUMMARY
{'='*60}
Model: Google Veo 3.1 (via Vertex AI)
Successful Videos: {len(df_success)}/{len(df)}

PERFORMANCE METRICS:
  Inference Time: {avg_time:.2f} ± {std_time:.2f} seconds
  Peak Memory: 0.0 GB (Cloud-hosted)
  
QUALITY METRICS:
  CLIP Score: {avg_clip:.1f} ± {std_clip:.1f}
  Frame Consistency: {avg_consistency:.3f} ± {std_consistency:.3f}

COST METRICS:
  Total Cost: ${total_cost:.2f}
  Avg Cost/Video: ${avg_cost:.4f}

STATUS BREAKDOWN:
"""
    
    # Status counts
    status_counts = df['status'].value_counts()
    for status, count in status_counts.items():
        summary += f"  {status}: {count}\n"
    
    summary += "="*60
    
    log_message(summary)
    
    # Save summary to file
    with open(SUMMARY_FILE, 'w') as f:
        f.write(summary)
    
    log_message(f"\n✓ Summary saved to {SUMMARY_FILE}")
    log_message(f"✓ Results saved to {RESULTS_CSV_FILE}")


def main():
    """Main execution function."""
    global clip_metric
    
    # Check for API key
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        log_message("ERROR: GEMINI_API_KEY environment variable not set", "ERROR")
        log_message("Please set your Gemini API key: export GEMINI_API_KEY='your-key-here'")
        log_message("Get a free API key from: https://ai.google.dev/aistudio")
        return
    
    log_message("✓ GEMINI_API_KEY found, ready to call Gemini API")
    
    # Initialize CLIP metric
    log_message("Initializing CLIP metric for quality evaluation...")
    try:
        clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")
        clip_metric = clip_metric.to("cpu")
        log_message("✓ CLIP metric initialized successfully")
    except Exception as e:
        log_message(f"Warning: Failed to initialize CLIP metric: {e}", "WARNING")
        log_message("Continuing without CLIP score evaluation...")
        clip_metric = None
    
    # Create output directories
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)
    
    # Run experiments
    results = run_experiments()
    
    # Calculate statistics
    if len(results) > 0:
        calculate_statistics()
    else:
        log_message("No results collected. Experiment failed.", "ERROR")


if __name__ == "__main__":
    main()
