# CLIP Score Fix Summary

## Changes Made to `experiment_veo.py`

### 1. ✅ CLIP Scoring Now Fails Loudly
- **Removed**: Silent `try/except` that returned `0.0` on CLIP errors
- **Added**: Explicit exception raising with full traceback logging
- **Status Field**: On CLIP failure, status becomes `CLIP_ERROR:<message>` instead of `SUCCESS`
- **Logging**: Full exception traceback written to log file

### 2. ✅ Frame Validation with Explicit Assertions
- **Added**: Explicit check: `if not frames or len(frames) == 0` → raises exception
- **Added**: Assertion: `assert len(frames) > 0` after extraction
- **Status Field**: If frames empty, status becomes `FRAME_EXTRACT_ERROR`
- **Logging**: Number of extracted frames logged for debugging

### 3. ✅ Standardized Frame Sampling (Matching CogVideoX)
- **Changed**: CLIP now samples K=8 frames uniformly across video (not just middle frame)
- **Method**: `np.linspace(0, num_frames-1, K)` for uniform sampling
- **Format**: Frames validated as RGB uint8 (H,W,3) numpy arrays
- **Tensor Conversion**: `(H,W,C) → (C,H,W)` with proper permute
- **Averaging**: CLIP score computed per frame, then averaged

### 4. ✅ Verified Prompt Text Usage
- **Logging**: Exact prompt text logged: `CLIP: Using prompt text: '{prompt_text}'`
- **Validation**: Full prompt string passed to CLIP encoder (not truncated)
- **Debug**: Prompt saved to `debug_clip_frames/prompt.txt` for sanity runs

### 5. ✅ CLIP Score Scale Matches CogVideoX
- **Scale**: 0-100 (same as CogVideoX)
- **Validation**: Checks for NaN/Inf and raises exception if invalid
- **No Clamping**: Raw cosine similarity preserved (not forced to 0.0)

### 6. ✅ Sanity Experiment Mode Added
- **Flag**: `SANITY_ONLY = True` (default)
- **Sanity Run**: Generates 1 video (prompt_index=1, run_number=1)
- **Acceptance Criteria**:
  - `status == "SUCCESS"`
  - `clip_score > 0` (and not NaN)
- **On Failure**: Stops immediately with error message
- **On Success**: Automatically proceeds to full 9-video experiment

### 7. ✅ Debug Artifacts Saved
- **Location**: `output_videos/debug_clip_frames/`
- **Artifacts**:
  - `sanity_prompt1_run1_frame_0.jpg` through `frame_7.jpg` (8 sampled frames)
  - `sanity_prompt1_run1_prompt.txt` (exact prompt text)
- **Logging**: Per-frame CLIP scores logged for inspection

## How to Use

### First Run (Sanity Check)
```bash
cd GoogleVeo
export GEMINI_API_KEY='your-api-key'
python experiment_veo.py
```

**Expected Output**:
```
🔍 Running SANITY CHECK mode...
============================================================
SANITY CHECK: TESTING CLIP SCORING
============================================================
Generating 1 video (prompt_index=1, run_number=1)
Acceptance criteria: status=SUCCESS, clip_score > 0
============================================================

[Video generation logs...]

CLIP: Sampling 8 frames from 192 total frames
CLIP: Using prompt text: 'A person walking in a park on a sunny day'
CLIP: Frame 0 score = 28.45
CLIP: Frame 1 score = 29.12
...
CLIP: Average score across 8 frames = 29.87

============================================================
SANITY CHECK RESULTS
============================================================
Status: SUCCESS
CLIP Score: 29.87
Frame Consistency: 0.165
============================================================

✅ SANITY CHECK PASSED!
CLIP scoring is working: score = 29.87
Proceeding to full experiment...

🚀 Proceeding to FULL EXPERIMENT...
```

### Full Experiment (After Sanity Pass)
The script automatically proceeds to generate all 9 videos after sanity check passes.

Alternatively, manually set `SANITY_ONLY = False` and rerun.

## Debugging Failed Sanity Check

If sanity check fails:

1. **Check log file**: `veo_experiment.log` for full traceback
2. **Check debug frames**: `output_videos/debug_clip_frames/` to see what frames were extracted
3. **Check prompt**: `output_videos/debug_clip_frames/sanity_prompt1_run1_prompt.txt`
4. **Common issues**:
   - CLIP metric not initialized (check transformers installation)
   - Frames not extracted (check video download/format)
   - API key issues (check GEMINI_API_KEY)

## Expected Results

After successful run, `results_veo.csv` should show:
- All rows with `status=SUCCESS`
- All rows with `clip_score > 0` (typically 25-40 range)
- No rows with `clip_score=0.0`

## Configuration

Key parameters in `experiment_veo.py`:
- `SANITY_ONLY = True/False` - Enable/disable sanity check
- `NUM_CLIP_SAMPLE_FRAMES = 8` - Number of frames to sample for CLIP
- `NUM_RUNS_PER_PROMPT = 3` - Runs per prompt (9 total videos)

## Comparison with CogVideoX

Both experiments now use identical CLIP methodology:
- Sample K frames uniformly across video
- Compute CLIP score per frame
- Average scores
- Same 0-100 scale
- Same validation and error handling
