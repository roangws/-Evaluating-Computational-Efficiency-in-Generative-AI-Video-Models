import cv2
import os

def extract_middle_frame(video_path, output_path):
    """Extract the middle frame from a video and save as JPG."""
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"✓ Created thumbnail: {output_path}")
    else:
        print(f"✗ Failed to extract frame from: {video_path}")
    
    cap.release()

def main():
    base_dir = "/Users/roan-aparavi/aparavi-repo/Cost-Efficiency Metrics Evaluating Computational and Resource Efficiency in Gerative AI Video Models/HunyuanVideo"
    output_videos_dir = os.path.join(base_dir, "output_videos")
    output_frames_dir = os.path.join(base_dir, "output_frames")
    
    os.makedirs(output_frames_dir, exist_ok=True)
    
    for prompt_num in range(1, 4):
        prompt_dir = f"hunyuan_prompt_{prompt_num}"
        video_dir = os.path.join(output_videos_dir, prompt_dir)
        frame_dir = os.path.join(output_frames_dir, prompt_dir)
        
        os.makedirs(frame_dir, exist_ok=True)
        
        for run_num in range(1, 4):
            video_file = f"hunyuan_prompt_{prompt_num}_run_{run_num}.mp4"
            video_path = os.path.join(video_dir, video_file)
            
            frame_file = f"hunyuan_prompt_{prompt_num}_run_{run_num}_middle_frame.jpg"
            frame_path = os.path.join(frame_dir, frame_file)
            
            if os.path.exists(video_path):
                extract_middle_frame(video_path, frame_path)
            else:
                print(f"✗ Video not found: {video_path}")
    
    print("\n✅ Thumbnail extraction complete!")

if __name__ == "__main__":
    main()
