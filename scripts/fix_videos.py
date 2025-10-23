import av
import json
import imageio
import os
import shutil
from tqdm import tqdm


def check_video(path):
    try:
        container = av.open(path)
        for frame in container.decode(video=0):
            pass
        return True
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return False


def save_frames(frames, path, fps=30):
    frames = [frame.to_image() for frame in frames]
    imageio.mimsave(path, frames, fps=fps)


def fix_video(path):
    frames = []
    try:
        container = av.open(path)
        for frame in container.decode(video=0):
            frames.append(frame)

    except Exception as e:
        print(f"Only read {len(frames)} frames from {path}, fixing...")
    finally:
        num_current_frames = len(frames)
        episode_index = int(path.split('_')[-1].split('.')[0])
        meta_path = os.path.join(*path.split('/')[:-4], 'meta', 'episodes.jsonl')
        with open(meta_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if data['episode_index'] == episode_index:
                num_expected_frames = data['length']
                break
            
        if num_current_frames == num_expected_frames:
            print(f"Frame count matches for {path}, no fix needed.")
            return
        
        pad_frames = num_expected_frames - num_current_frames
        if pad_frames <= 0:
            print(f"Unexpected: current frames {num_current_frames} >= expected {num_expected_frames} for {path}")
            raise ValueError("Unexpected frame count")
        
        last_frame = frames[-1]
        for _ in range(pad_frames):
            frames.append(last_frame)

        output_path = path.replace('.mp4', '_fixed.mp4')
        save_frames(frames, output_path)
        print(f"Saved fixed video to {output_path}")

        frames = []
        container = av.open(output_path)
        for frame in container.decode(video=0):
            frames.append(frame)

        if len(frames) != num_expected_frames:
            print(f"Fix failed: still {len(frames)} frames, expected {num_expected_frames} for {path}")
            raise ValueError("Fix failed")
        else:
            print(f"Fix succeeded: now {len(frames)} frames for {path}")
            shutil.move(output_path, path)


def main(args):
    import os

    video_paths = []
    for root, _, files in os.walk(args.root):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mov")):
                full_path = os.path.join(root, file)
                video_paths.append(full_path)
    
    video_paths = video_paths[:346]  # Limit to first 100 videos for testing
    for full_path in tqdm(video_paths, desc="Checking videos"):
        if not check_video(full_path):
            print(f"Corrupted video found: {full_path}")
            fix_video(full_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check integrity of video files in a directory.")
    parser.add_argument("--root", type=str, help="Root directory to scan for video files.")
    args = parser.parse_args()
    main(args)