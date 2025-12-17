from pathlib import Path
from tqdm import tqdm
import cv2
import shutil
import json
import pandas as pd
import tarfile
import decord
import subprocess
from huggingface_hub import snapshot_download
from .video_base import VideoDataset


class AgibotWorldDataset(VideoDataset):
    """
    Agibot world dataset from https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha
    """

    def preprocess_record(self, record):
        record["fps"] = self.cfg.fps_override
        return record

    def download(self):

        raw_dir = self.data_root / "agibot_world_alpha"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # snapshot_download(
        #     repo_id="agibot-world/AgiBotWorld-Alpha",
        #     local_dir=raw_dir,
        #     repo_type="dataset",
        # )

        # print("Extracting tar files...")
        # for task_dir in tqdm((raw_dir / "observations").glob("*")):
        #     for tar_file in task_dir.glob("*.tar"):
        #         tar = tarfile.open(tar_file)
        #         tar.extractall(path=task_dir)
        #         tar.close()
        #         # Delete the tar file after extraction
        #         tar_file.unlink()
        #     for episode_dir in task_dir.glob("*/"):
        #         depth_dir = episode_dir / "depth"
        #         video_dir = episode_dir / "videos"
        #         # Delete the depth directory if it exists
        #         if depth_dir.exists():
        #             shutil.rmtree(depth_dir)

        #         for video_file in video_dir.glob("*.mp4"):
        #             if video_file.name != "head_color.mp4":
        #                 video_file.unlink()
        #             else:
        #                 reencoded_video_path = video_file.with_name(
        #                     f"{video_file.stem}_reencoded.mp4"
        #                 )
        #                 command = [
        #                     "ffmpeg",
        #                     "-y",
        #                     "-i",
        #                     str(video_file),
        #                     "-c:v",
        #                     "libx264",
        #                     "-crf",
        #                     "23",
        #                     "-c:a",
        #                     "copy",
        #                     str(reencoded_video_path),
        #                 ]
        #                 print(f"Reencoding {video_file} to {reencoded_video_path}")
        #                 subprocess.run(command, check=True)

        print("Creating metadata CSV...")
        records = []

        for info_file in (raw_dir / "task_info").glob("*.json"):
            with open(info_file, "r") as f:
                info = json.load(f)
            for episode_info in tqdm(info):
                episode_id = episode_info["episode_id"]
                task_id = episode_info["task_id"]
                video_path = raw_dir / (
                    f"observations/{task_id}/{episode_id}/videos/head_color_reencoded.mp4"
                )
                if not video_path.exists():
                    print(f"Skipping {video_path} because it doesn't exist")
                    continue
                try:
                    vr = decord.VideoReader(str(video_path))
                except Exception as e:
                    print(f"Error loading video {video_path}: {e}")
                    continue
                fps = 30
                width = 640
                height = 480
                clips = episode_info["label_info"]["action_config"]
                for clip in clips:
                    trim_start = clip["start_frame"]
                    trim_end = clip["end_frame"]
                    caption = clip["action_text"]

                    records.append(
                        {
                            "video_path": video_path.relative_to(self.data_root),
                            "original_caption": caption,
                            "trim_start": trim_start,
                            "trim_end": trim_end,
                            "fps": fps,
                            "width": width,
                            "height": height,
                            "n_frames": len(vr),
                        }
                    )

        # Save as CSV
        metadata_path = self.data_root / self.metadata_path
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame.from_records(records)
        df.to_csv(metadata_path, index=False)
        print(f"Created metadata CSV with {len(records)} videos")
