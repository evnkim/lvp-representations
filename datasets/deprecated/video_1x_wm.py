from pathlib import Path
from tqdm import tqdm
import cv2
import pandas as pd

from ..video_base import VideoDataset


class WorldModel1XDataset(VideoDataset):
    """
    1X world model challenge dataset from https://huggingface.co/datasets/1x-technologies/worldmodel_raw_data
    """

    def download(self):
        from huggingface_hub import snapshot_download

        raw_dir = self.data_root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id="1x-technologies/worldmodel_raw_data",
            local_dir=raw_dir,
            repo_type="dataset",
        )

        records = []
        split_dict = {
            "training": list((raw_dir / "train_v2.0_raw/videos/").glob("*.mp4")),
            "validation": list((raw_dir / "val_v2.0_raw/").glob("*.mp4")),
        }
        for split, video_paths in split_dict.items():
            for video_path in tqdm(video_paths, desc=f"Verifying {split} videos"):
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    continue

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                records.append(
                    {
                        "video_path": str(video_path.relative_to(self.data_root)),
                        "height": height,
                        "width": width,
                        "fps": fps,
                        "n_frames": n_frames,
                        "split": split,
                    }
                )

        # Save as CSV
        metadata_path = self.data_root / self.metadata_path
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame.from_records(records)
        df.to_csv(metadata_path, index=False)
        print(f"Created metadata CSV with {len(records)} videos")
