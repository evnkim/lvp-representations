import pandas as pd
from typing import List, Tuple, Any, Dict
import time
import json
from pathlib import Path
import decord
from .video_base import VideoDataset


class PandasVideoDataset(VideoDataset):
    def _load_records(self) -> Tuple[List[str], List[str]]:
        """
        Given the metadata file, loads the records as a list.
        Each record is a dictionary containing a datapoint's mp4 path / caption etc.
        Require these entries: "video_path", "caption", "height", "width", "n_frames", "fps"

        For pandas70m, there are one extra key "youtube_key_segment", looks like: "2NQDnwJEBeQ_segment_7".
        It's the key identifier for the video.

        Pandas 70M comes with json config file. This method will convert the json config file to a csv file and save it before using.
        """
        if self.metadata_path.suffix == ".json":
            # convert a legacy json file to a csv file we need
            start_time = time.time()
            records = []
            with open(self.data_root / self.metadata_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    if "mp4_path" in item:
                        item["video_path"] = item["mp4_path"]
                        del item["mp4_path"]
                    if "start_frame_index" in item:
                        item["trim_start"] = item["start_frame_index"]
                        del item["start_frame_index"]
                    if "end_frame_index" in item:
                        item["trim_end"] = item["end_frame_index"]
                        del item["end_frame_index"]
                    if "prompt_embed_path" in item:
                        item["prompt_embed_path"] = (
                            "prompt_embeds/" + item["prompt_embed_path"] + ".pt"
                        )
                    if "answers_for_four_questions" in item:
                        del item["answers_for_four_questions"]
                    records.append(item)

            df = pd.DataFrame.from_records(records)
            csv_path = self.metadata_path.with_suffix(".csv")
            df.to_csv(self.data_root / csv_path, index=False)
            self.metadata_path = csv_path
            end_time = time.time()
            print(f"Time taken for converting records: {end_time - start_time} seconds")

        return super()._load_records()


if __name__ == "__main__":
    # do debug test
    import torch
    from omegaconf import OmegaConf

    debug_config = {
        "debug": True,
        "data_root": "/n/holylfs06/LABS/sham_lab/Lab/eiwm_data/pandas/",
        "metadata_path": "pandas_filtered_human_clip_meta_gemini_1.5_flash.json",
        "auto_download": False,
        "force_download": False,
        "test_percentage": 0.1,
        "id_token": "",
        "resolution": [256, 256],
        "n_frames": 8,
        "fps": 30,
        "trim_mode": "speedup",
        "pad_mode": "pad_last",
        "filtering": {
            "disable": False,
            "height": [32, 2160],
            "width": [32, 3840],
            "n_frames": [8, 1000],
            "fps": [1, 60],
        },
        "load_video_latent": False,
        "load_prompt_embed": False,
        "augmentation": {"random_flip": 0.5, "ratio": None, "scale": None},
        "image_to_video": False,
        "check_video_path": False,
    }

    # Convert dict to OmegaConf
    cfg = OmegaConf.create(debug_config)

    # Create dataset
    dataset = PandasVideoDataset(cfg=cfg, split="training")

    # Load one sample and print its contents
    sample = dataset[0]
    print("\nSample contents:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: Tensor of shape {value.shape}")
        elif isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
