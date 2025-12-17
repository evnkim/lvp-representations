import pandas as pd
from pathlib import Path
import ijson
from typing import Dict, Any
from .video_base import VideoDataset


class Ego4DVideoDataset(VideoDataset):

    def download(self):
        from ego4d.cli.cli import main_cfg as download_ego4d
        from ego4d.cli.config import Config as Ego4DConfig

        raw_dir = self.data_root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        aws_credentials_path = Path.home() / ".aws" / "credentials"
        if not aws_credentials_path.exists():
            raise FileNotFoundError(
                f"AWS credentials file not found at {aws_credentials_path}"
                "For Ego4D auto download, you need to request access and use the "
                "emailed key to set up AWS credentials first."
                "See https://ego4d-data.org/ for more information."
            )

        cfg = Ego4DConfig(
            output_directory=str(raw_dir),
            datasets=["annotations", "clips"],
            benchmarks=["FHO"],
            metadata=True,
            assume_yes=True,
        )

        import botocore

        try:
            download_ego4d(cfg)
        except botocore.exceptions.ClientError as e:
            print(e)
            raise RuntimeError(
                "Failed to download Ego4D dataset due to the above error."
                "If you see an error occurred (403) when calling the HeadObject operation: Forbidden",
                "It's likely due to an expired Ego4D AWS credential. Renew the dataset's online form and update the AWS credentials.",
            )

        annotation_file = "v2/annotations/fho_main.json"
        print("Creating metadata CSV...")
        records = []
        with open(raw_dir / annotation_file, "rb") as file:
            # Create a parser for the videos array
            videos = ijson.items(file, "videos.item")
            total = 0

            for v in videos:
                fps = round(v["video_metadata"]["fps"])
                n_frames = v["video_metadata"]["num_frames"]
                width = v["video_metadata"]["width"]
                height = v["video_metadata"]["height"]
                for c in v["annotated_intervals"]:
                    video_path = "raw/v2/clips/" + c["clip_uid"] + ".mp4"

                    if not Path(self.data_root / video_path).exists():
                        continue

                    for a in c["narrated_actions"]:
                        total += 1
                        critical_frames = a["clip_critical_frames"]
                        is_valid_action = a["is_valid_action"]
                        is_rejected = a["is_rejected"]
                        is_invalid_annotation = a["is_invalid_annotation"]
                        is_partial = a["is_partial"]
                        if (
                            not critical_frames
                            or not is_valid_action
                            or is_rejected
                            or is_invalid_annotation
                            or is_partial
                        ):
                            continue
                        caption = a["narration_text"]
                        caption = (
                            caption.replace("#cC c ", " ")
                            .replace("#Cc C ", " ")
                            .replace("#C C ", "")
                            .replace("#c  c ", " ")
                            .replace("#c- c ", " ")
                            .replace("#c C ", " ")
                            .replace("#c c", " ")
                            .replace("#CC ", " ")
                            .replace("#C  C ", " ")
                            .replace("#C c ", " ")
                            .replace("#cc ", " ")
                            .replace("#C- C ", " ")
                            .replace("#c C ", " ")
                            .replace("#C ", " ")
                            .replace("#c ", " ")
                            .replace("#", " ")
                        )
                        pre_frame = critical_frames["pre_frame"]
                        post_frame = critical_frames["post_frame"]
                        pnr_frame = critical_frames["pnr_frame"]
                        contact_frame = critical_frames["contact_frame"]

                        # some manual heuristics to trim the video
                        target_len = self._n_frames_in_src(fps)
                        trim_start = pre_frame
                        psudo_min_end = int((post_frame - pnr_frame) * 0.1) + pnr_frame
                        if psudo_min_end - pre_frame >= target_len:
                            trim_end = psudo_min_end
                        elif post_frame - pnr_frame < target_len:
                            trim_end = post_frame
                            trim_start = max(trim_end - target_len, pre_frame - 15)
                        else:
                            trim_end = target_len + pre_frame

                        trim_start = max(0, trim_start)
                        trim_end = min(n_frames, trim_end)

                        records.append(
                            {
                                "video_path": video_path,
                                "height": height,
                                "width": width,
                                "n_frames": n_frames,
                                "fps": fps,
                                "original_caption": caption,
                                "trim_start": trim_start,
                                "trim_end": trim_end,
                                "pre_frame": pre_frame,
                                "pnr_frame": pnr_frame,
                                "post_frame": post_frame,
                                "contact_frame": contact_frame,
                            }
                        )
        metadata_path = self.data_root / self.metadata_path
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame.from_records(records)
        df.to_csv(metadata_path, index=False)
        print(f"Created metadata CSV with {len(records)} records")
