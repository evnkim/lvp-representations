import requests
import subprocess
import json
import pandas as pd
import zipfile
import cv2
from pathlib import Path
from tqdm import tqdm

from .video_base import VideoDataset


class SomethingSomethingDataset(VideoDataset):
    """
    Something Something Dataset from https://arxiv.org/abs/1706.04261
    """

    def download(self):
        self.data_root.mkdir(parents=True, exist_ok=True)

        urls = [
            "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-00",
            "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-01",
            "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip",
        ]

        for url in urls:
            filename = Path(url).name
            filepath = self.data_root / filename

            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Use shell command to concatenate and extract tar video files
        print("Concatenating and extracting tar files...")
        cmd = f"cd {self.data_root} && cat 20bn-something-something-v2-0? | tar -xvzf -"
        subprocess.run(cmd, shell=True, check=True)
        print(f"Deleting zip files for video data...")
        for zip_file in self.data_root.glob("20bn-something-something-v2-0*"):
            print(f"Deleting {zip_file.name}...")
            zip_file.unlink()

        # Unzip the labels package
        labels_zip_path = (
            self.data_root / "20bn-something-something-download-package-labels.zip"
        )
        if labels_zip_path.exists():
            print(f"Extracting {labels_zip_path.name}...")
            with zipfile.ZipFile(labels_zip_path, "r") as zip_ref:
                zip_ref.extractall(self.data_root)
        print(f"Deleting zip file for labels...")
        labels_zip_path.unlink()

        # Create metadata CSV from labels
        print("Creating metadata CSV file for Something Something Dataset")

        json_files = {
            "training": "labels/train.json",
            "validation": "labels/validation.json",
        }

        records = []
        for split, json_file in json_files.items():
            with open(self.data_root / json_file, "r") as f:
                labels = json.load(f)

            for item in tqdm(labels, desc=f"Creating metadata for {split}"):
                webm_video_path = f"20bn-something-something-v2/{item['id']}.webm"
                mp4_video_path = f"20bn-something-something-v2/{item['id']}.mp4"

                total_videos = len(labels)
                successful_conversions = 0

                if (self.data_root / webm_video_path).exists():
                    # Convert webm to mp4 using ffmpeg
                    input_path = str(self.data_root / webm_video_path)
                    output_path = str(self.data_root / mp4_video_path)
                    cmd = f'ffmpeg -i {input_path} -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -c:a aac {output_path}'
                    try:
                        subprocess.run(
                            cmd,
                            shell=True,
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        # Delete the webm file after successful conversion
                        (self.data_root / webm_video_path).unlink()

                        # Get video metadata using cv2
                        cap = cv2.VideoCapture(output_path)
                        if not cap.isOpened():
                            continue

                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()

                        caption = item["label"].replace("pretending to ", "")

                        records.append(
                            {
                                "video_path": mp4_video_path,
                                "caption": caption,
                                "height": height,
                                "width": width,
                                "fps": fps,
                                "n_frames": n_frames,
                                "split": split,
                            }
                        )
                        successful_conversions += 1
                    except subprocess.CalledProcessError:
                        print(f"Conversion failed for {webm_video_path}")

                conversion_rate = (successful_conversions / total_videos) * 100
                print(f"Conversion success rate: {conversion_rate:.2f}%")

        # Save as CSV
        metadata_path = self.data_root / self.metadata_path
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame.from_records(records)
        df.to_csv(metadata_path, index=False)
        print(f"Created metadata CSV with {len(records)} videos")
