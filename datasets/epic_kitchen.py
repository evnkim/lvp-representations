from pathlib import Path
import hashlib
import os
import cv2
import csv
import shutil
import urllib.request
import urllib.error
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import decord
from .video_base import VideoDataset


class EpicKitchenDataset(VideoDataset):
    """
    Epic Kitchen Dataset from https://epic-kitchens.github.io/
    """

    def __init__(self, cfg, split: str = "training"):
        self.annotation_url = cfg.download.annotation_url
        self.md5_url = cfg.download.md5_url
        self.errata_url = cfg.download.errata_url
        self.splits_url = cfg.download.splits_url
        super().__init__(cfg)

    def download(self):
        self.data_root.mkdir(parents=True, exist_ok=True)

        urls = list(self.splits_url.values()) + [
            self.md5_url,
            self.errata_url,
        ]

        for url in urls + list(self.annotation_url.values()):
            file_name = url.split("/")[-1]
            file_path = self.data_root / file_name
            if not file_path.exists():
                try:
                    print(f"Downloading {file_name}...")
                    urllib.request.urlretrieve(url, file_path)
                    print(f"Downloaded {file_name} to {file_path}")
                except urllib.error.URLError as e:
                    print(f"Failed to download {file_name}: {e}")
            else:
                print(f"{file_name} already exists, skipping download.")

        # use the official downloader
        downloader = EpicDownloader(
            base_output=self.data_root,
            splits_path_epic_55=self.data_root / "epic_55_splits.csv",
            splits_path_epic_100=self.data_root / "epic_100_splits.csv",
            md5_path=self.data_root / "md5.csv",
            errata_path=self.data_root / "errata.csv",
        )
        downloader.download(
            what=["videos"],
            participants="all",
            specific_videos="all",
            splits="all",
            challenges="all",
            extension_only=False,
            epic55_only=False,
        )

        # Delete the downloaded csv files
        for url in urls:
            file_name = url.split("/")[-1]
            file_path = self.data_root / file_name
            if file_path.exists():
                print(f"Deleting {file_name}...")
                file_path.unlink()

        # Create metadata CSV
        records = []
        for split, url in self.annotation_url.items():
            annotation_file = self.data_root / url.split("/")[-1]
            df = pd.read_csv(annotation_file)
            video_metadata_cache = {}
            for _, row in tqdm(
                df.iterrows(), desc=f"Processing {split} annotations", total=len(df)
            ):
                video_path = f"EPIC-KITCHENS/{row['participant_id']}/videos/{row['video_id']}.MP4"
                if video_path in video_metadata_cache:
                    fps, n_frames, width, height = video_metadata_cache[video_path]
                else:
                    # don't use cv2 here, it will return 0 height and width
                    vr = decord.VideoReader(str(self.data_root / video_path))
                    fps = vr.get_avg_fps()
                    n_frames = len(vr)
                    width = vr[0].shape[1]
                    height = vr[0].shape[0]
                    del vr
                    video_metadata_cache[video_path] = (fps, n_frames, width, height)

                original_start = row["start_frame"]
                original_end = row["stop_frame"]
                trim_start = original_start
                trim_end = original_end
                fps = round(fps)

                ## a bunch of herustics to handle videos that are too long
                # original_len = original_end - original_start + 1
                # removal_threshold = self.cfg.download.removal_threshold
                # removal_rate_max = self.cfg.download.removal_rate_max
                # removal_front, removal_back = self.cfg.download.removal_front_back
                # if original_len > removal_threshold[0]:
                #     amount_above = original_len - removal_threshold[0]
                #     r = amount_above / (removal_threshold[1] - removal_threshold[0])
                #     removal_rate = removal_rate_max * min(r, 1)
                #     removal_len = (original_len - removal_threshold[0]) * removal_rate
                #     trim_start = original_start + np.round(removal_len * removal_front)
                #     trim_end = original_end - np.round(removal_len * removal_back)
                records.append(
                    {
                        "video_path": video_path,
                        "original_caption": row["narration"],
                        "trim_start": trim_start,
                        "trim_end": trim_end,
                        "fps": fps,
                        "height": height,
                        "width": width,
                        "n_frames": n_frames,
                        "split": split,
                        "original_start": original_start,
                        "original_end": original_end,
                    }
                )
        # Save as CSV
        metadata_path = self.data_root / self.metadata_path
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame.from_records(records)
        df.to_csv(metadata_path, index=False)
        print(f"Created metadata CSV with {len(records)} videos")


def print_header(header, char="*"):
    print()
    print(char * len(header))
    print(header)
    print(char * len(header))
    print()


class EpicDownloader:
    # the official downloader from
    # https://github.com/epic-kitchens/epic-kitchens-download-scripts/
    def __init__(
        self,
        epic_55_base_url="https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d",
        epic_100_base_url="https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m",
        masks_base_url="https://data.bris.ac.uk/datasets/3l8eci2oqgst92n14w2yqi5ytu",
        base_output=str(Path.home()),
        splits_path_epic_55="data/epic_55_splits.csv",
        splits_path_epic_100="data/epic_100_splits.csv",
        md5_path="data/md5.csv",
        errata_path="data/errata.csv",
        errata_only=False,
    ):
        self.base_url_55 = epic_55_base_url.rstrip("/")
        self.base_url_100 = epic_100_base_url.rstrip("/")
        self.base_url_masks = masks_base_url.rstrip("/")
        self.base_output = os.path.join(base_output, "EPIC-KITCHENS")
        self.videos_per_split = {}
        self.challenges_splits = []
        self.md5 = {"55": {}, "100": {}, "errata": {}}
        self.errata = {}
        self.parse_splits(splits_path_epic_55, splits_path_epic_100)
        self.load_md5(md5_path)
        self.load_errata(errata_path)
        self.errata_only = errata_only

    def load_errata(self, path):
        with open(path) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=",")

            for row in reader:
                self.errata[row["rdsf_path"]] = row["dropbox_path"]

    def load_md5(self, path):
        with open(path) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=",")

            for row in reader:
                v = row["version"]
                self.md5[v][row["file_remote_path"]] = row["md5"]

    @staticmethod
    def download_file(url, output_path):
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

        try:
            with urllib.request.urlopen(url) as response, open(
                output_path, "wb"
            ) as output_file:
                print("Downloading\nfrom  {}\nto    {}".format(url, output_path))
                shutil.copyfileobj(response, output_file)
        except Exception as e:
            print("Could not download file from {}\nError: {}".format(url, str(e)))

    @staticmethod
    def parse_bool(b):
        return b.lower().strip() in ["true", "yes", "y"]

    @staticmethod
    def md5_checksum(path):
        hash_md5 = hashlib.md5()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        return hash_md5.hexdigest()

    def parse_splits(self, epic_55_splits_path, epic_100_splits_path):
        epic_55_videos = {}

        with open(epic_55_splits_path) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=",")

            for row in reader:
                epic_55_videos[row["video_id"]] = row["split"]

        with open(epic_100_splits_path) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=",")
            self.challenges_splits = [f for f in reader.fieldnames if f != "video_id"]

            for f in self.challenges_splits:
                self.videos_per_split[f] = []

            for row in reader:
                video_id = row["video_id"]
                parts = video_id.split("_")
                participant = int(parts[0].split("P")[1])
                extension = len(parts[1]) == 3
                epic_55_split = None if extension else epic_55_videos[video_id]
                v = {
                    "video_id": video_id,
                    "participant": participant,
                    "participant_str": parts[0],
                    "extension": extension,
                    "epic_55_split": epic_55_split,
                }

                for split in self.challenges_splits:
                    if self.parse_bool(row[split]):
                        self.videos_per_split[split].append(v)

    def download_consent_forms(self, video_dicts):
        files_55 = ["ConsentForm.pdf", "ParticipantsInformationSheet.pdf"]

        for f in files_55:
            output_path = os.path.join(
                self.base_output, "ConsentForms", "EPIC-55-{}".format(f)
            )
            url = "/".join([self.base_url_55, "ConsentForms", f])
            self.download_file(url, output_path)

        output_path = os.path.join(
            self.base_output, "ConsentForms", "EPIC-100-ConsentForm.pdf"
        )
        url = "/".join([self.base_url_100, "ConsentForms", "consent-form.pdf"])
        self.download_file(url, output_path)

    def download_videos(self, video_dicts, file_ext="MP4"):
        def epic_55_parts(d):
            return [
                "videos",
                d["epic_55_split"],
                d["participant_str"],
                "{}.{}".format(d["video_id"], file_ext),
            ]

        def epic_100_parts(d):
            return [
                d["participant_str"],
                "videos",
                "{}.{}".format(d["video_id"], file_ext),
            ]

        self.download_items(video_dicts, epic_55_parts, epic_100_parts)

    def download_rgb_frames(self, video_dicts, file_ext="tar"):
        def epic_55_parts(d):
            return [
                "frames_rgb_flow",
                "rgb",
                d["epic_55_split"],
                d["participant_str"],
                "{}.{}".format(d["video_id"], file_ext),
            ]

        def epic_100_parts(d):
            return [
                d["participant_str"],
                "rgb_frames",
                "{}.{}".format(d["video_id"], file_ext),
            ]

        self.download_items(video_dicts, epic_55_parts, epic_100_parts)

    def download_flow_frames(self, video_dicts, file_ext="tar"):
        def epic_55_parts(d):
            return [
                "frames_rgb_flow",
                "flow",
                d["epic_55_split"],
                d["participant_str"],
                "{}.{}".format(d["video_id"], file_ext),
            ]

        def epic_100_parts(d):
            return [
                d["participant_str"],
                "flow_frames",
                "{}.{}".format(d["video_id"], file_ext),
            ]

        self.download_items(video_dicts, epic_55_parts, epic_100_parts)

    def download_object_detection_images(self, video_dicts, file_ext="tar"):
        # these are available for epic 55 only, but we will use the epic_100_parts func to create a consistent output
        # path
        epic_55_dicts = {k: v for k, v in video_dicts.items() if not v["extension"]}

        def epic_55_parts(d):
            return [
                "object_detection_images",
                d["epic_55_split"],
                d["participant_str"],
                "{}.{}".format(d["video_id"], file_ext),
            ]

        def epic_100_parts(d):
            return [
                d["participant_str"],
                "object_detection_images",
                "{}.{}".format(d["video_id"], file_ext),
            ]

        self.download_items(epic_55_dicts, epic_55_parts, epic_100_parts)

    def download_metadata(self, video_dicts, file_ext="csv"):
        epic_100_dicts = {k: v for k, v in video_dicts.items() if v["extension"]}

        def epic_100_accl_parts(d):
            return [
                d["participant_str"],
                "meta_data",
                "{}-accl.{}".format(d["video_id"], file_ext),
            ]

        def epic_100_gyro_parts(d):
            return [
                d["participant_str"],
                "meta_data",
                "{}-gyro.{}".format(d["video_id"], file_ext),
            ]

        self.download_items(epic_100_dicts, None, epic_100_accl_parts)
        self.download_items(epic_100_dicts, None, epic_100_gyro_parts)

    def download_masks(self, video_dicts, file_ext="pkl"):
        def remote_object_hands_parts(d):
            return [
                "hand-objects",
                d["participant_str"],
                "{}.{}".format(d["video_id"], file_ext),
            ]

        def remote_masks_parts(d):
            return [
                "masks",
                d["participant_str"],
                "{}.{}".format(d["video_id"], file_ext),
            ]

        def output_object_hands_parts(d):
            return [
                d["participant_str"],
                "hand-objects",
                "{}.{}".format(d["video_id"], file_ext),
            ]

        def output_masks_parts(d):
            return [
                d["participant_str"],
                "masks",
                "{}.{}".format(d["video_id"], file_ext),
            ]

        # data is organised in the same way for both epic-55 and the extension so we pass the same functions
        self.download_items(
            video_dicts,
            remote_object_hands_parts,
            remote_object_hands_parts,
            from_url=self.base_url_masks,
            output_parts=output_object_hands_parts,
        )
        self.download_items(
            video_dicts,
            remote_masks_parts,
            remote_masks_parts,
            from_url=self.base_url_masks,
            output_parts=output_masks_parts,
        )

    def download_items(
        self,
        video_dicts,
        epic_55_parts_func,
        epic_100_parts_func,
        from_url=None,
        output_parts=None,
    ):
        for video_id, d in video_dicts.items():
            extension = d["extension"]
            remote_parts = (
                epic_100_parts_func(d) if extension else epic_55_parts_func(d)
            )
            erratum_url = self.errata.get("/".join(remote_parts), None)

            if erratum_url is None:
                if self.errata_only:
                    continue

                if from_url is None:
                    base_url = self.base_url_100 if extension else self.base_url_55
                else:
                    base_url = from_url

                url = "/".join([base_url] + remote_parts)
                version = "100" if extension else "55"
            else:
                print_header("~ Going to download an erratum now! ~", char="~")
                url = erratum_url
                version = "errata"

            output_parts = epic_100_parts_func if output_parts is None else output_parts
            output_path = os.path.join(self.base_output, *output_parts(d))

            if self.file_already_downloaded(output_path, remote_parts, version):
                print(
                    "This file was already downloaded, skipping it: {}".format(
                        output_path
                    )
                )
            else:
                self.download_file(url, output_path)

    def file_already_downloaded(self, output_path, parts, version):
        if not os.path.exists(output_path):
            return False

        key = "/".join(parts)
        remote_md5 = self.md5[version].get(key, None)

        if remote_md5 is None:
            return False

        local_md5 = self.md5_checksum(
            output_path
        )  # we already checked file exists so we are safe here
        return local_md5 == remote_md5

    def download(
        self,
        what=("videos", "rgb_frames", "flow_frames"),
        participants="all",
        specific_videos="all",
        splits="all",
        challenges="all",
        extension_only=False,
        epic55_only=False,
    ):

        video_dicts = {}

        if splits == "all" and challenges == "all":
            download_splits = self.challenges_splits
        elif splits == "all":
            download_splits = [
                cs
                for cs in self.challenges_splits
                for c in challenges
                if c == cs.split("_")[0]
            ]
        elif challenges == "all":
            download_splits = [
                cs
                for cs in self.challenges_splits
                for s in splits
                if s in cs.partition("_")[2]
            ]
        else:
            download_splits = [
                cs
                for cs in self.challenges_splits
                for c in challenges
                for s in splits
                if c == cs.split("_")[0] and s in cs.partition("_")[2]
            ]

        for ds in download_splits:
            if not extension_only and not epic55_only:
                vl = self.videos_per_split[ds]
            else:
                # we know that only one between extension_only and epic_55_only will be True
                vl = [
                    v
                    for v in self.videos_per_split[ds]
                    if (extension_only and v["extension"])
                    or (epic55_only and not v["extension"])
                ]

            if participants != "all" and specific_videos == "all":
                if type(participants[0]) == int:
                    vl = [v for v in vl if v["participant"] in participants]
                else:
                    vl = [v for v in vl if v["participant_str"] in participants]
            if specific_videos != "all" and participants == "all":
                vl = [v for v in vl if v["video_id"] in specific_videos]
            elif participants != "all" and specific_videos != "all":
                if type(participants[0]) == int:
                    vp = [v for v in vl if v["participant"] in participants]
                else:
                    vp = [v for v in vl if v["participant_str"] in participants]
                vs = [v for v in vl if v["video_id"] in specific_videos]
                vl = vp + vs

            video_dicts.update(
                {v["video_id"]: v for v in vl}
            )  # We use a dict to avoid duplicates

        # sorting the dictionary
        video_dicts = {k: video_dicts[k] for k in sorted(video_dicts.keys())}

        if epic55_only:
            source = "EPIC 55"
        elif extension_only:
            source = "EPIC 100 (extension only)"
        else:
            source = "EPIC 100"

        what_str = ", ".join(" ".join(w.split("_")) for w in what)
        if participants == "all":
            participants_str = "all"
        elif type(participants[0]) == int:
            participants_str = ", ".join(["P{:02d}".format(p) for p in participants])
        else:
            participants_str = ", ".join([f"{p}" for p in participants])
        videos_str = (
            "all"
            if specific_videos == "all"
            else ", ".join([f"{v}" for v in specific_videos])
        )

        if not self.errata_only:
            print(
                "Going to download: {}\n"
                "for challenges: {}\n"
                "splits: {}\n"
                "participants: {}\n"
                "specific videos: {}\n"
                "data source: {}".format(
                    what_str, challenges, splits, participants_str, videos_str, source
                )
            )

        for w in what:
            if not self.errata_only:
                print_header(
                    "| Downloading {} now |".format(" ".join(w.split("_"))), char="-"
                )

            func = getattr(self, "download_{}".format(w))
            func(video_dicts)
