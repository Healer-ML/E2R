from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FRSCMRDataset(Dataset):
    """CSV-backed image-caption dataset used by the E2R training code."""

    def __init__(self, csv_file, image_root, transform, image_key="filename", caption_key="title"):
        self.records = pd.read_csv(csv_file)
        self.image_root = Path(image_root)
        self.transform = transform
        self.image_key = image_key
        self.caption_key = caption_key

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]
        image = Image.open(self.image_root / str(row[self.image_key])).convert("RGB")
        return index, self.transform(image), str(row[self.caption_key]), -1
