import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from .glove_converter import GloveEmbeddingConverter as GEC

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir: str, captions_file: str, glove_ec: GEC, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.glove_ec = glove_ec
        # Load (image, caption) pairs
        self.data = []
        with open(captions_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                img_name, caption = line.split(',', 1)
                self.data.append((img_name.strip(), caption.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, caption = self.data[index]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption to token IDs
        caption_tensor = self.glove_ec.caption_to_indices(caption)

        return image, caption_tensor
