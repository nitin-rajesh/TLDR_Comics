import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions_file, vocab, transform=None):
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform

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
        tokens = self.vocab.tokenize(caption)
        caption_ids = [self.vocab.stoi["<SOS>"]] + \
                      [self.vocab.stoi.get(token, self.vocab.stoi["<UNK>"]) for token in tokens] + \
                      [self.vocab.stoi["<EOS>"]]
        caption_tensor = torch.tensor(caption_ids)

        return image, caption_tensor
