import torch
import os
from PIL import Image
from torch.utils.data import Dataset

class FlickrDataset(Dataset):
    def __init__(self, root_folder, captions_file, vocab, transform=None):
        self.root_folder = root_folder
        self.df = [line.strip().split(',') for line in open(captions_file, 'r') if len(line.strip()) > 0]
        self.transform = transform
        self.vocab = vocab
        self.captions = [line[1] for line in self.df]
        self.vocab.build_vocab(self.captions)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id, caption = self.df[index]
        img_path = os.path.join(self.root_folder, img_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return image, torch.tensor(numericalized_caption)
