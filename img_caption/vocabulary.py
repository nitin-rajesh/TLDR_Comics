import nltk
import os
import torch
from PIL import Image
from torchvision import transforms
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')

# Basic Vocabulary class
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        return nltk.tokenize.word_tokenize(text.lower())

    def build_vocab(self, sentence_list):
        freqs = Counter()
        for sentence in sentence_list:
            freqs.update(self.tokenizer(sentence))
        
        for word, freq in freqs.items():
            if freq >= self.freq_threshold:
                idx = len(self.itos)
                self.stoi[word] = idx
                self.itos[idx] = word

    def numericalize(self, text):
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in self.tokenizer(text)
        ]
 