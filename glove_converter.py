import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk import word_tokenize
import ssl
from torch.nn.utils.rnn import pad_sequence

class GloveEmbeddingConverter:
    def __init__(self, file_path):

        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        nltk.download('punkt_tab')

        self.embeddings = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                self.embeddings[word] = vector
                if len(self.embeddings) > 30000:
                    break

        self.embedding_dim = len(self.embeddings['the'])

        self.embeddings["<SOS>"] = np.random.uniform(-0.5, 0.5, self.embedding_dim)
        self.embeddings["<EOS>"] = np.random.uniform(-0.5, 0.5, self.embedding_dim)
        self.embeddings["<UNK>"] = np.random.uniform(-0.5, 0.5, self.embedding_dim)
        self.embeddings["<PAD>"] = np.zeros(self.embedding_dim)

        special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        vocab_tokens = [w for w in self.embeddings.keys() if w not in special_tokens]
        self.all_tokens = special_tokens + vocab_tokens

        self.word2idx = {word: idx for idx, word in enumerate(self.all_tokens)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def caption_to_embeddings(self, caption):
        caption_vectors = []
        caption_tokens = word_tokenize(caption.lower())
        caption_tokens.insert(0,"<SOS>")
        caption_tokens.append("<EOS>")

        for word in caption_tokens:
            if word in self.embeddings:
                caption_vectors.append(self.embeddings[word])
            else:
                caption_vectors.append(self.embeddings["<UNK>"])  # Handle unknown words
        
        caption_vectors_array = np.array(caption_vectors, dtype=np.float32)
        caption_tensor = torch.tensor(caption_vectors_array, dtype=torch.float)

        return caption_tensor
    
    def caption_to_indices(self, caption):
        caption_indices = []
        caption_tokens = word_tokenize(caption.lower())
        caption_tokens.insert(0, "<SOS>")
        caption_tokens.append("<EOS>")

        for word in caption_tokens:
            if word in self.word2idx:
                caption_indices.append(self.word2idx[word])
            else:
                caption_indices.append(self.word2idx["<UNK>"])  # Fallback for unknowns

        return torch.tensor(caption_indices, dtype=torch.long)

    
    def get_sos_eos(self):
        return (self.embeddings["<SOS>"], self.embeddings["<EOS>"])

    def get_vocab_size(self):
        return len(self.embeddings)
    
    def build_embedding_matrix(self):
        # Building embedding matrix as per vocab order
        embedding_matrix = np.zeros((len(self.all_tokens), self.embedding_dim), dtype=np.float32)
        for idx, word in enumerate(self.all_tokens):
            embedding_matrix[idx] = self.embeddings.get(word, self.embeddings["<UNK>"])
        
        return torch.tensor(embedding_matrix, dtype=torch.float)
    

def collate_fn_with_padding(batch):

    imgs, captions = zip(*batch)
    imgs = torch.stack(imgs, 0)    

    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    
    return imgs, captions
