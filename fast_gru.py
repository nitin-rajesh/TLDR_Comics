import torchvision.models as models
import torch
import torch.nn as nn

class FastDecoderGRU(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, embeddings=None):
        super(FastDecoderGRU, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.tensor(embeddings, dtype=torch.float))
            self.embedding.weight.requires_grad = False  # Freeze embeddings
        
        # GRU layer
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
        # Fully connected layer to predict words
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, captions, hidden):
        captions = captions.long()
        # Embedding the input captions
        embedded = self.embedding(captions)  # Shape: (batch_size, seq_length, embedding_dim)
        
        # GRU forward pass
        output, hidden = self.gru(embedded, hidden)  # Output shape: (batch_size, seq_length, hidden_dim)
        
        # Predict next words
        predictions = self.fc(output)  # Shape: (batch_size, seq_length, vocab_size)
        return predictions, hidden