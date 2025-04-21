import torchvision.models as models
import torch
import torch.nn as nn

class FastDecoderGRU(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)

        self.fc_out = nn.Linear(hidden_size, vocab_size)

        self.feat2hidden = nn.Linear(embed_size, hidden_size)

    def forward(self, features, captions):

        embeddings = self.embedding(captions)  # [B, T, E]

        h0 = self.feat2hidden(features).unsqueeze(0)  # [1, B, H]

        outputs, _ = self.gru(embeddings, h0)  # outputs: [B, T, H]

        outputs = self.fc_out(outputs)  # [B, T, vocab_size]
        return outputs
