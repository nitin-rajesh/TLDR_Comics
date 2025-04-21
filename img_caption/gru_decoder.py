import torchvision.models as models
import torch
import torch.nn as nn

class DecoderGRU(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        
        # GRU parameters
        self.Wz = nn.Linear(embed_size + hidden_size, hidden_size)
        self.Wr = nn.Linear(embed_size + hidden_size, hidden_size)
        self.Wh = nn.Linear(embed_size + hidden_size, hidden_size)
        
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        batch_size, seq_len = captions.size()
        embeddings = self.embedding(captions)  # (B, T, E)

        h = features  # initial hidden state from image

        outputs = []
        for t in range(seq_len):
            x_t = embeddings[:, t, :]
            combined = torch.cat((x_t, h), dim=1)
            
            z = torch.sigmoid(self.Wz(combined))
            r = torch.sigmoid(self.Wr(combined))
            combined_reset = torch.cat((x_t, r * h), dim=1)
            h_tilde = torch.tanh(self.Wh(combined_reset))
            h = (1 - z) * h + z * h_tilde

            out = self.fc_out(h)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)
        return outputs
