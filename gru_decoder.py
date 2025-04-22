import torchvision.models as models
import torch
import torch.nn as nn

class DecoderGRU(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, embeddings = None,num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.tensor(embeddings, dtype=torch.float))
            self.embedding.weight.requires_grad = False  # Freeze embeddings

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        
        # GRU parameters
        self.Wz = nn.Linear(embed_size + hidden_size, hidden_size)
        self.Wr = nn.Linear(embed_size + hidden_size, hidden_size)
        self.Wh = nn.Linear(embed_size + hidden_size, hidden_size)
        
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        print(embed_size)
        # self.feat2embed = nn.Linear(256, embed_size)


    def forward(self, features, captions):
        # print(captions.shape)
        # print(len(captions.size()))
        # print(features.shape)
        batch_size, seq_len, embed_dim = captions.size()

        h = features.unsqueeze(0)  # initial hidden state from image

        # features = self.feat2embed(features)  
        # features = features.unsqueeze(1)      

        embeddings = self.embedding(captions)    

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
