import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size, num_heads=8, num_layers=3, max_len=50):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embed_size))

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.feat2embed = nn.Linear(256, embed_size)  
        
    def forward(self, image_features, tgt_captions):

        B, T = tgt_captions.shape

        # Prepare target embeddings
        tgt_emb = self.embed(tgt_captions) + self.positional_encoding[:, :T, :]  # [B, T, E]
        tgt_emb = tgt_emb.permute(1, 0, 2)  # [T, B, E]

        # Prepare image features as memory
        memory = self.feat2embed(image_features).unsqueeze(0)  # [1, B, E]

        # Create mask for autoregressive decoding
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(image_features.device)

        out = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        out = self.fc_out(out)  # [T, B, vocab]
        return out.permute(1, 0, 2)  # [B, T, vocab]
