import torch
from torch import nn
from PIL import Image

class CaptionGenerator:
    def __init__(self, encoder, decoder, idx_to_token, token_to_idx, transform, device='cuda', max_length=32):
        self.encoder = encoder.to(device).eval()
        self.decoder = decoder.to(device).eval()
        self.idx_to_token = idx_to_token
        self.token_to_idx = token_to_idx
        self.transform = transform
        self.device = device
        self.max_length = max_length
        self.hidden_dim = decoder.hidden_dim 
        self.encoder_dim = encoder.encoder_dim
        self.feat2hidden = nn.Linear(self.encoder_dim, self.hidden_dim)

        
    def preprocess_image(self, image):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image_tensor

    def generate(self, image_path):

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess_image(image)

        with torch.no_grad():
            features = self.encoder(image_tensor) # Shape is (hidden_dim)
            print(features.shape)

            # Ensure features shape is (1, 1, hidden_dim)
            if features.dim() == 1:
                features = features.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
            elif features.dim() == 2:
                features = features.unsqueeze(0)  # (1, 1, hidden_dim)
            else:
                pass 

            hidden = self.feat2hidden(features.squeeze(0)).unsqueeze(0)
            print(hidden)
            print(features)
            caption_idxs = [self.token_to_idx["<SOS>"]]
            input_token = torch.tensor([[caption_idxs[-1]]], device=self.device)

            for _ in range(self.max_length):
                outputs, hidden = self.decoder(input_token, hidden)
                next_token_logits = outputs.squeeze(1).squeeze(0)
                next_token_idx = next_token_logits.argmax().item()

                if next_token_idx == self.token_to_idx["<EOS>"]:
                    break

                caption_idxs.append(next_token_idx)
                input_token = torch.tensor([[next_token_idx]], device=self.device)

        caption_words = [self.idx_to_token[idx] for idx in caption_idxs[1:]]  # exclude <SOS>
        return ' '.join(caption_words)
