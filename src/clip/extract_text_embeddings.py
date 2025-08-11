import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import json
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

with open("clip_captions.json", "r") as f:
    clip_captions = json.load(f)

for key, grouping in clip_captions.items():
    for caption in grouping:
        text_inputs = processor(text=caption, return_tensors="pt", padding=True)

        with torch.no_grad():
            text_embeddings = model.get_text_features(**text_inputs)

        text_embeddings_np = text_embeddings.cpu().numpy()

        directory = f"embeddings/text/{key}/{caption}/"
        os.makedirs(directory)
        np.save(f"{directory}/embedding.npy", text_embeddings_np)
