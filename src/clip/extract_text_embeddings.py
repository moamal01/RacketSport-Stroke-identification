import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import json
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

with open("clip_captions.json", "r") as f:
    clip_captions = json.load(f)


with open("notebooks/empty_event_keys.json", "r") as keypoint_file:
    loaded_keys = json.load(keypoint_file)

for key, grouping in clip_captions.items():
    text_inputs = processor(text=grouping, return_tensors="pt", padding=True)

    # Extract embeddings
    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_inputs)

    text_embeddings_np = text_embeddings.cpu().numpy()

    print(f"Text Embeddings Shape: {text_embeddings_np.shape}")

    directory = f"imbeddings/text/{key}/"
    os.makedirs(directory)
    np.save(directory + "embeddings.npy", text_embeddings_np)
