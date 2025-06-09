import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import json
import os
import pandas as pd
from tqdm import tqdm

video = 4
mirror = False
full_video = False

if mirror:
    m = "m"
else:
    m = ""
    
file_path = f"../../data/video_{video}/midpoints_video{video}.csv"
df = pd.read_csv(file_path)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def save_people_embedding(path):    
    full_path = "../../cropped/" + path
    if not os.path.exists(full_path):
        return 
    
    image = Image.open(full_path).convert("RGB")    
    image_inputs = processor(images=image, return_tensors="pt", do_convert_rgb=False)

    # Extract embedding
    with torch.no_grad():
        image_embeddings = model.get_image_features(**image_inputs)

    # Save embedding
    directory = "../../embeddings/" + '/'.join(path.split('/')[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    image_embeddings_np = image_embeddings.cpu().numpy()
    
    if "left" in path:
        np.save(directory + "/left.npy", image_embeddings_np)
        pass
    else:
        pass
        np.save(directory + "/right.npy", image_embeddings_np)

def save_object_embedding(path):
    image = Image.open("../../cropped/" + path)
    image_inputs = processor(images=image, return_tensors="pt", do_convert_rgb=False)

    # Extract embeddings
    with torch.no_grad():
        image_embeddings = model.get_image_features(**image_inputs)

    # Save embeddings
    path = '/'.join(path.split('/')[:-1])
    directory = "../../embeddings/" + path
    os.makedirs(directory)
    
    image_embeddings_np = image_embeddings.cpu().numpy()
    np.save(directory + "/image_embeddings.npy", image_embeddings_np)


with open(f"../../data/extended_events/events_markup{video}.json", "r") as keypoint_file:
    data = json.load(keypoint_file)
    
excluded_values = {"empty_event", "bounce", "net"}
loaded_keys = {k: v for k, v in data.items() if v not in excluded_values}

for _, row in tqdm(df.iterrows(), total=len(df)):
    event_frame = row["Event frame"]
    
    if not full_video:
        if str(event_frame) not in loaded_keys:
            continue
    
    for i in range(1):
        save_people_embedding(f"video_{video}{m}/{event_frame}/0/left.png")
        save_people_embedding(f"video_{video}{m}/{event_frame}/0/right.png")
    

def image_grid(imgs, cols):
    rows = (len(imgs) + cols - 1) // cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid 