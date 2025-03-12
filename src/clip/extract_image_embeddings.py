import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import json
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def save_people_embedding(paths):
    images = [Image.open("cropped/" + path).convert("RGB") for path in paths]
    image_inputs = processor(images=images, return_tensors="pt", do_convert_rgb=False)

    # Extract embeddings
    with torch.no_grad():
        image_embeddings = model.get_image_features(**image_inputs)

    # Save embeddings
    path = '/'.join(paths[0].split('/')[:-1])
    directory = "imbeddings/" + path
    os.makedirs(directory)
    
    image_embeddings_np = image_embeddings.cpu().numpy()
    np.save(directory + "/image_embeddings.npy", image_embeddings_np)

def save_object_embedding(path):
    image = Image.open("cropped/" + path)
    image_inputs = processor(images=image, return_tensors="pt", do_convert_rgb=False)

    # Extract embeddings
    with torch.no_grad():
        image_embeddings = model.get_image_features(**image_inputs)

    # Save embeddings
    path = '/'.join(path.split('/')[:-1])
    directory = "imbeddings/" + path
    os.makedirs(directory)
    
    image_embeddings_np = image_embeddings.cpu().numpy()
    np.save(directory + "/image_embeddings.npy", image_embeddings_np)


with open("notebooks/empty_event_keys.json", "r") as keypoint_file:
    loaded_keys = json.load(keypoint_file)

for key_frame, _ in loaded_keys.items():
    frame = int(key_frame) - 2
    for i in range(5):
        save_people_embedding([f"video_1/{frame}/0/left.png", f"video_1/{frame}/0/right.png"])
        
        if os.path.exists(f"cropped/video_1/{frame}/32"):
            save_object_embedding(f"video_1/{frame}/32/object.png")
                
        if os.path.exists(f"cropped/video_1/{frame}/38"):
            save_object_embedding(f"video_1/{frame}/38/object.png")
            
        if os.path.exists(f"cropped/video_1/{frame}/60"):
            save_object_embedding(f"video_1/{frame}/60/object.png")

        frame += 1
    

def image_grid(imgs, cols):
    rows = (len(imgs) + cols - 1) // cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid 