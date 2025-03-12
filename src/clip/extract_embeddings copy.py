import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
#import matplotlib.pyplot as plt
import json
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

with open("clip_captions.json", "r") as f:
    clip_captions = json.load(f)


with open("notebooks/empty_event_keys.json", "r") as keypoint_file:
    loaded_keys = json.load(keypoint_file)

# Process images and text
for key_frame, _ in loaded_keys.items():
    for key, grouping in clip_captions.items():
        frame = int(key_frame) - 2
        for i in range(5):
            print(f"key_frame: {key_frame}")

            image_paths = [
                f"cropped/video_1/{frame}/0/left.png",
                f"cropped/video_1/{frame}/0/right.png"
            ]

            images = [Image.open(path).convert("RGB") for path in image_paths]

            image_inputs = processor(images=images, return_tensors="pt", do_convert_rgb=False)
            text_inputs = processor(text=grouping, return_tensors="pt", padding=True)

            # Extract embeddings
            with torch.no_grad():
                image_embeddings = model.get_image_features(**image_inputs)
                text_embeddings = model.get_text_features(**text_inputs)

            # Convert embeddings to lists (to save in CSV)
            image_embeddings_list = image_embeddings.cpu().numpy().tolist()  # Convert to list
            text_embeddings_list = text_embeddings.cpu().numpy().tolist()  # Convert to list

            # Convert to NumPy
            image_embeddings_np = image_embeddings.cpu().numpy()
            text_embeddings_np = text_embeddings.cpu().numpy()

            # Print embedding shapes
            print(f"Image Embeddings Shape: {image_embeddings_np.shape}")  # (num_images, embedding_dim)
            print(f"Text Embeddings Shape: {text_embeddings_np.shape}")    # (num_texts, embedding_dim)

            # Save embeddings if needed
            directory = f"imbeddings/{frame}/{key}/"
            os.makedirs(directory)
            np.save(directory + "image_embeddings.npy", image_embeddings_np)
            np.save(directory + "text_embeddings.npy", text_embeddings_np)
            
            frame += 1
        

def image_grid(imgs, cols):
    rows = (len(imgs) + cols - 1) // cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid 

# Compute similarity using cosine similarity (dot product of normalized embeddings)
# image_paths = [
#     'cropped/video_1/56/0/left.png', 
#     'cropped/video_1/44175/0/left.png', 
#     'cropped/video_1/79199/0/left.png',
# ]

# images = [Image.open(path).convert("RGB") for path in image_paths]

# similarity_scores = image_embeddings @ text_embeddings.T  # Shape: (num_images, num_texts)
# probs = similarity_scores.softmax(dim=1)  # Normalize scores using softmax

# # Visualization
# fig = plt.figure(figsize=(10, 10))

# for idx in range(len(images)):
#     # Show original image
#     fig.add_subplot(len(images), 2, 2 * (idx + 1) - 1)
#     plt.imshow(images[idx])
#     plt.xticks([])
#     plt.yticks([])

#     # Show probabilities
#     fig.add_subplot(len(images), 2, 2 * (idx + 1))
#     plt.barh(range(len(probs[0].detach().numpy())), probs[idx].detach().numpy(), tick_label=leg_foot_position)
#     plt.xlim(0, 1.0)

#     plt.subplots_adjust(
#         left=0.1,
#         bottom=0.1,
#         right=0.9,
#         top=0.9,
#         wspace=0.2,
#         hspace=0.2
#     )

# plt.show()