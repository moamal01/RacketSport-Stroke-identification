import numpy as np
import torch
import csv
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def image_grid(imgs, cols):
    rows = (len(imgs) + cols - 1) // cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

image_paths = [
    'cropped/video_1/56/0/left.png', 
    'cropped/video_1/44175/0/left.png', 
    'cropped/video_1/79199/0/left.png',
    ]

images = [Image.open(path).convert("RGB") for path in image_paths]


# Body Orientation
body_orientation = [
    "Person is standing upright",
    "Person is leaning forward",
    "Person is leaning backward",
    "Person is leaning forward",
]

# Arm and Hand Position
arm_hand_position = [
    "One arm is raised to the head",
    "One arm is stretched out to the side",
    "Both arms are stretched out to the sides",
    "One arm is infront of the body",
]

# Leg and Foot Position
leg_foot_position = [
    "One foot is lifted off the ground",
    "Person is stepping forward",
    "Person's legs are wide apart",
    "Person's legs are close together",
    #"Person's legs are shoulder-width apart",
]

# Motion-Based Descriptions
motion_based = [
    "Person is mid-swing",
    "Person is preparing to hit something",
    "Person is following through after a motion",
    "Person is in a crouched position with arms raised",
    "Person is moving one arm forward quickly",
    "Person is moving one arm across the body",
    "Person is swinging their arm downward",
    "Person is holding an object and moving it forward"
]

# General Pose Descriptions
general_pose = [
    "Person is in an active stance",
    "Person is in a relaxed stance",
    "Person is in a defensive position",
    "Person is stretching to reach something",
    "Person is twisting their torso",
    "Person is bending their knees"
]

# Question-Based Prompts for CLIP
question_based = [
    "Is the person leaning forward?",
    "Is the person in mid-air?",
    "Does the person have one arm raised above their head?",
    "Is the person moving one arm forward?",
    "Is the person stepping forward?",
    "Is the person jumping?",
    "Is the person in a crouched position?",
    "Is the person's hand fully extended?"
]

# Contrasting Captions for Better CLIP Comparison
contrasting_pairs = [
    "Person is leaning forward", "Person is leaning backward", "Person is standing upright",
    "One leg is lifted", "Both feet are on the ground",
    "Person is jumping", "Person is stepping forward",
    "Person is in an active stance", "Person is in a relaxed stance", "Person is in a crouched stance"
]

# Combine into a dictionary for easy access
clip_captions = {
    "body_orientation": body_orientation,
    "arm_hand_position": arm_hand_position,
    "leg_foot_position": leg_foot_position,
    "motion_based": motion_based,
    "general_pose": general_pose,
    "question_based": question_based,
    "contrasting_pairs": contrasting_pairs
}


# Process images and text
image_inputs = processor(images=images, return_tensors="pt", do_convert_rgb=False)
text_inputs = processor(text=leg_foot_position, return_tensors="pt", padding=True)

# Extract embeddings
with torch.no_grad():
    image_embeddings = model.get_image_features(**image_inputs)
    text_embeddings = model.get_text_features(**text_inputs)
    
# Convert embeddings to lists (to save in CSV)
image_embeddings_list = image_embeddings.cpu().numpy().tolist()  # Convert to list
text_embeddings_list = text_embeddings.cpu().numpy().tolist()  # Convert to list

# Save to CSV
embedding_dim = len(image_embeddings_list[0])
with open('image_embeddings.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([f"Image_Embedding_{i}" for i in range(embedding_dim)])  # Column headers
    writer.writerows(image_embeddings_list)  # Write image embeddings

with open('text_embeddings.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([f"Text_Embedding_{i}" for i in range(embedding_dim)])  # Column headers
    writer.writerows(text_embeddings_list)  # Write text embeddings

# Normalize embeddings for cosine similarity
#image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
#text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

# Convert to NumPy
image_embeddings_np = image_embeddings.cpu().numpy()
text_embeddings_np = text_embeddings.cpu().numpy()

# Print embedding shapes
print(f"Image Embeddings Shape: {image_embeddings_np.shape}")  # (num_images, embedding_dim)
print(f"Text Embeddings Shape: {text_embeddings_np.shape}")    # (num_texts, embedding_dim)

# Save embeddings if needed
np.save("image_embeddings.npy", image_embeddings_np)
np.save("text_embeddings.npy", text_embeddings_np)

# Compute similarity using cosine similarity (dot product of normalized embeddings)
similarity_scores = image_embeddings @ text_embeddings.T  # Shape: (num_images, num_texts)
probs = similarity_scores.softmax(dim=1)  # Normalize scores using softmax

# Visualization
fig = plt.figure(figsize=(10, 10))

for idx in range(len(images)):
    # Show original image
    fig.add_subplot(len(images), 2, 2 * (idx + 1) - 1)
    plt.imshow(images[idx])
    plt.xticks([])
    plt.yticks([])

    # Show probabilities
    fig.add_subplot(len(images), 2, 2 * (idx + 1))
    plt.barh(range(len(probs[0].detach().numpy())), probs[idx].detach().numpy(), tick_label=leg_foot_position)
    plt.xlim(0, 1.0)

    plt.subplots_adjust(
        left=0.1,
        bottom=0.1,
        right=0.9,
        top=0.9,
        wspace=0.2,
        hspace=0.2
    )

#plt.show()