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
    "Person's legs are shoulder-width apart",
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


inputs = processor(text=leg_foot_position, images=images, return_tensors="pt", padding=True, do_convert_rgb=False)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

fig = plt.figure(figsize=(10, 10))

for idx in range(len(images)):
    # show original image
    fig.add_subplot(len(images), 2, 2*(idx+1)-1 )
    plt.imshow(images[idx])
    plt.xticks([])
    plt.yticks([])

    # show probabilities
    fig.add_subplot(len(images), 2, 2*(idx+1))
    plt.barh(range(len(probs[0].detach().numpy())),probs[idx].detach().numpy(), tick_label=leg_foot_position)
    plt.xlim(0,1.0)

    plt.subplots_adjust(
        left=0.1,
        bottom=0.1,
        right=0.9,
        top=0.9,
        wspace=0.2,
        hspace=0.2
    )

plt.show()