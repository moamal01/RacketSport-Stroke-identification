# Racket sport stroke identification
![screenshot](figures/high-level_overview.png)


## Project Structure

- **folder structure**:

```plaintext
.
├── LICENSE
├── README.md
├── __pycache__
│   └── utility_functions.cpython-311.pyc
├── clip_captions.json
├── cropped
│   ├── video_1
│   ├── video_1m
│   ├── video_2
│   ├── video_2m
│   ├── video_3
│   └── video_3m
├── data
│   ├── events
│   ├── splits
│   ├── splits_old
│   ├── video_1
│   ├── video_2
│   └── video_3
├── embeddings
│   ├── text
│   ├── video_1
│   ├── video_1m
│   ├── video_2
│   ├── video_2m
│   ├── video_3
│   └── video_3m
├── figures
│   ├── t-snes
│   └── umaps
├── notebooks
│   ├── cropped
│   ├── empty_event_keys.json
│   ├── empty_event_keys2.json
│   ├── empty_event_keys_1_frame.json
│   ├── extract_frames.ipynb
│   ├── low_qual.ipynb
│   └── show_crops.ipynb
├── pyproject.toml
├── requirements.txt
├── src
│   ├── annotation
│   │   └── annotate.py
│   ├── augement
│   │   ├── demo.py
│   │   └── mirror_keypoints.py
│   ├── classification
│   │   ├── data_split.py
│   │   ├── data_split_oversampling.py
│   │   ├── log_reg.py
│   │   └── log_reg_better.py
│   ├── clip
│   │   ├── crop_objects.py
│   │   ├── crop_people.py
│   │   ├── extract_image_embeddings.py
│   │   ├── extract_text_embeddings.py
│   │   └── read_embedding.py
│   ├── data_utils
│   │   └── dataloader.py
│   ├── dectectron2.py
│   ├── extract_hits.py
│   ├── plotting
│   │   ├── dbscan.py
│   │   ├── distance_plot.py
│   │   ├── joint_dbscan.py
│   │   ├── joint_time_dbscan.py
│   │   ├── nn.py
│   │   ├── pca_plot.py
│   │   ├── scatter_plot.py
│   │   ├── t-sne.py
│   │   └── umap_plotting.py
│   ├── post_processing
│   │   ├── combine_tables.py
│   │   ├── distance_to_midtpoints.py
│   │   ├── midpoints.py
│   │   └── normalize.py
│   └── verify.py
├── utility_functions.py
└── videos
    ├── game_1.mp4
    ├── game_1f.mp4
    ├── game_2.mp4
    ├── game_2f.mp4
    ├── game_3.mp4
    └── game_3f.mp4
```

# Annotating Data
The annotation process consists of two main steps:

1. Identifying stroke frames — Marking the frames where a stroke occurs.
2. Annotating stroke frames — Labeling each identified frame with the corresponding stroke type.
3. Add phase classes. 

The relevant python scripts are found under `src/annotation`.

1. Start by adding the video you want to annotate to the **videos** folder. It must be prefixed with "game_" and have a unique suffix. The suffix is currently a number. 
2. Then add or create a json file with the frames you wish to annotate with frames as keys and **empty_event** as value.  

## Marking Stroke Frames - add_events.py
Allows for traversing a video, frame by frame and adding a specified event. This event is currently set to **empty_event** in the **event_name** field at the top of the script.

### Variables

* **video_path**: Path to the video.
* **json_path**: Path to the json file.
* **start_from**: Frame to start the stroke markation process from. 
* **event_name**: Name of the event you want to add.

The script will display the frame specified in the **start_from** variable. From here, you have four options.
* a — Go 1 frame back.
* d — Go 1 frame forward.
* s — Go 10 frames forward.
* space — Add or overwrite event to the current frame.

All changes are saved automatically after each event is added to prevent data loss.

The json file will be populated with values such as:
   
    ```json
    {
      "123": "empty_event",
      "456": "empty_event"
    }
    ```

## Annotating Stroke Frames - annotate.py

Enables annotations of frames that are labelled with a specific value. This value is currently set to **empty_event** in the **label_of_interest** field at the top of the script.

1. Specify the path to the video and json file at the top of the `annotate.py` script.  
2. Run the script.


The script will show you the frames you have specified in the json file, and request the label in the terminal. Predefined codes has been specified to speed the process up. These codes are currently just short hand notation, for example **lbl** is short for **left_backhand_loop**.

To leave a label as is, simply leave the prompt empty and press 'enter'.

The labels will be updated as soon as the script is prompted with the new label.

To check other labels, simply change the **label_of_interest** field.

## Add Phase Events - add_phase_events.py
To add phase classes, simply execute the `add_phase_events.py` file.

1. Specify the path to the annotated json file.
2. The script will save a new json file to the `data/extended_events` folder.

# Extracting Mask-RCNN Features
Once the relevant frames the Mask-RCNN model can be applied. This is done with `detectron-full_video.py`.

### Variables

* video — Suffix of video name.
* efficient — If is to true, the script will only process frames in intervals around annotated frames. Otherwise, every frame in the video will be processed.
* start_at — Frame number where the video processing starts from.
* video_path — Path to the video.
* Threshold — Decides how many frames around the annotated frame that should be processed.
* write_video — Create video with overlayed Mask-RCNN detections.


**Mask R-CNN for object detection**  
The bjects of interest are:
- `0` — Person  
- `32` — Sports ball  
- `38` — Tennis racket  
- `60` — Dining table

The results are saved to a bbox_video{video}.csv.

| **Event Frame** | **Class ID** | **Score** | **BBoxes**         |
|------------------|--------------|-----------|---------------------|

**Mask R-CNN adaptation for skeletal keypoint detection**  
The Results are saved to keypoints_video{video}.csv.

| **Path**         | **Event Frame** | **Keypoints**     | **People Boxes**   | **People Scores**   |
|------------------|------------------|--------------------|---------------------|----------------------|

# Post Processing
The post processing contain several steps to structure and make the data interpretable for models later on.

All the post processing scripts are located in `src/post_processing`.

## Combining Tables — combine_tables.py
As both the bbox_video{video}.csv and keypoints_video{video}.csv has the `Event Frame` key, they can be combined using this value. 

### Variables
* video — Suffix of video name.
* keypoints_file — Path to the keypoints_video{video}.csv.
* bbox_file — Path to the bbox_video{video}.csv.

This file produces a table with combined output of the two other csv files and outputs it to `merged_output_video{video}`

| **Path**   | **Event Frame** | **Keypoints**     | **People Boxes**   | **People Scores**    | **Ball Boxes**     | **Ball Scores**     | **Racket Boxes**     | **Racket Scores** | **Table Boxes**     | **Table Scores**       |
|------------|------------------|--------------------|---------------------|------------------------|---------------------|----------------------|-----------------------|-------------------|----------------------|-------------------------|

## Min-Max Normalization — normalize.py

## Midpoint Normalization — midpoints.py

# Extracting CLIP Embeddings

# Training a model


## Order of Execution
1. Get keypoints
2. Combine the tables
3. Normalize
4. Find midpoints. Can also find mirrored


### Third Party Dependencies and licenses

- **CLIP**:
- **Detectron2**:

## Authors and Acknowledgment

This project was started by **Moamal Fadhil Abdul-Mahdi**.