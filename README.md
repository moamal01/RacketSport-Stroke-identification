# Racket sport stroke identification

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

## Order of Execution


### Third Party Dependencies and licenses

- **CLIP**:
- **Detectron2**:

## Authors and Acknowledgment

This project was started by **Moamal Fadhil Abdul-Mahdi**.