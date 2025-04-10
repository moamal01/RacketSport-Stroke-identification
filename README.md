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
├── embeddings_old
│   ├── video_1
│   └── video_2
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
│   ├── augement
│   ├── classification
│   ├── clip
│   ├── data_utils
│   ├── dectectron2.py
│   ├── extract_hits.py
│   ├── plotting
│   ├── post_processing
│   └── verify.py
├── test.mp4
├── test2.mp4
├── utility_functions.py
└── videos
    ├── game_1.mp4
    ├── game_1f.mp4
    ├── game_2.mp4
    ├── game_2f.mp4
    ├── game_3.mp4
    └── game_3f.mp4
```