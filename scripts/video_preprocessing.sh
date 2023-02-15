#!/bin/bash

DATA_DIR=/user/es.aau.dk/xu68nv/data/GRID
SPEAKER_IDS="24 25 26 27 28 29 30 31 32 33 34"
VIDEO_DIR=mpg_vcd
DEST_DIR=landmarks
SHAPE_PREDICTOR=/user/es.aau.dk/xu68nv/data/shape_predictor_68_face_landmarks.dat
EXT=mpg

cd /user/es.aau.dk/xu68nv/code/AudioVisualSpeechInpainting

time python -u speech_inpainting_main.py video_preprocessing \
	--data_dir $DATA_DIR \
	--speaker_ids $SPEAKER_IDS \
	--video_dir $VIDEO_DIR \
	--dest_dir $DEST_DIR \
	--shape_predictor $SHAPE_PREDICTOR \
	--ext $EXT 
