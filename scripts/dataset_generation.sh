#!/bin/bash

CLEAN_AUDIO_DIR=/user/es.aau.dk/xu68nv/data/GRID
SPEAKERS="1 2 3 4"
DEST_DIR=/user/es.aau.dk/xu68nv/data/GRID/syn_data/full_av_dataset/test-set-si
AUDIO_LEN=3000
NUM_SAMPLES=0
N_MAX_INTR=1
COV_MEAN=1
COV_STD=0
EXT=wav


cd /user/es.aau.dk/xu68nv/code/AudioVisualSpeechInpainting

time python -u speech_inpainting_main.py dataset_generator \
	--clean_audio_dir $CLEAN_AUDIO_DIR \
	--speaker_ids $SPEAKERS \
	--dest_dir $DEST_DIR \
	--num_samples $NUM_SAMPLES \
	--audio_length $AUDIO_LEN \
	--num_max_intr $N_MAX_INTR \
	--mask_coverage_mean $COV_MEAN \
	--mask_coverage_std $COV_STD \
	--ext $EXT
