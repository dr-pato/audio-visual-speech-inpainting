#!/bin/bash

DATA_PATH=/user/es.aau.dk/xu68nv/data/GRID/tfrecords/full_av_dataset_asr/test-set
AUDIO_PATH=/user/es.aau.dk/xu68nv/data/GRID/syn_data/full_av_dataset/test-set
TFRECORD_MODE=fixed
AUDIO_FEAT_DIM=257
VIDEO_FEAT_DIM=136
NUM_AUDIO_SAMPLES=48000
BATCH_SIZE=32

cd /user/es.aau.dk/xu68nv/code/AudioVisualSpeechInpainting

time python -u speech_inpainting_main.py masking \
	--data_dir $DATA_PATH \
	--audio_dir $AUDIO_PATH \
	--mode $TFRECORD_MODE \
	--audio_feat_dim $AUDIO_FEAT_DIM \
	--video_feat_dim $VIDEO_FEAT_DIM \
	--num_audio_samples $NUM_AUDIO_SAMPLES \
	--batch_size $BATCH_SIZE \
	#--oracle_phase
