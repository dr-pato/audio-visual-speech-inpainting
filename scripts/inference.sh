#!/bin/bash

MODEL_PATH=/user/es.aau.dk/xu68nv/data/GRID/logs/full_av_dataset/av-blstm_exp1/netmodel
DATA_PATH=/user/es.aau.dk/xu68nv/data/GRID/tfrecords/full_av_dataset/training-set
AUDIO_PATH=/user/es.aau.dk/xu68nv/data/GRID/syn_data/full_av_dataset/training-set
OUT_FILE_PREFIX=av-blstm_exp1
BATCH_SIZE=32

cd /user/es.aau.dk/xu68nv/code/AudioVisualSpeechInpainting

time python -u speech_inpainting_main.py inference \
	--model_path $MODEL_PATH \
	--data_dir $DATA_PATH \
	--audio_dir $AUDIO_PATH \
	--out_file_prefix $OUT_FILE_PREFIX \
	--norm \
	--batch_size $BATCH_SIZE \
#	--oracle_phase
