#!/bin/bash

EVAL_AUDIO_PATH=/user/es.aau.dk/xu68nv/data/LibriSpeech/syn_data/full_dataset/validation-set-10
ENHANCED_FILE=../masked
OUT_FILE=masked
PESQ_PATH=/user/es.aau.dk/xu68nv/code/PESQv2/pesq
PESQ_MODE=nb
FFT_SIZE=256
WINDOW_SIZE=16
STEP_SIZE=8

cd /user/es.aau.dk/xu68nv/code/SpeechInpaintingAudioBaseline

time python -u evaluation_mateng.py \
	--eval_audio_dir $EVAL_AUDIO_PATH \
	--enhanced_file $ENHANCED_FILE \
	--out_file $OUT_FILE \
	--pesq_path $PESQ_PATH \
	--pesq_mode $PESQ_MODE \
	--fft_size $FFT_SIZE \
	--window_size $WINDOW_SIZE \
	--step_size $STEP_SIZE \
#	--masked_eval 
