#!/bin/bash

EVAL_AUDIO_PATH=/user/es.aau.dk/xu68nv/data/GRID/syn_data/full_av_dataset/test-set
ENHANCED_FILE=../target
OUT_FILE=target_masked
PESQ_PATH=/user/es.aau.dk/xu68nv/code/PESQv2/pesq
PESQ_MODE=nb
FFT_SIZE=512
WINDOW_SIZE=24
STEP_SIZE=12

cd /user/es.aau.dk/xu68nv/code/AudioVisualSpeechInpainting

time python -u speech_inpainting_main.py evaluation \
	--eval_audio_dir $EVAL_AUDIO_PATH \
	--enhanced_file $ENHANCED_FILE \
	--out_file $OUT_FILE \
	--pesq_path $PESQ_PATH \
	--pesq_mode $PESQ_MODE \
	--fft_size $FFT_SIZE \
	--window_size $WINDOW_SIZE \
	--step_size $STEP_SIZE \
	--masked_eval 
