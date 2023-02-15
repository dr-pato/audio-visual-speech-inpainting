#!/bin/bash

AUDIO_DIR=/user/es.aau.dk/xu68nv/data/GRID/syn_data/full_av_dataset/training-set
TYPE=fbanks
SAMPLE_RATE=16000
FFT_SIZE=512
WINDOW_SIZE=24
STEP_SIZE=12
EXT=wav
FILE_PREFIX=target
OUT_PREFIX=fbanks_norm_full
EXT=wav

cd /user/es.aau.dk/xu68nv/code/AudioVisualSpeechInpainting

time python -u speech_inpainting_main.py audio_preprocessing \
	--audio_dir $AUDIO_DIR \
	--file_prefix $FILE_PREFIX \
	--out_prefix $OUT_PREFIX \
	--type $TYPE \
	--sample_rate $SAMPLE_RATE \
	--fft_size $FFT_SIZE \
	--window_size $WINDOW_SIZE \
	--step_size $STEP_SIZE \
	--ext $EXT \
	#--apply_mask
