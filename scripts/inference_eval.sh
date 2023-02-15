#!/bin/bash

cd /user/es.aau.dk/xu68nv/code/AudioVisualSpeechInpainting

# INFERENCE
MODEL_PATH=/user/es.aau.dk/xu68nv/data/GRID/logs/full_av_dataset/v-blstm-ssnn_exp0/netmodel
DATA_PATH=/user/es.aau.dk/xu68nv/data/GRID/tfrecords/full_av_dataset_asr/test-set
AUDIO_PATH=/user/es.aau.dk/xu68nv/data/GRID/syn_data/full_av_dataset/test-set
OUT_FILE_PREFIX=v-blstm-ssnn_exp0
BATCH_SIZE=32

time python -u speech_inpainting_main.py inference \
	--model_path $MODEL_PATH \
	--data_dir $DATA_PATH \
	--audio_dir $AUDIO_PATH \
	--out_file_prefix $OUT_FILE_PREFIX \
	--norm \
	--batch_size $BATCH_SIZE \
#	--oracle_phase


# EVALUATION
EVAL_AUDIO_PATH=/user/es.aau.dk/xu68nv/data/GRID/syn_data/full_av_dataset/test-set
OUT_FILE=v-blstm-ssnn_exp0_eval
PESQ_PATH=/user/es.aau.dk/xu68nv/code/PESQv2/pesq
PESQ_MODE=nb
FFT_SIZE=512
WINDOW_SIZE=24
STEP_SIZE=12

time python -u speech_inpainting_main.py evaluation \
	--eval_audio_dir $EVAL_AUDIO_PATH \
	--enhanced_file $OUT_FILE_PREFIX \
	--out_file $OUT_FILE \
	--pesq_path $PESQ_PATH \
	--pesq_mode $PESQ_MODE \
	--fft_size $FFT_SIZE \
	--window_size $WINDOW_SIZE \
	--step_size $STEP_SIZE \
#	--masked_eval
