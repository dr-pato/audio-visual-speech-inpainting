#!/bin/bash

MODEL_PATH_SI=/user/es.aau.dk/xu68nv/data/GRID/logs/full_av_dataset/av-blstm-ssnn_exp2/netmodel
MODEL_PATH_ASR=/user/es.aau.dk/xu68nv/data/GRID/logs/full_av_dataset_asr/a-blstm_exp1/netmodel
DATA_PATH=/user/es.aau.dk/xu68nv/data/GRID/tfrecords/full_av_dataset_asr/test-set
AUDIO_PATH=/user/es.aau.dk/xu68nv/data/GRID/syn_data/full_av_dataset/test-set
OUT_FILE_PREFIX=av-blstm-ssnn_exp2
DICT_FILE=/user/es.aau.dk/xu68nv/data/GRID/dictionary.txt
BATCH_SIZE=32

cd /user/es.aau.dk/xu68nv/code/AudioVisualSpeechInpainting

time python -u speech_inpainting_main.py inference_siasr \
	--model_path_si $MODEL_PATH_SI \
	--model_path_asr $MODEL_PATH_ASR \
	--data_dir $DATA_PATH \
	--audio_dir $AUDIO_PATH \
	--out_file_prefix $OUT_FILE_PREFIX \
	--norm \
	--dict_file $DICT_FILE \
	--batch_size $BATCH_SIZE \
#	--oracle_phase
