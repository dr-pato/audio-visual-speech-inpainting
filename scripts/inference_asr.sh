#!/bin/bash

for i in 100 200 400 800 1600; do
MODEL_PATH=/user/es.aau.dk/xu68nv/data/GRID/logs/full_av_dataset_asr/a-blstm_exp1/netmodel
DATA_PATH=/user/es.aau.dk/xu68nv/data/GRID/tfrecords/full_av_dataset_asr/test-set-$i
AUDIO_PATH=/user/es.aau.dk/xu68nv/data/GRID/syn_data/full_av_dataset/test-set-$i
OUT_FILE_PREFIX="../masked"
DICT_FILE=/user/es.aau.dk/xu68nv/data/GRID/dictionary.txt
BATCH_SIZE=32

cd /user/es.aau.dk/xu68nv/code/AudioVisualSpeechInpainting

time python -u speech_inpainting_main.py inference_asr \
	--model_path $MODEL_PATH \
	--data_dir $DATA_PATH \
	--audio_dir $AUDIO_PATH \
	--out_file_prefix $OUT_FILE_PREFIX \
	--norm \
	--batch_size $BATCH_SIZE \
	--dict_file $DICT_FILE \
	--apply_mask
done
