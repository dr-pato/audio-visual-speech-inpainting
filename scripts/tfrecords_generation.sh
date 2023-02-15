#!/bin/bash

DATA_DIR=/user/es.aau.dk/xu68nv/data/GRID/syn_data/full_av_dataset
DEST_DIR=/user/es.aau.dk/xu68nv/data/GRID/tfrecords/full_ave_dataset_asr
TFRECORD_MODE=fixed
DICT_FILE=/user/es.aau.dk/xu68nv/data/GRID/dictionary.txt

cd /user/es.aau.dk/xu68nv/code/AudioVisualSpeechInpainting

time python -u speech_inpainting_main.py tfrecords_generator \
	--dataset_dir $DATA_DIR \
	--dest_dir $DEST_DIR \
	--mode $TFRECORD_MODE \
	--dict_file $DICT_FILE
