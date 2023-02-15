#!/bin/bash

CONFIG_FILE=/user/es.aau.dk/xu68nv/scripts/av-speech-inpainting/configurations/blstm_ctc.config

cd /user/es.aau.dk/xu68nv/code/AudioVisualSpeechInpainting

time python -u speech_inpainting_main.py training --config $CONFIG_FILE
