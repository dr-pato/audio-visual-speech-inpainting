import sys
import os
from glob import glob
import numpy as np


def load_dictionary(filename):
    with open(filename,'r') as f:
        dictionary = f.read()

    phonemes = dictionary.replace('\n',' ').split(' ')
    phonemes = [ph for ph in sorted(set(phonemes)) if ph is not '']
    
    return phonemes


def get_labels(phonemes, dictionary):
    labels = phonemes.replace('SP', '').split(',')
    labels = [lab for lab in labels if lab is not '']
    labels = [dictionary.index(ph) for ph in labels]
    
    return np.asarray(labels)


def get_phonemes_from_labels(labels, dictionary):
    return list(map(lambda x: dictionary[int(x)], labels))


def get_phonemes(transcription, word_list, dict_list):
    for word, phonemes in zip(word_list, dict_list):
        transcription = transcription.replace(word, phonemes)

    return transcription


def linearize(transcription):
    transcription = transcription.replace('\n', ' ').split(' ')
    lin_transcription = []

    for ph in transcription:
        if ph.isalpha() and ph != 'SIL':
            lin_transcription.append(ph)
            lin_transcription.append(',')
    lin_transcription = ''.join(lin_transcription[:-1]) # remove last comma

    return lin_transcription


def save_phonemes_labels(data_path, word_list, dict_list):
    file_list = glob(os.path.join(data_path, '**', '*.align'), recursive=True)
    for trascription_file in file_list:
        phonemes_file = trascription_file.replace('.align', '.phalign')
        phonemes_lin_file = trascription_file.replace('.align', '.lbl')

        with open(trascription_file) as f:
            transcription = f.read()
        
        # Save aligned phonemes
        phonemes = get_phonemes(transcription, word_list, dict_list)
        with open(phonemes_file, 'w') as f:
            f.write(phonemes)

        # Save linearized phonemes (the labels)
        lin_phonemes = linearize(phonemes)
        with open(phonemes_lin_file, 'w') as f:
            f.write(lin_phonemes)


if __name__ == '__main__':
    data_path = sys.argv[1]
    word_list = open(os.path.join(sys.argv[2]), 'r').read().split('\n')
    dict_list = open(os.path.join(sys.argv[3]), 'r').read().split('\n')

    #input files are the .align file from GRID/align folders, the output will be the files that contain the sequences of correspondent phonemes
    #data_path = 'C:\\Users\\Public\\aau_data\\GRID'
    #word_list = open(os.path.join(data_path, 'word.txt'), 'r').read().split('\n')
    #dict_list = open(os.path.join(data_path, 'dictionary.txt'), 'r').read().split('\n')

    save_phonemes_labels(data_path, word_list, dict_list)


