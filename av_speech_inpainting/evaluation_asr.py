from __future__ import division

import os
import argparse
from glob import glob
import numpy as np
from scipy.io import wavfile
from eval_metrics import pesq_eval, l1_eval
import csv
from pystoi import stoi as stoi_eval


def speech_enhancement_eval(test_audio_dir, enhanced_file, out_file, masked_eval=True, pesq_path='pesq', pesq_mode='nb', n_fft=512, window_size=25, step_size=10):
    sample_dirs = [d for d in glob(os.path.join(test_audio_dir, '*')) if os.path.isdir(d)]
    print('Test dataset name:', test_audio_dir)
    print('Enhanced file prefix:', enhanced_file)
    print('Number of samples:', len(sample_dirs))

    sample_names = []
    l1_enhanced = []
    l1_masked = []
    pesq_enhanced = []
    pesq_masked = []
    stoi_enhanced = []
    stoi_masked = []
    
    # Compute speech enhancement eval metrics
    for i, ex_dir in enumerate(sample_dirs):
        sample_name = os.path.basename(ex_dir)
        sample_names.append(sample_name)
        print('{:d} - {:s}'.format(i, sample_name))
        os.chdir(ex_dir) # Change directory to avoid errors in PESQ computation
        
        sr, target = wavfile.read('target.wav')
        _, enhanced = wavfile.read(os.path.join('enhanced', enhanced_file + '.wav'))
        wavlen = min([len(target), len(enhanced)])
        target = target[: wavlen]
        enhanced = enhanced[: wavlen]
        
        # L1
        l1_enhanced.append(l1_eval(target, enhanced, sample_rate=sr, n_fft=n_fft, window_size=window_size, step_size=step_size))
        # PESQ
        pesq_enhanced.append(pesq_eval('target.wav', os.path.join('enhanced', enhanced_file + '.wav'), pesq_path, pesq_mode)[0])
        # STOI
        stoi_enhanced.append(stoi_eval(target, enhanced, sr))
        
        if masked_eval:
            _, masked = wavfile.read(os.path.join('masked.wav'))
            
            # L1
            l1_masked.append(l1_eval(target, masked, sample_rate=sr, n_fft=n_fft, window_size=window_size, step_size=step_size))
            # PESQ
            pesq_masked.append(pesq_eval('target.wav', 'masked.wav', pesq_path, pesq_mode)[0])
            # STOI
            stoi_masked.append(stoi_eval(target, masked, sr))
            
            print('[Masked|Enhanced] L1: {:5f}|{:5f} - PESQ: {:.5f}|{:5f} - STOI: {:.5f}|{:5f}'. \
                format(l1_masked[i], l1_enhanced[i], pesq_masked[i], pesq_enhanced[i], stoi_masked[i], stoi_enhanced[i]))
        else:
            print('[Enhanced] L1: {:.5f} - PESQ: {:.5f} - STOI: {:.5f}'.format(l1_enhanced[i], pesq_enhanced[i], stoi_enhanced[i]))
            
    # Print results
    print('')
    print('Test dataset name:', test_audio_dir)
    print('Enhanced file prefix:', enhanced_file)
    print('')
    print('Enhanced L1 (spectrogram): {:.5f} ({:.5f})'.format(np.nanmean(l1_enhanced), np.nanstd(l1_enhanced)))
    print('Enhanced PESQ: {:.5f} ({:.5f})'.format(np.nanmean(pesq_enhanced), np.nanstd(pesq_enhanced)))
    # Replace 1e-5 values in STOI array with NaN
    stoi_enhanced = np.where(np.array(stoi_enhanced) <= 1e-4, np.nan, stoi_enhanced)
    print('Enhanced STOI: {:.5f} ({:.5f})'.format(np.nanmean(stoi_enhanced), np.nanstd(stoi_enhanced)))
    if masked_eval:
        print('')
        print('Masked L1 (spectrogram): {:.5f} ({:.5f})'.format(np.nanmean(l1_masked), np.nanstd(l1_masked)))
        print('Masked PESQ: {:.5f} ({:.5f})'.format(np.nanmean(pesq_masked), np.nanstd(pesq_masked)))
        # Replace 1e-5 values in STOI array with NaN
        stoi_enhanced = np.where(np.array(stoi_enhanced) <= 1e-4, np.nan, stoi_enhanced)
        print('Masked STOI: {:.5f} ({:.5f})'.format(np.nanmean(stoi_masked), np.nanstd(stoi_masked)))
        print('')
        l1_r = np.array(l1_masked) - np.array(l1_enhanced)
        pesq_i = np.array(pesq_enhanced) - np.array(pesq_masked)
        stoi_i = np.array(stoi_enhanced) - np.array(stoi_masked)
        print('L1 (spectrogram) reduction: {:.5f} ({:.5f})'.format(np.nanmean(l1_r), np.nanstd(l1_r)))
        print('PESQ improvement: {:.5f} ({:.5f})'.format(np.nanmean(pesq_i), np.nanstd(pesq_i)))
        print('STOI improvement: {:.5f} ({:.5f})'.format(np.nanmean(stoi_i), np.nanstd(stoi_i)))
    
    # Save results on file
    results_file = os.path.join(test_audio_dir, out_file + '.csv')
    with open(results_file, 'w') as f:
        wr = csv.writer(f, lineterminator='\n')
        if masked_eval:
            wr.writerow(['SAMPLE', 'L1_MASK', 'L1_ENH', 'PESQ_MASK', 'PESQ_ENH', 'STOI_MASK', 'STOI_ENH', 'L1r', 'PESQi', 'STOI_I']) # header
            data = list(zip(sample_names, l1_masked, l1_enhanced, pesq_masked, pesq_enhanced, stoi_masked, stoi_enhanced, l1_r, pesq_i, stoi_i))
            data.sort(key=lambda x: x[0]) # sort rows by filename
            wr.writerows(data) # results
        else:
            wr.writerow(['SAMPLE', 'L1_ENH', 'PESQ_ENH', 'STOI_ENH']) # header
            data = list(zip(sample_names, l1_enhanced, pesq_enhanced, stoi_enhanced))
            data.sort(key=lambda x: x[0]) # sort rows by filename
            wr.writerows(data) # results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Speech inpainting evaluation""")

    parser.add_argument('-ed', '--eval_audio_dir', required=True, help='Directory with audio samples.')
    parser.add_argument('-ef', '--enhanced_file', required=True,
                        help='Enhanced audio file to be evaluate. Pathname: <eval_audio_dir>/<sample_dir>/enhanced/<enhanced_file>.wav')
    parser.add_argument('-o', '--out_file', required=True,
                        help='File where all results are saved. Pathname: <eval_audio_dir>/<out_file>.csv')
    parser.add_argument('-me', '--masked_eval', action='store_const', const=True, default=False,
                        help='If it set, also masked samples are evaluated.')
    parser.add_argument('--pesq_path', required=True, help='Absolute path of PESQ executable.')
    parser.add_argument('--pesq_mode', required=True, choices=['nb', 'wb'],
                        help='PESQ mode selection.')
    parser.add_argument('-fs', '--fft_size', type=int, default=512,
                        help='FFT size used when computing STFT (default: 512)')
    parser.add_argument('-ws', '--window_size', type=int, default=25,
                            help='Window size (in milliseconds) when computing STFT (default: 25 ms)')
    parser.add_argument('-ss', '--step_size', type=int, default=10,
                                   help='Step size (in milliseconds) when computing STFT (default: 10 ms)')

    args = parser.parse_args()

    speech_enhancement_eval(args.eval_audio_dir, args.enhanced_file, args.out_file, args.masked_eval,
                            args.pesq_path, args.pesq_mode, args.fft_size, args.window_size, args.step_size)

    #test_audio_dir = 'C:\\Users\\Public\\aau_data\\LibriSpeech\\test_si_dataset\\training-set-2'
    #enhanced_file = 'basic_blstm_ter'
    #out_file = 'basic_blstm_eval'
    #masked_eval = False
    #pesq_path = 'C:\\Users\\Giovanni Morrone\\Documents\\Visual Studio 2017\\Projects\\PESQ\\Release\\PESQ.exe'
    #pesq_mode = 'nb'
    #n_fft = 256
    #window_size = 16
    #step_size = 8
    #
    #speech_enhancement_eval(test_audio_dir, enhanced_file, out_file, masked_eval, pesq_path, pesq_mode, n_fft, window_size, step_size)