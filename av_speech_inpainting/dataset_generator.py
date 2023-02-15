import os
from glob import glob 
import random
import shutil
import numpy as np
import array
from scipy.io import wavfile
from pydub import AudioSegment


def get_intrusions_mask(frame_dim, spec_len, cov_mean, cov_std, n_max_intr, min_intr_len=3):
    # number of intrusions selection
    n_intr = random.randint(1, n_max_intr)

    # mask coverage selection
    mask_cov = max(min_intr_len * n_intr / spec_len, min(random.gauss(cov_mean, cov_std), 0.8))
    mask_bins = int(np.around(spec_len * mask_cov))
    true_mask_cov = mask_bins / spec_len # it is slightly different from sampled value due to rounding
    
    # distribution of mask to intrusions
    intr_lens = []
    for i in range(0, n_intr):
        if i == n_intr - 1:
            intr_lens.append(mask_bins - sum(intr_lens))
        elif i == 0:
            intr_lens.append(random.randint(min_intr_len, max(min_intr_len, int((mask_bins - min_intr_len * (n_intr - 1)) * np.exp(-(n_intr-1) / 6)))))
        else:
            intr_lens.append(random.randint(min_intr_len, max(min_intr_len, int((mask_bins - sum(intr_lens) - min_intr_len * (n_intr - i - 1)) * np.exp(-(n_intr-1) / 6)))))
    random.shuffle(intr_lens)

    # intrusions onset selection
    onset_pos = []
    for i, l in enumerate(intr_lens):
        if i == 0 and i == n_intr - 1:
            onset_pos.append(random.randint(0, spec_len - mask_bins))
        elif i == 0:
            onset_pos.append(random.randint(0, (spec_len - mask_bins - (n_intr - 1))) // 2)
        elif i == n_intr - 1:
            onset_pos.append(random.randint(onset_pos[-1], onset_pos[-1] + intr_lens[i-1] + 1 + spec_len - intr_lens[i]))
        else:
            onset_pos.append(random.randint(onset_pos[-1] + intr_lens[i-1] + 1, (onset_pos[-1] + intr_lens[i-1] + 1 + spec_len - sum(intr_lens[i:]) - (n_intr - i - 1)) // 2))

    # create mask
    mask = np.ones([spec_len, frame_dim])
    for os, il in zip(onset_pos, intr_lens):
        mask[os : os+il] = 0

    return mask, true_mask_cov, n_intr


def create_syn_data_speaker(dataset_dir, dest_dir, n_speaker, n_samples=0, audio_len=3000, n_max_intr=1,
                                     cov_mean=1000, cov_std=300, file_ext='wav'):
    """
    This function generate a corrupted audio sample for each sample in clean_audio_dir.
    """

    # Get the list of speech samples
    clean_audio_dir = os.path.join(dataset_dir, 's' + str(n_speaker), 's' + str(n_speaker) + '_16kHz')
    clean_speech_list = glob(os.path.join(clean_audio_dir, '*.' + file_ext))

    # Landmark dir and normalization files
    landmarks_dir = os.path.join(dataset_dir, 's' + str(n_speaker), 's' + str(n_speaker) + '.landmarks')
    transcriptions_dir = os.path.join(dataset_dir, 's' + str(n_speaker), 'align')
    video_mean_file = os.path.join(landmarks_dir, 'video_feat_mean.npy')
    video_std_file = os.path.join(landmarks_dir, 'video_feat_std.npy')
    
    # if num_samples is > 0 we choice num_samples random samples from clean_speech_list
    if n_samples > 0:
        random.seed(30)
        random.shuffle(clean_speech_list)
        clean_speech_list = clean_speech_list[: n_samples]
    
    spec_len = audio_len // 12 # We suppose that step size is always 12 ms at 16 kHz (192 samples)
    frame_dim = 257
    samples_audio_len = int(audio_len * 16)
    cov_mean_ratio = cov_mean / audio_len
    cov_std_ratio = cov_std / audio_len
    mask_cov_list = []
    # now, for each speech we have to create n_corr_per_sample combinations:
    for n, clean_speech_file in enumerate(clean_speech_list):
        print('{:d} - {:s}'.format(n, clean_speech_file))
        # get random mask
        mask, mask_cov, n_intr = get_intrusions_mask(frame_dim, spec_len, cov_mean_ratio, cov_std_ratio, n_max_intr)
        mask_cov_list.append(mask_cov)

        # save mask and symbolic links to clean speech and spectrogram
        example_name = 's' + str(n_speaker) + '_' + os.path.splitext(os.path.basename(clean_speech_file))[0] + '_' + \
                '{:d}'.format(int(mask_cov * audio_len)) + '_' + str(n_intr)
        dest_example_dir = os.path.join(dest_dir, example_name)
        os.makedirs(dest_example_dir, exist_ok=True)
        
        # Save/copy files
        #os.symlink(clean_speech_file, os.path.join(dest_example_dir, 'target.wav'))
        #landmarks_file = os.path.join(landmarks_dir, os.path.basename(clean_speech_file).replace('.' + file_ext, '.npy'))
        #os.symlink(landmarks_file, os.path.join(dest_example_dir, 'landmarks.npy'))
        #transcription_file = os.path.join(transcriptions_dir, os.path.basename(clean_speech_file).replace('.' + file_ext, '.lbl'))
        #os.symlink(transcription_file, os.path.join(dest_example_dir, 'transcription.lbl'))
        #os.symlink(video_mean_file, os.path.join(dest_example_dir, 'video_feat_mean.npy'))
        #os.symlink(video_std_file, os.path.join(dest_example_dir, 'video_feat_std.npy'))
        shutil.copy(clean_speech_file, os.path.join(dest_example_dir, 'target.wav'))
        landmarks_file = os.path.join(landmarks_dir, os.path.basename(clean_speech_file).replace('.' + file_ext, '.npy'))
        shutil.copy(landmarks_file, os.path.join(dest_example_dir, 'landmarks.npy'))
        transcription_file = os.path.join(transcriptions_dir, os.path.basename(clean_speech_file).replace('.' + file_ext, '.lbl'))
        shutil.copy(transcription_file, os.path.join(dest_example_dir, 'transcription.lbl'))
        shutil.copy(video_mean_file, os.path.join(dest_example_dir, 'video_feat_mean.npy'))
        shutil.copy(video_std_file, os.path.join(dest_example_dir, 'video_feat_std.npy'))
        np.save(os.path.join(dest_example_dir, 'mask.npy'), mask)

    return mask_cov_list


def create_syn_dataset(dataset_dir, dest_dir, speakers=[], n_samples=0, audio_len=3000, n_max_intr=1,
                       cov_mean=1000, cov_std=300, file_ext='wav'):
    # Create destination directory if not exists
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    mask_cov_list = []
    print('Starting dataset generation...')
    for s in speakers:
        print('Creating masks of speaker {:d}...'.format(s))
        mask_cov_list_spk = create_syn_data_speaker(dataset_dir, dest_dir, s, n_samples, audio_len,
                                                    n_max_intr, cov_mean, cov_std, file_ext)
        mask_cov_list += mask_cov_list_spk
        print('done.'.format(s))


    print('Dataset generation completed.')
    print('Number of generated samples: {:d}. Total length: {:.2f} seconds'.format(len(mask_cov_list), len(mask_cov_list) * audio_len / 1000))
    print('True mask coverage mean: {:.2f} ms - std: {:.2f} ms'.format(np.mean(mask_cov_list) * audio_len, np.std(mask_cov_list) * audio_len))


if __name__ == '__main__':
    clean_audio_dir = 'C:\\Users\\Public\\aau_data\\GRID'
    dest_dir = 'C:\\Users\\Public\\aau_data\\GRID\\test_si_dataset\\test-set-lbl'
    speaker_list = [11]
    audio_len = 3000
    n_max_intr = 1
    cov_mean = 100
    cov_std = 0
    ext = 'wav'

    create_syn_dataset(clean_audio_dir, dest_dir, speaker_list, 0, audio_len, n_max_intr, cov_mean, cov_std, ext)