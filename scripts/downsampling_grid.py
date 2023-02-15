import os
from glob import glob
import numpy as np
from scipy import signal
from scipy.io import wavfile


def downsampling(samples, sample_rate, downsample_rate):
	secs = len(samples) / float(sample_rate)
	num_samples = int(downsample_rate * secs)
	
	if sample_rate != downsample_rate:
		return signal.resample(samples, num_samples)
	else:
		return samples


if __name__ == '__main__':
	for s in range(1, 35):
		files = glob(os.path.join('/user/es.aau.dk/xu68nv/data/GRID', 's' + str(s), 's' + str(s) + '_50kHz', '*.wav'))
		dest_dir = os.path.join('/user/es.aau.dk/xu68nv/data/GRID', 's' + str(s), 's' + str(s) + '_16kHz')
		if not os.path.isdir(dest_dir):
			os.makedirs(dest_dir)
		
		print('Processing speaker:', s)
		for f in files:
			sr, wav = wavfile.read(f)
			wav = downsampling(wav, sr, 16000)
			out_file = os.path.join('/user/es.aau.dk/xu68nv/data/GRID', 's' + str(s), 's' + str(s) + '_16kHz', os.path.basename(f))
			wavfile.write(out_file, 16000, wav.astype(np.int16))
		print('done.')
