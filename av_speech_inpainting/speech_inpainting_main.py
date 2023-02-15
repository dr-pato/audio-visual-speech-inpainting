import sys
import argparse

from dataset_generator import create_syn_dataset
from audio_feat_preprocessing import compute_mean_std_features
from face_landmarks import save_face_landmarks
from tfrecord_emb_utils import create_dataset, group_tfrecords
from masking import mask_app
from training_ctc import train
from training_asr import train as train_asr
from inference_model_generator import save_inference_model
from inference import infer
from inference_asr import infer as infer_asr
from inference_siasr_ctc import infer as infer_siasr
from evaluation import speech_inpainting_eval


def parse_args():
    parser = argparse.ArgumentParser(description="""Audio-visual speech inpainting system.
    Try 'python speech_inpainting_main.py <subcommand> --help' for more information about subcommands.
    A full list of subcommands is shown below (positional arguments).""")
    subparsers = parser.add_subparsers(dest='subparser_name')

    # Dataset generator
    dataset_gen_parser = subparsers.add_parser('dataset_generator', description="""Generate masks dataset. Files are saved in <dest_dir>.""")
    dataset_gen_parser.add_argument('-ca', '--clean_audio_dir', required=True,
                                    help='The directory that contains clean wav files to be processed')
    dataset_gen_parser.add_argument('-bs', '--speaker_ids', nargs='+', type=int, required=True,
                                     help='Speaker IDs to be processed')
    dataset_gen_parser.add_argument('-d', '--dest_dir', required=True,
                                    help='The directory where files will be saved')
    dataset_gen_parser.add_argument('-num', '--num_samples', type=int, required=True,
                                    help='Number of randomly chosen clean wavs (0 if you want to take all)')
    dataset_gen_parser.add_argument('-al', '--audio_length', type=int, default=1024,
                                    help='Maximum length (in milliseconds) of audio samples')
    dataset_gen_parser.add_argument('-i', '--num_max_intr', type=int, default=1,
                                    help='Maximum number of intrusions')
    dataset_gen_parser.add_argument('-cm', '--mask_coverage_mean', type=float, default=0.3,
                                    help='Mask coverage mean (from normal distribution) in milliseconds')
    dataset_gen_parser.add_argument('-cs', '--mask_coverage_std', type=float, default=0.1,
                                    help='Mask coverage standard deviation (from normal distribution) in milliseconds')
    dataset_gen_parser.add_argument('-e', '--ext', default='wav', help='The extension of audio files')


    # Compute mean and standard deviation of audio spectrograms
    audio_preprocessing_parser = subparsers.add_parser('audio_preprocessing', description="""Compute mean and standard deviation of audio spectrograms for normalization.
    Power-law compressed spectrograms are computed for all wavs in <audio_dir>/<sample_dir>/<file_prefix>.<ext>.
    If <save_spec> argumement is set, the spectrograms are saved in NPY format in <audio_dir>/<sample_dir>/<file_prefix>.npy.""")
    audio_preprocessing_parser.add_argument('-a', '--audio_dir', required=True,
                                            help='The subdirectory that contains audio samples')
    audio_preprocessing_parser.add_argument('-p', '--file_prefix', required=True, help='File prefix of audio files to be processed')
    audio_preprocessing_parser.add_argument('-o', '--out_prefix', required=True, help='File prefix of output files')
    audio_preprocessing_parser.add_argument('-t', '--type', default='spec', choices=['spec', 'fbanks', 'mfcc'],
                                            help='Features selection (default: spec')
    audio_preprocessing_parser.add_argument('-sr', '--sample_rate', type=int, default=16000,
                                            help='Desired sample rate (in Hz) (default: 16000 Hz)')
    audio_preprocessing_parser.add_argument('-fs', '--fft_size', type=int, default=512,
                                            help='FFT size used when computing STFT (default: 512)')
    audio_preprocessing_parser.add_argument('-ws', '--window_size', type=int, default=25,
                                            help='Window size (in milliseconds) when computing STFT (default: 25 ms)')
    audio_preprocessing_parser.add_argument('-ss', '--step_size', type=int, default=10,
                                            help='Step size (in milliseconds) when computing STFT (default: 10 ms)')
    audio_preprocessing_parser.add_argument('-pe', '--preemph', type=float, default=0,
                                            help='Coefficient of pre-emphasis filter (default: 0 - no pre-emphasis)')
    audio_preprocessing_parser.add_argument('-nm', '--num_mel_bins', type=int, default=80,
                                            help='Number of mel filters (default: 80)')
    audio_preprocessing_parser.add_argument('-nmf', '--num_mfcc', type=int, default=13,
                                            help='Number of MFCCs (default: 13)')
    audio_preprocessing_parser.add_argument('-d', '--delta', type=int, default=0,
                                            help='Number of derivative orders to be added to features (default: 0 - static features only)')
    audio_preprocessing_parser.add_argument('-am', '--apply_mask', action='store_const', const=True, default=False,
                                            help='If it is set, masking is applied before computing statistics.')
    audio_preprocessing_parser.add_argument('-s', '--save_feat', action='store_const', const=True, default=False,
                                            help='If it is set, features are saved in NPY format')
    audio_preprocessing_parser.add_argument('-e', '--ext', default='wav', help='The extension of audio files')


    # Compute face landmarks
    video_preprocessing_parser = subparsers.add_parser('video_preprocessing', description="""Compute face landmarks.
    For each <speaker_id> in <speaker_ids> face landmarks are computed for all videos in
    <data_dir>/s<speaker_id>/<video_dir>. The raw face landmarks are saved in NPY format in <data_dir>/s<speaker_id>/<video_dir>.
    Motion vectors and normalization data are saved in <data_dir>/s<speaker_id>/<dest_dir>""")
    video_preprocessing_parser.add_argument('-data', '--data_dir', required=True, help='The base pathname of dataset')
    video_preprocessing_parser.add_argument('-s', '--speaker_ids', nargs='+', type=int, required=True,
                                            help='Speaker IDs of videos to be processed')
    video_preprocessing_parser.add_argument('-v', '--video_dir', required=True,
                                            help='The subdirectory that contains video files to be processed')
    video_preprocessing_parser.add_argument('-d', '--dest_dir', required=True,
                                            help='The subdirectory where output files are saved')
    video_preprocessing_parser.add_argument('-sp', '--shape_predictor', required=True,
                                            help='Path of the file that contains the parameters of Dlib face landmark extractor')
    video_preprocessing_parser.add_argument('-e', '--ext', required=True, default='mpg',
                                            help='The extension of video files')
    

    # Create TFRecords of training data
    tfrecords_parser = subparsers.add_parser('tfrecords_generator', description="""Create TFRecords of training dataset.
    <audio_dir> have to contain three directories named 'training-set', 'validation-set' and 'test-set'.
    These directories cointains several directories, one for each audio sample, generated with 'dataset_generator' subcommand.
    A directory of a single audio sample cointains target audio and mask.
    """)
    tfrecords_parser.add_argument('-m', '--mode', default='fixed', choices=['fixed', 'var'],
                                  help='"fixed" (default) if wavs have all the same length, "var" otherwise')
    tfrecords_parser.add_argument('-a', '--dataset_dir', required=True,
                                  help='The subdirectory that contains audio wavs')
    tfrecords_parser.add_argument('-d', '--dest_dir', required=True,
                                  help='The subdirectory where TFRecords will be saved')
    tfrecords_parser.add_argument('-df', '--dict_file', required=True,
                                  help='The dictionary file of ASR labels')

    # Group TFRecords with similar lenghts
    group_tfrecords_parser = subparsers.add_parser('tfrecords_grouping', description="""Group TFRecords with similar lengths.
    Each TFRecord in <input_dir> have to contain only a single utterance/sample.""")
    group_tfrecords_parser.add_argument('-i', '--input_dir', required=True, help='The subdirectory that contains input TFRecords')
    group_tfrecords_parser.add_argument('-o', '--output_dir', required=True, help='The subdirectory that will contain grouped TFRecords')
    group_tfrecords_parser.add_argument('-gs', '--group_size', type=int, default=16, help='Number of samples in a single TFRecord')
    group_tfrecords_parser.add_argument('-d', '--del_input_dir', action='store_const', const=True, default=False,
                                        help='If it is set input directory will be deleted')

    # Oracle mask evaluation
    masking_parser = subparsers.add_parser('masking', description='Generate masked wavs.')
    masking_parser.add_argument('-d', '--data_dir', required=True, help='Subdirectory with TFRecords of dataset')
    masking_parser.add_argument('-ad', '--audio_dir', required=True, help='Subdirectory with audio samples of dataset')
    masking_parser.add_argument('-m', '--mode', default='fixed', choices=['fixed', 'var'],
                                  help='"fixed" (default) if wavs have all the same length, "var" otherwise')
    masking_parser.add_argument('-af', '--audio_feat_dim', type=int, default=257, help='Dimension of audio features (default: 257)')
    masking_parser.add_argument('-vf', '--video_feat_dim', type=int, default=136, help='Dimension of video features (default: 136)')
    masking_parser.add_argument('-ns', '--num_audio_samples', type=int, default=48000, help='Number of samples in waveform (default: 0 - unknown)')
    masking_parser.add_argument('-op', '--oracle_phase', action='store_const', const=True, default=False,
                                help='If it is set oracle phase is used for inverse STFT')
    masking_parser.add_argument('-bs', '--batch_size', type=int, default=0, help='Batch size')
    
    # Train a speech inpainting model
    training_parser = subparsers.add_parser('training', description='Train a speech inpainting model.')
    training_parser.add_argument('--config', required=True, type=str, help='Configuration file')

    # Train ASR model
    training_parser_asr = subparsers.add_parser('training_asr', description='Train an ASR model.')
    training_parser_asr.add_argument('--config', required=True, type=str, help='Configuration file')

    # Save model for inference (removing batch size information from graph)
    infer_model_gen_parser = subparsers.add_parser('inference_model_generation', description='Save inference model.')
    infer_model_gen_parser.add_argument('--config', required=True, type=str, default="",
                                        help='Configuration file')
    infer_model_gen_parser.add_argument('--model', type=str, choices=['enh', 'asr', 'enhasr'], default='enh',
                                        help='Model selection')
    infer_model_gen_parser.add_argument('--input_model', required=True, type=str,
                                        help='Checkpoint of training model')
    infer_model_gen_parser.add_argument('--output_model', required=True, type=str,
                                        help='Checkpoint where inference model will be saved')

    # Speech inpainting model inference
    inference_parser = subparsers.add_parser('inference', description='Inference with trained speech inpainting model.')
    inference_parser.add_argument('-d', '--data_dir', required=True, help='Subdirectory with TFRecords of dataset')
    inference_parser.add_argument('-ad', '--audio_dir', required=True, help='Subdirectory with audio samples of dataset')
    inference_parser.add_argument('-ef', '--out_file_prefix', required=True,
                                  help='Enhanced audio filename to be saved. Pathname: <eval_audio_dir>/<sample_dir>/enhanced/<out_file_prefix>.wav')
    inference_parser.add_argument('-m', '--model_path', required=True,
                                  help='Model checkpoint to be restored.')
    inference_parser.add_argument('-n', '--norm', action='store_const', const=True, default=False,
                                  help='If it is set, standard normalization is applied to input features')
    inference_parser.add_argument('-bs', '--batch_size', type=int, default=0, help='Batch size')
    inference_parser.add_argument('-op', '--oracle_phase', action='store_const', const=True, default=False,
                                  help='If it is set oracle phase is used for inverse STFT')

    # Speech inpainting model inference
    inference_asr_parser = subparsers.add_parser('inference_asr', description='Inference with trained ASR model.')
    inference_asr_parser.add_argument('-d', '--data_dir', required=True, help='Subdirectory with TFRecords of dataset')
    inference_asr_parser.add_argument('-ad', '--audio_dir', required=True, help='Subdirectory with audio samples of dataset')
    inference_asr_parser.add_argument('-ef', '--out_file_prefix', required=True,
                                      help='Enhanced audio filename to be saved. Pathname: <eval_audio_dir>/<sample_dir>/trancriptions/<out_file_prefix>.wav')
    inference_asr_parser.add_argument('-m', '--model_path', required=True,
                                      help='Model checkpoint to be restored.')
    inference_asr_parser.add_argument('-am', '--apply_mask', action='store_const', const=True, default=False,
                                      help='If it is set, masking is applied before inference.')
    inference_asr_parser.add_argument('-n', '--norm', action='store_const', const=True, default=False,
                                      help='If it is set, stndard normalization is applied to input features')
    inference_asr_parser.add_argument('-bs', '--batch_size', type=int, default=0, help='Batch size')
    inference_asr_parser.add_argument('-df', '--dict_file', required=True,
                                      help='The dictionary file of ASR labels')

    # Speech inpainting and ASR model inference
    inference_siasr_parser = subparsers.add_parser('inference_siasr', description='Inpainting and ASR inference with trained models.')
    inference_siasr_parser.add_argument('-d', '--data_dir', required=True, help='Subdirectory with TFRecords of dataset')
    inference_siasr_parser.add_argument('-ad', '--audio_dir', required=True, help='Subdirectory with audio samples of dataset')
    inference_siasr_parser.add_argument('-ef', '--out_file_prefix', required=True,
                                        help='Enhanced audio filename to be saved. Pathname: <eval_audio_dir>/<sample_dir>/trancriptions/<out_file_prefix>.wav')
    inference_siasr_parser.add_argument('-ms', '--model_path_si', required=True,
                                        help='Speech Inpaiting model checkpoint to be restored.')
    inference_siasr_parser.add_argument('-mr', '--model_path_asr', required=True,
                                        help='ASR model checkpoint to be restored.')
    inference_siasr_parser.add_argument('-n', '--norm', action='store_const', const=True, default=False,
                                        help='If it is set, stndard normalization is applied to input features')
    inference_siasr_parser.add_argument('-bs', '--batch_size', type=int, default=0, help='Batch size')
    inference_siasr_parser.add_argument('-df', '--dict_file', required=True,
                                        help='The dictionary file of ASR labels')
    inference_siasr_parser.add_argument('-op', '--oracle_phase', action='store_const', const=True, default=False,
                                        help='If it is set oracle phase is used for inverse STFT')
    
    # Evaluate speech samples with standard metrics
    evaluation_parser = subparsers.add_parser('evaluation', description='Evaluate audio samples with standard speech enhancement metrics.')
    evaluation_parser.add_argument('-ed', '--eval_audio_dir', required=True, help='Directory with audio samples.')
    evaluation_parser.add_argument('-ef', '--enhanced_file', required=True,
                                   help='Enhanced audio file to be evaluate. Pathname: <eval_audio_dir>/<sample_dir>/enhanced/<enhanced_file>.wav')
    evaluation_parser.add_argument('-o', '--out_file', required=True,
                                   help='File where all results are saved. Pathname: <eval_audio_dir>/<out_file>.csv')
    evaluation_parser.add_argument('-me', '--masked_eval', action='store_const', const=True, default=False,
                                   help='If it set, also masked samples are evaluated.')
    evaluation_parser.add_argument('--pesq_path', required=True, help='Absolute path of PESQ executable.')
    evaluation_parser.add_argument('--pesq_mode', required=True, choices=['nb', 'wb'],
                                   help='PESQ mode selection.')
    evaluation_parser.add_argument('-fs', '--fft_size', type=int, default=512,
                                   help='FFT size used when computing STFT (default: 512)')
    evaluation_parser.add_argument('-ws', '--window_size', type=int, default=25,
                                       help='Window size (in milliseconds) when computing STFT (default: 25 ms)')
    evaluation_parser.add_argument('-ss', '--step_size', type=int, default=10,
                                   help='Step size (in milliseconds) when computing STFT (default: 10 ms)')
    
    return parser.parse_args()


def main():
    args = parse_args()

    if args.subparser_name == 'dataset_generator':
        create_syn_dataset(args.clean_audio_dir, args.dest_dir, args.speaker_ids, args.num_samples, args.audio_length,
                           args.num_max_intr, args.mask_coverage_mean, args.mask_coverage_std, args.ext)
    elif args.subparser_name == 'audio_preprocessing':
        compute_mean_std_features(args.audio_dir, args.file_prefix, args.out_prefix, args.type, args.sample_rate, args.fft_size, args.window_size,
                                  args.step_size, args.preemph, args.num_mel_bins, args.num_mfcc, args.delta, args.apply_mask, args.save_feat, args.ext)
    elif args.subparser_name == 'video_preprocessing':
        save_face_landmarks(args.data_dir, args.speaker_ids, args.video_dir, args.dest_dir, args.shape_predictor, args.ext)
    elif args.subparser_name == 'tfrecords_generator':
        create_dataset(args.dataset_dir, args.dest_dir, args.dict_file, args.mode)
    elif args.subparser_name == 'tfrecords_grouping':
        group_tfrecords(args.input_dir, args.output_dir, args.group_size, args.del_input_dir)
    elif args.subparser_name == 'masking':
        mask_app(args.data_dir, args.audio_dir, args.mode, args.oracle_phase, args.audio_feat_dim,
                 args.video_feat_dim, args.num_audio_samples, args.batch_size)
    elif args.subparser_name == 'training':
        train(args.config)
    elif args.subparser_name == 'training_asr':
        train_asr(args.config)
    elif args.subparser_name == 'inference_model_generation':
        save_inference_model(args.config, args.input_model, args.output_model, args.model)
    elif args.subparser_name == 'inference':
        infer(args.model_path, args.data_dir, args.audio_dir, args.out_file_prefix, args.norm, args.oracle_phase, args.batch_size)
    elif args.subparser_name == 'inference_asr':
        infer_asr(args.model_path, args.data_dir, args.audio_dir, args.out_file_prefix, args.dict_file, args.apply_mask, args.norm, args.batch_size)
    elif args.subparser_name == 'inference_siasr':
        infer_siasr(args.model_path_si, args.model_path_asr, args.data_dir, args.audio_dir, args.out_file_prefix, args.dict_file, args.norm, args.oracle_phase, args.batch_size)
    elif args.subparser_name == 'evaluation':
        speech_inpainting_eval(args.eval_audio_dir, args.enhanced_file, args.out_file, args.masked_eval,
                               args.pesq_path, args.pesq_mode, args.fft_size, args.window_size, args.step_size)
    else:
        print('Bad subcommand name. Closing...')
        sys.exit(1)


if __name__ == '__main__':
    main()