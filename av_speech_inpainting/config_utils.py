import sys
import os
import re
import ast


def load_configfile(cfile):
    """ This Function loads a configuration file, checks the values of the fields and creates configuration dictionary
           Args:
               config: configuration file
           Returns:
               cD: configuration dictionay
           """
    if not os.path.isfile(cfile):
        raise ValueError("Cannot find configuration file ", cfile)

    cD = {}
    with open(cfile, 'r') as fh:
        nline = 1
        for rawline in fh:
            line = rawline.rstrip()

            if not line or line[0] == '#':
                nline += 1
                continue
            else:
                par = re.search('(\w+)\s*=\s*(.*)', line)
                if par is None:
                    raise ValueError("Wrong syntax in the configuration file at line ", nline)

                if par.group(2).find('[') == -1:
                    if par.group(2).find(' ') > -1:
                        raise ValueError("Wrong syntax in the configuration file at line {:d} "
                                         "(may be a space in the param value?)".format(nline))
                    if re.search('[0-9]', par.group(2)) and par.group(2).find('/') == -1:
                        try:
                            cD[par.group(1)] = ast.literal_eval(par.group(2))
                        except:
                            raise ValueError("Wrong syntax in the configuration file at line {:d} "
                                             "(may be due to mixed letters and integers?)".format(nline))
                    else:
                        cD[par.group(1)] = par.group(2)
                else:
                    try:
                        cD[par.group(1)] = ast.literal_eval(par.group(2))
                    except:
                        raise ValueError("Wrong syntax in the configuration file at line {:d} "
                                         "(may be a missing square parenthesis?)".format(nline))

            nline += 1

    return cD


def check_trainconfiguration(config):
    if 'root_folder' not in config:
        raise ValueError("Root folder not defined")
    if 'exp_folder' not in config:
        raise ValueError("Experiment folder (exp_folder) not defined")
    if 'model_ckp' not in config:
        config['model_ckp'] = ""
    if 'model_ckp_vnet' not in config:
        config['model_ckp_vnet'] = ""
    if 'device' not in config:
        print("WARNING: using cpu as device has not been defined in the config file", file=sys.stderr)
        config['device'] = "/cpu:0"

    if 'model' not in config:
        raise ValueError("Model type (model) not defined in config file")
    if 'net_dim' not in config:
        raise ValueError("Enhancement net dimensions (enh_net_dim) not defined in config file")
    if 'integration_layer' not in config:
        config['integration_layer'] = 0
        print("WARNING: Embedding integration layer not defined in config file. Set to 0 by default", file=sys.stderr)
    if 'audio_feat_dim' not in config:
        config['audio_feat_dim'] = 257
        print("WARNING: No. of audio input features of inpainting model not defined in config file. Set to 257 by default", file=sys.stderr)
    if 'video_feat_dim' not in config:
        config['video_feat_dim'] = 136
        print("WARNING: No. of video input features of inpainting model not defined in config file. Set to 136 by default", file=sys.stderr)
    if 'audio_len' not in config:
        config['audio_len'] = 16384
        print("WARNING: Length of input wavs of inpainting model not defined in config file. Set to 0 by default (variable-length)", file=sys.stderr)
    if 'audio_feat_mean' not in config:
        raise ValueError("File with mean of features (audio_feat_mean) not defined in config file")
    if 'audio_feat_std' not in config:
        raise ValueError("File with standard deviation of features (audio_feat_std) not defined in config file", file=sys.stderr)
    if 'num_asr_labels' not in config:
        config['num_asr_labels'] = 33 # Number of labels for GRID dataset
        print("WARNING: No. of speech recognition labels not defined in config file. Set to 33 by default", file=sys.stderr)
    config['num_asr_labels'] += 1 # Add "blank" label
    if 'audio_len' not in config:
        config['ctc_loss'] = 1
        print("WARNING: CTC loss weigth not defined in config file. Set to 1 by default", file=sys.stderr)
        
    if 'batch_size' not in config:
        print("WARNING: Batch size not defined in config file. Set to 1 by default", file=sys.stderr)
        config['batch_size'] = 1
    if 'dropout_rate' not in config:
        print("WARNING: Dropout rate not defined in config file. Set to 1 by default", file=sys.stderr)
        config['dropout_rate'] = 0.0
    if 'starter_learning_rate' not in config:
        print("WARNING: Starter learning rate not defined in config file. Set to 0.06 by default", file=sys.stderr)
        config['starter_learning_rate'] = 0.06
    if 'learning_rate' not in config:
        print("WARNING: Learning rate not defined in config file. Set to 0.06 by default", file=sys.stderr)
        config['learning_rate'] = 0.06
    if 'lr_updating_steps' not in config:
        print("WARNING: Updating steps of learning rate decay not defined in config file. Set to 10000 by default", file=sys.stderr)
        config['lr_updating_steps'] = 10000
    if 'lr_decay' not in config:
        print("WARNING: Learning rate decay not defined in config file. Set to 0.5 by default", file=sys.stderr)
        config['lr_decay'] = 0.5
    if 'l2' not in config:
        print("WARNING: L2 regularization coefficient not defined in config file. Set to 0 by default", file=sys.stderr)
        config['l2'] = 0.0
    if 'optimizer_type' not in config:
        print("WARNING: Optimizer type not defined in config file. Set to 'adam' by default", file=sys.stderr)
        config['optimizer_type'] = 'adam'
    if config['optimizer_type'] == 'momentum_dlr' and 'momentum' not in config:
        raise ValueError("momentum missing from config file")
    if 'max_n_epochs' not in config:
        print("WARNING: max_n_epochs not defined. Set to 100 by default", file=sys.stderr)
        config['max_n_epochs'] = 30
    if 'n_earlystop_epochs' not in config:
        print("WARNING: n_earlystop_epochs not defined. Set to 3 by default", file=sys.stderr)
        config['n_earlystop_epochs'] = 30
    
    return config