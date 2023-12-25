import json

# Transformer model params
VOCAB_SIZE = 50304
CONTEXT_LENGTH = 128
N_EMBED = 128
N_HEAD = 8

# Autoencoder model params
N_FEATURES = 512

# Transformer training params
TRAIN_PATH = "data/med_pile_train.h5"
DEV_PATH = "data/pile_val.h5"
T_BATCH_SIZE = 32
T_CONTEXT_LENGTH = 128
T_TRAIN_STEPS = 200000
T_EVAL_STEPS = 1000
T_EVAL_ITERS = 250
T_LR_DECAY_STEP = 50000
T_LR = 5e-4
T_LR_DECAYED = 5e-5
T_OUT_PATH = "models/transformer_full.pt"

# Autoencoder training params
A_BATCH_SIZE = 512
A_TRAIN_STEPS = 100
A_LR = 1e-3
A_OUT_PATH = "models/autoencoder.pt"
LAMBDA_REG = 3e-3


DEVICE = 'mps'

default_config = {
    'vocab_size': VOCAB_SIZE,
    'context_length': CONTEXT_LENGTH,
    'n_embed': N_EMBED,
    'n_head': N_HEAD,
    'n_features': N_FEATURES,
    'train_path': TRAIN_PATH,
    'dev_path': DEV_PATH,
    't_batch_size': T_BATCH_SIZE,
    't_context_length': T_CONTEXT_LENGTH,
    't_train_steps': T_TRAIN_STEPS,
    't_eval_steps': T_EVAL_STEPS,
    't_eval_iters': T_EVAL_ITERS,
    't_lr_decay_step': T_LR_DECAY_STEP,
    't_lr': T_LR,
    't_lr_decayed': T_LR_DECAYED,
    't_out_path': T_OUT_PATH,
    'a_batch_size': A_BATCH_SIZE,
    'a_train_steps': A_TRAIN_STEPS,
    'a_lr': A_LR,
    'a_out_path': A_OUT_PATH,
    'lambda_reg': LAMBDA_REG,
    'device': DEVICE,
}

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config