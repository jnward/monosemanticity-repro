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
T_TRAIN_STEPS = 100
T_EVAL_STEPS = 250
T_EVAL_ITERS = 50
T_LR_DECAY_STEP = 500
T_LR = 1e-3
T_LR_DECAYED = 5e-5
MODEL_OUT_PATH = "models/transformer.pt"

# Autoencoder training params
A_BATCH_SIZE = 512


DEVICE = 'mps'