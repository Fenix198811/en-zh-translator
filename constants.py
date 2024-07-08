import torch

# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<sos>', '<eos>']

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'zh'

# hyperparameters
HIDDEN_SIZE = 512
N_LAYERS = 6
N_HEADS = 8
FF_SIZE = 1024
DROPOUT_RATE = 0.1
N_EPOCHS = 20
CLIP = 1.0
BATCH_SIZE = 64
LEARNING_RATE = 0.0001

MODEL_PATH = 'en-cn-translator.pt'
MODEL_STATE = 'model_state'
VOCAB = 'vocab'
TOKENIZER = 'tokenizer'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')