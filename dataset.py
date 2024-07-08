import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from constants import SPECIAL_SYMBOLS, SRC_LANGUAGE, TGT_LANGUAGE, UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX, HIDDEN_SIZE, N_LAYERS, N_HEADS, FF_SIZE, DROPOUT_RATE, N_EPOCHS, CLIP, LEARNING_RATE, MODEL_PATH, BATCH_SIZE, DEVICE, MODEL_STATE, VOCAB, TOKENIZER
from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.transformer import Transformer
from utils import model_summary
from training import train, evaluate
import warnings
warnings.filterwarnings('ignore')
import random


# for reproducibility
# refer https://pytorch.org/docs/stable/notes/randomness.html
SEED = 256

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# Place-holders
tokenizer = {}
vocab = {}

tokenizer[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
tokenizer[TGT_LANGUAGE] = get_tokenizer('spacy', language='zh_core_web_sm')

train_list = []
valid_list = []

idx = 0
for line in open('data\en-zh.txt', 'r', encoding='utf-8'):
    list = line.strip().split('\t')
    if idx % 10 == 1:
        valid_list.append((list[0], list[1]))
        valid_list.append((list[0], list[1]))
        valid_list.append((list[0], list[1]))
    else:
        train_list.append((list[0], list[1]))
        train_list.append((list[0], list[1]))
        train_list.append((list[0], list[1]))
        train_list.append((list[0], list[1]))
        train_list.append((list[0], list[1]))
    idx += 1

random.shuffle(train_list)
random.shuffle(valid_list)

print(train_list[-1])
print(valid_list[-1])
def yield_tokens(train_list, ln):
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for text in train_list:
        yield tokenizer[ln](text[language_index[ln]])

def get_vocab(train_list, ln):
    # Create torchtext's Vocab object
    vocab = build_vocab_from_iterator(yield_tokens(train_list, ln),
                                                    min_freq=1,
                                                    specials=SPECIAL_SYMBOLS,
                                                    special_first=True)
    # Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
    # If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
    vocab.set_default_index(UNK_IDX)
    return vocab

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab[ln] = get_vocab(train_list, ln)

SRC_VOCAB_SIZE = len(vocab[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab[TGT_LANGUAGE])

encoder = Encoder(SRC_VOCAB_SIZE, HIDDEN_SIZE, N_LAYERS, N_HEADS, FF_SIZE, DROPOUT_RATE, DEVICE)
decoder = Decoder(TGT_VOCAB_SIZE, HIDDEN_SIZE, N_LAYERS, N_HEADS, FF_SIZE, DROPOUT_RATE, DEVICE)

model = Transformer(encoder, decoder, PAD_IDX, PAD_IDX, DEVICE).to(DEVICE)
model_summary(model)
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(tokenizer[ln], #Tokenization
                                               vocab[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor

# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, src_len, tgt_batch = [], [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        src_len.append(len(src_batch[-1]))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, torch.LongTensor(src_len), tgt_batch

train_dataloader = DataLoader(train_list, batch_size=BATCH_SIZE, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_list, batch_size=BATCH_SIZE, collate_fn=collate_fn)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

best_valid_loss = float('inf')

to_save_dict = {
    MODEL_STATE: model.state_dict(),
    VOCAB: vocab,
    TOKENIZER: tokenizer
}

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_dataloader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        to_save_dict[MODEL_STATE] = model.state_dict()
        torch.save(to_save_dict, MODEL_PATH)

    to_save_dict[MODEL_STATE] = model.state_dict()
    torch.save(to_save_dict, "en-cn-translator-no-valid.pt")

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')