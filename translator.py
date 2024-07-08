import torch
from inference import translate_sentence
from constants import HIDDEN_SIZE, N_LAYERS, N_HEADS, FF_SIZE, PAD_IDX, DROPOUT_RATE, DEVICE, SRC_LANGUAGE, TGT_LANGUAGE, MODEL_PATH, MODEL_STATE, VOCAB
from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.transformer import Transformer

saved_dict = torch.load(MODEL_PATH)

vocab = saved_dict[VOCAB]
model_state_dict = saved_dict[MODEL_STATE]

SRC_VOCAB_SIZE = len(vocab[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab[TGT_LANGUAGE])

encoder = Encoder(SRC_VOCAB_SIZE, HIDDEN_SIZE, N_LAYERS, N_HEADS, FF_SIZE, DROPOUT_RATE, DEVICE)
decoder = Decoder(TGT_VOCAB_SIZE, HIDDEN_SIZE, N_LAYERS, N_HEADS, FF_SIZE, DROPOUT_RATE, DEVICE)

model = Transformer(encoder, decoder, PAD_IDX, PAD_IDX, DEVICE).to(DEVICE)

model.load_state_dict(model_state_dict)

model.eval()

while True:
    print("Enter the English you want to translate into Chinese:")
    en_sentence = input()
    cn_sentence, attn_scores = translate_sentence(en_sentence, vocab, model)
    print("Your translation result in Chinese:")
    print(cn_sentence)