import torch
import spacy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from constants import SRC_LANGUAGE, TGT_LANGUAGE, PAD_IDX, SOS_IDX, EOS_IDX, DEVICE, SPECIAL_SYMBOLS
from utils import make_src_mask, make_trg_mask


def translate_sentence(sentence, vocabs, model,max_len=100):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('en_core_web_sm')
        tokens = [token.text for token in nlp(sentence)]
    else:
        tokens = [token for token in sentence]

    tokens = [SPECIAL_SYMBOLS[SOS_IDX]] + tokens + [SPECIAL_SYMBOLS[EOS_IDX]]

    src_indexes = [vocabs[SRC_LANGUAGE][token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(DEVICE)

    src_mask = make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [SOS_IDX]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(DEVICE)

        trg_mask = make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == EOS_IDX:
            break

    trg_tokens = [vocabs[TGT_LANGUAGE].vocab.get_itos()[i] for i in trg_indexes]

    return trg_tokens[1:-1], attention


def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):
    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + [SPECIAL_SYMBOLS[SOS_IDX]] + [t.lower() for t in sentence] + [SPECIAL_SYMBOLS[EOS_IDX]],
                           rotation=45)
        ax.set_yticklabels([''] + translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()