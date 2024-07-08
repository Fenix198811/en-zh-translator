from constants import PAD_IDX, DEVICE
import torch


def model_summary(model):
    print(model)
    print(f'# of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    print(f'# of non-trainable params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}')

def make_src_mask(src):
    # src = [batch size, src len]
    src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
    # src_mask = [batch size, 1, 1, src len]
    return src_mask

def make_trg_mask(trg):
    # trg = [batch size, trg len]
    trg_pad_mask = (trg != PAD_IDX).unsqueeze(1).unsqueeze(2)
    # trg_pad_mask = [batch size, 1, 1, trg len]
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=DEVICE)).bool()
    # trg_sub_mask = [trg len, trg len]
    trg_mask = trg_pad_mask & trg_sub_mask
    # trg_mask = [batch size, 1, trg len, trg len]
    return trg_mask
