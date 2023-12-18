from tqdm import tqdm
import numpy as np
from data_utils import get_batch_iterator
from transformer import Transformer
import config
import torch
import os

device = config.DEVICE
n_steps = config.T_TRAIN_STEPS
lr = config.T_LR
decayed_lr = config.T_LR_DECAYED
decay_lr_step = config.T_LR_DECAY_STEP
model_out_path = config.MODEL_OUT_PATH
eval_steps = config.T_EVAL_STEPS
eval_iters = config.T_EVAL_ITERS
avg_window = 64

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

@torch.no_grad()
def estimate_loss(model, steps):
    out = {}
    model.eval()
    for split in ['train', 'dev']:
        batch_iterator = get_batch_iterator(split)
        losses = torch.zeros(steps)
        for k in range(steps):
            xb, yb = next(batch_iterator)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train():
    print(device)
    print(model_out_path)
    batch_iterator = get_batch_iterator('train', device=device)

    model = Transformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    pbar = tqdm(range(n_steps))
    for step in pbar:
        xb, yb = next(batch_iterator)
        _, loss = model(xb, yb)
        losses.append(loss.item())
        pbar.set_description(f"Train loss: {np.mean(losses[-avg_window:]):.4f}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % eval_steps == 0:
            train_loss, dev_loss = estimate_loss(model, eval_iters).values()
            print(f"Step: {step}, Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}")
        if step == decay_lr_step:
            print('Decaying learning rate')
            for g in optimizer.param_groups:
                g['lr'] = decayed_lr

    train_loss, dev_loss = estimate_loss(model, 200).values()

    modified_model_out_path = model_out_path
    save_tries = 0
    while os.path.exists(modified_model_out_path):
        save_tries += 1
        model_out_name = os.path.splitext(model_out_path)[0]
        modified_model_out_path = model_out_name + f"_{save_tries}" + ".pt"
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            'train_loss': train_loss,
            'dev_loss': dev_loss,
            'steps': len(losses),
        }, modified_model_out_path)
    print(f"Saved model to {modified_model_out_path}")
    print(f"Finished training. Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}")



if __name__ == '__main__':
    train()