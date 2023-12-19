from tqdm import tqdm
import numpy as np
from data_utils import get_batch_iterator
from transformer import Transformer
import torch
import os


AVG_WINDOW = 64

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def train_transformer(config):
    batch_iterator = get_batch_iterator(
        config['train_path'],
        config['t_batch_size'],
        config['t_context_length'],
        device=config['device']
    )

    model = Transformer(
        n_head=config['n_head'],
        n_embed=config['n_embed'],
        context_length=config['context_length'],
        vocab_size=config['vocab_size']
    ).to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['t_lr'])
    losses = []

    @torch.no_grad()
    def estimate_loss(steps):
        out = {}
        model.eval()
        for split in ['train', 'dev']:
            data_path = config['train_path'] if split == 'train' else config['dev_path']
            batch_iterator = get_batch_iterator(data_path, config['t_batch_size'], config['t_context_length'], device=config['device'])
            losses = torch.zeros(steps)
            for k in range(steps):
                xb, yb = next(batch_iterator)
                _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    pbar = tqdm(range(config['t_train_steps']))
    for step in pbar:
        xb, yb = next(batch_iterator)
        _, loss = model(xb, yb)
        losses.append(loss.item())
        pbar.set_description(f"Train loss: {np.mean(losses[-AVG_WINDOW:]):.4f}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % config['t_eval_steps'] == 0:
            train_loss, dev_loss = estimate_loss(config['t_eval_iters']).values()
            print(f"Step: {step}, Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}")
        if step == config['t_lr_decay_step']:
            print('Decaying learning rate')
            for g in optimizer.param_groups:
                g['lr'] = config['t_lr_decayed']

    train_loss, dev_loss = estimate_loss(200).values()

    modified_model_out_path = config['t_out_path']
    save_tries = 0
    while os.path.exists(modified_model_out_path):
        save_tries += 1
        model_out_name = os.path.splitext(config['t_out_path'])[0]
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
    import sys
    from config import default_config, load_config
    if len(sys.argv) > 1:
        print("Using config file")
        config = load_config(sys.argv[1])
    else:
        print("Using default config")
        config = default_config
    train_transformer(config)