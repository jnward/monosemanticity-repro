from autoencoder import SparseAutoencoder
from transformer import Transformer
from data_utils import get_batch_iterator
import torch
import numpy as np
import os
from tqdm import tqdm


def train_autoencoder(config):
    # def get_embedding_iterator(transformer, data_path, batch_size, context_length, device="cpu"):
    #     # This can be sped up a lot by sampling embeddings from the transformer first and saving to disk,
    #     # but requires a large amount of disk space
    #     batch_iterator = get_batch_iterator(data_path, batch_size, context_length, device=device)
    #     while True:
    #         xb, _ = next(batch_iterator)
    #         with torch.no_grad():
    #             x_embedding, _ = transformer.forward_embedding(xb)
    #         random_idxs = torch.randint(context_length, (batch_size))
    #         filtered_x_embedding = x_embedding[range(batch_size), random_idxs, :]  # only take one sample per batch
    #         yield filtered_x_embedding
    autoencoder = SparseAutoencoder(
        config['n_features'],
        config['n_embed'],
    ).to(config['device'])
    transformer = Transformer(
        n_head=config['n_head'],
        n_embed=config['n_embed'],
        context_length=config['context_length'],
        vocab_size=config['vocab_size']
    ).to(config['device'])
    transformer_checkpoint = torch.load(config['t_out_path'])
    transformer.load_state_dict(transformer_checkpoint['model_state_dict'])
    
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=config['a_lr'])
    batch_iterator = get_batch_iterator(
        config['train_path'],
        config['a_batch_size'],
        config['t_context_length'],
        device=config['device']
    )
    losses = []
    recon_losses = []
    reg_losses = []

    pbar = tqdm(range(config['a_train_steps']))
    for _ in pbar:
        xb, _ = next(batch_iterator)
        with torch.no_grad():
            x_embedding, _ = transformer.forward_embedding(xb)
        random_idxs = torch.randint(config['t_context_length'], (config['a_batch_size'],))
        filtered_x_embedding = x_embedding[range(config['a_batch_size']), random_idxs, :]

        optimizer.zero_grad()
        _, recon_loss, reg_loss = autoencoder(filtered_x_embedding, compute_loss=True)
        reg_loss = reg_loss * config['lambda_reg']
        loss = recon_loss + reg_loss
        loss.backward()
        optimizer.step()
        autoencoder.normalize_decoder_weights()

        losses.append(loss.item())
        recon_losses.append(recon_loss.item())
        reg_losses.append(reg_loss.item())

        pbar.set_description(f"Recon Loss: {np.mean(recon_losses[-100:]):.3f} Reg Loss: {np.mean(reg_losses[-100:]):.3f}")
        modified_model_out_path = config['a_out_path']
    save_tries = 0
    while os.path.exists(modified_model_out_path):
        save_tries += 1
        model_out_name = os.path.splitext(config['a_out_path'])[0]
        modified_model_out_path = model_out_name + f"_{save_tries}" + ".pt"
    torch.save(
        {
            'model_state_dict': autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            'steps': len(losses),
        }, modified_model_out_path)
    print(f"Saved model to {modified_model_out_path}")
    print(f"Finished training. Recon loss: {np.mean(recon_losses[-100:]):.3f}, Reg loss: {np.mean(reg_losses[-100:]):.3f}")


if __name__ == '__main__':
    import sys
    from config import default_config, load_config
    if len(sys.argv) > 1:
        print("Using config file")
        config = load_config(sys.argv[1])
    else:
        print("Using default config")
        config = default_config
    train_autoencoder(config)