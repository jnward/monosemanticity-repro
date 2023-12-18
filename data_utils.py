import numpy as np
import config
import h5py
import torch

def get_batch_iterator(split, batch_size=config.T_BATCH_SIZE, context_length=config.CONTEXT_LENGTH, train_path=config.TRAIN_PATH, dev_path=config.DEV_PATH, device=config.DEVICE):
    hdf5_path = train_path if split=='train' else dev_path
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        dataset = hdf5_file['tokens']
        dataset_size = dataset.shape[0]
        n_examples = (dataset_size - 1) // context_length  # subtract 1 for y
        example_idxs = np.arange(n_examples)
        np.random.shuffle(example_idxs)
        epochs = 0
        counter = 0
        while True:
            if counter + batch_size > n_examples:
                np.random.shuffle(example_idxs)
                counter = 0
                print(f"Finished epoch {epochs}")
                epochs += 1
            random_indices = example_idxs[counter:counter+batch_size] * context_length
            random_samples = torch.tensor(np.array([dataset[idx:idx+context_length+1] for idx in random_indices]))
            xb = random_samples[:, :context_length].to(device)
            yb = random_samples[:, 1:context_length+1].to(device)
            counter += batch_size
            yield xb, yb
