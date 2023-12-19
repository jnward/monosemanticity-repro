import numpy as np
import h5py
import torch

def get_batch_iterator(data_path, batch_size, context_length, device="cpu"):
    with h5py.File(data_path, 'r') as hdf5_file:
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
