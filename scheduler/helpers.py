import torch
import numpy as np


def generate_new_classes(dataset, n_currently_generated_classes, ways, shots):
    batch_labels = dataset.labels[n_currently_generated_classes:n_currently_generated_classes + ways]
    if len(batch_labels) < ways:  # edge case - padding
        batch_labels.extend(dataset.labels[0:ways - len(batch_labels)])
    return labels_to_dataset(dataset, batch_labels, shots)


def labels_to_dataset(dataset, labels, shots):
    label_size = min([len(dataset.labels_to_indices[label]) for label in labels])
    random_sample = np.random.choice(np.arange(label_size), size=shots)
    indices = [dataset.labels_to_indices[label][ind] for label in labels for ind in random_sample]
    try:
        data, orig_labels = dataset[indices]
    except TypeError as e:  # not a task dataset
        data_tensor = None
        label_ndarray = np.zeros(len(indices))
        for i, ind in enumerate(indices):
            example_tensor, lbl = dataset[ind]
            label_ndarray[i] = lbl
            if data_tensor is None:
                data_tensor = example_tensor.unsqueeze(0)
            else:
                data_tensor = torch.concat((data_tensor, example_tensor.unsqueeze(0)), dim=0)
        data, orig_labels = data_tensor, label_ndarray

    label_convert_dict = {lbl: i for i, lbl in enumerate(labels)}
    labels = torch.tensor([label_convert_dict[lbl] for lbl in orig_labels])
    return data, labels, orig_labels


def get_evaluation_set(batch_size, data, labels):
    adapt_indices = np.zeros(batch_size, dtype=bool)
    train_frac = round(batch_size / 2)
    adapt_indices[np.arange(train_frac) * 2] = True
    error_eval_indices = ~adapt_indices

    # numpy -> torch
    error_eval_indices = torch.from_numpy(error_eval_indices)
    return data[error_eval_indices] if data is not None else None, \
           labels[error_eval_indices] if labels is not None else None
