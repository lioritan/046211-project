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
    indices = [dataset.labels_to_indices[label][ind] for label in labels for ind in random_sample ]
    data, orig_labels = dataset[indices]
    label_convert_dict = {lbl: i for i, lbl in enumerate(labels)}
    labels = torch.tensor([label_convert_dict[lbl] for lbl in orig_labels])
    return data, labels, orig_labels
