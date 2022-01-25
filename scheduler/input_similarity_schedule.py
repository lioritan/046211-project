import random

import torch
import numpy as np
from learn2learn.data.task_dataset import TaskDataset
from scheduler.base_schedule import BaseSchedule
from scheduler.helpers import generate_new_classes, labels_to_dataset, get_evaluation_set


class InputSimilaritySchedule(BaseSchedule):
    def __init__(self, taskset: TaskDataset, shots, ways, similar_first=False):
        super().__init__(taskset)
        self.last_generation = None
        self.generated_classes = 0
        self.ways = ways
        self.shots = shots
        self.similar_first = similar_first

    def update_from_feedback(self, last_loss, last_predict=None, last_features=None):
        pass

    def get_next_task(self):
        label_list = list(self.taskset.dataset.labels_to_indices.keys())
        first_label = random.sample(label_list, 1)  # pick one randomly, the rest chosen greedily
        label_sample = labels_to_dataset(self.taskset.dataset, first_label, self.shots)[0]
        sim_func = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        similarities = []
        with torch.no_grad():
            for label in label_list:
                other_label_sample = labels_to_dataset(self.taskset.dataset, [label], self.shots)[0]
                similarities.append(sim_func(label_sample, other_label_sample).mean().item())
        if self.similar_first:
            most_similar = np.argsort(similarities)[::-1][1:self.ways]
            labels = np.array(label_list)[most_similar]
        else:
            most_dissimilar = np.argsort(similarities)[:self.ways - 1]
            labels = np.array(label_list)[most_dissimilar]
        labels = labels.tolist() + first_label
        self.last_generation = labels_to_dataset(self.taskset.dataset, labels, self.shots)
        return self.last_generation[0], self.last_generation[1]
