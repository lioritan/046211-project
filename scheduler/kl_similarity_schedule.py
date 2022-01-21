import random

import torch
import numpy as np
from learn2learn.data.task_dataset import TaskDataset
from scheduler.base_schedule import BaseSchedule
from scheduler.helpers import generate_new_classes, labels_to_dataset, get_evaluation_set


class KLSimilaritySchedule(BaseSchedule):
    def __init__(self, taskset: TaskDataset, shots, ways, similar_first=False):
        super().__init__(taskset)
        self.last_generation = None
        self.generated_classes = 0
        self.ways = ways
        self.shots = shots
        self.similar_first = similar_first

    def update_from_feedback(self, last_loss, last_predict=None):
        pass

    def get_next_task(self):
        # warm-up
        if self.generated_classes < len(self.taskset.dataset.labels):
            self.last_generation = generate_new_classes(self.taskset.dataset, self.generated_classes, self.ways, self.shots)
            self.generated_classes += self.ways
            return self.last_generation[0], self.last_generation[1]
        else:
            label_list = list(self.taskset.dataset.labels_to_indices.keys())
            first_label = random.sample(label_list, 1)
            label_sample = labels_to_dataset(self.taskset.dataset, first_label, self.shots)[0]
            kl_func = torch.nn.KLDivLoss()
            divergences = []
            with torch.no_grad():
                for label in label_list:
                    other_label_sample = labels_to_dataset(self.taskset.dataset, [label], self.shots)[0]
                    divergences.append(kl_func(label_sample, other_label_sample).item())
            if self.similar_first:
                most_similar = np.argsort(divergences)[::-1][1:self.ways]
                labels = np.array(label_list)[most_similar]
            else:
                most_dissimilar = np.argsort(divergences)[:self.ways-1]
                labels = np.array(label_list)[most_dissimilar]
            labels = labels.tolist() + first_label
            self.last_generation = labels_to_dataset(self.taskset.dataset, labels, self.shots)
            return self.last_generation[0], self.last_generation[1]
