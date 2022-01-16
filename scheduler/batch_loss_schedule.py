import random

import torch
import numpy as np
from learn2learn.data.task_dataset import TaskDataset
from scheduler.base_schedule import BaseSchedule
from scheduler.helpers import generate_new_classes, labels_to_dataset


class BatchLossSchedule(BaseSchedule):
    def __init__(self, taskset: TaskDataset, hardest_first=True):
        super().__init__(taskset)
        self.last_generation = None
        self.task_to_batch_error_mapping = {}
        self.generated_classes = 0
        self.ways = self.taskset.task_transforms[0].n
        self.shots = self.taskset.task_transforms[1].k
        self.hardest_first = hardest_first

    def update_from_feedback(self, last_loss, last_predict=None):
        unique_labels = set(self.last_generation[2])
        for unique_label in unique_labels:
            self.task_to_batch_error_mapping[unique_label] = last_loss.item()

    def get_next_task(self):
        # warm-up
        if self.generated_classes < len(self.taskset.dataset.labels):
            self.last_generation = generate_new_classes(self.taskset.dataset, self.generated_classes, self.ways, self.shots)
            self.generated_classes += self.ways
            return self.last_generation[0], self.last_generation[1]
        else:
            label_list = list(self.task_to_batch_error_mapping.keys())
            if self.hardest_first:
                hardest_tasks = np.argsort(list(self.task_to_batch_error_mapping.values()))
                labels = np.array(label_list)[hardest_tasks[:self.ways]]
            else:
                easiest_tasks = np.argsort(list(self.task_to_batch_error_mapping.values()))[::-1]
                labels = np.array(label_list)[easiest_tasks[:self.ways]]
            labels = labels.tolist()
            self.last_generation = labels_to_dataset(self.taskset.dataset, labels, self.shots)
            return self.last_generation[0], self.last_generation[1]
