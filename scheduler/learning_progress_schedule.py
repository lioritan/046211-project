import random

import torch
import torch.nn as nn
import numpy as np
from learn2learn.data.task_dataset import TaskDataset
from scheduler.base_schedule import BaseSchedule
from scheduler.helpers import generate_new_classes, labels_to_dataset, get_evaluation_set


class LearningProgressSchedule(BaseSchedule):
    def __init__(self, taskset: TaskDataset, shots, ways):
        super().__init__(taskset)
        self.last_generation = None
        self.task_to_alp_mapping = {}
        self.generated_classes = 0
        self.ways = ways
        self.shots = shots

    def update_from_feedback(self, last_loss, last_predict=None):
        last_labels = self.last_generation[2] # divide to true false
        _, labels = get_evaluation_set(last_labels.size, None, last_labels)
        for unique_label in set(last_labels):
            inds = (labels == unique_label)
            lbl_predict = last_predict[inds, -1]
            loss = nn.MSELoss(reduction='mean')
            with torch.no_grad():
                actual_labels = torch.tensor(labels[inds], device=last_predict.device)
                task_loss = loss(lbl_predict, actual_labels)
                if unique_label not in self.task_to_alp_mapping:
                    self.task_to_alp_mapping[unique_label] = 0.0
                self.task_to_alp_mapping[unique_label] = abs(task_loss - self.task_to_alp_mapping[unique_label])

    def get_next_task(self):
        # warm-up
        if self.generated_classes < len(self.taskset.dataset.labels):
            self.last_generation = generate_new_classes(self.taskset.dataset, self.generated_classes, self.ways, self.shots)
            self.generated_classes += self.ways
            return self.last_generation[0], self.last_generation[1]
        else:
            label_list = list(self.task_to_alp_mapping.keys())
            highest_progress = np.argsort(list(self.task_to_alp_mapping.values()))
            labels = np.array(label_list)[highest_progress[:self.ways]]
            labels = labels.tolist()
            self.last_generation = labels_to_dataset(self.taskset.dataset, labels, self.shots)
            return self.last_generation[0], self.last_generation[1]
