import random

import torch
import numpy as np
from learn2learn.data.task_dataset import TaskDataset
from scheduler.base_schedule import BaseSchedule
from scheduler.helpers import generate_new_classes, labels_to_dataset


class PredictionSimilaritySchedule(BaseSchedule):
    def __init__(self, taskset: TaskDataset, similar_first=False):
        super().__init__(taskset)
        self.last_generation = None
        self.task_to_prediction_mapping = {}
        self.generated_classes = 0
        self.ways = self.taskset.task_transforms[0].n
        self.shots = self.taskset.task_transforms[1].k
        self.similar_first = similar_first

    def update_from_feedback(self, last_loss, last_predict=None):
        last_labels = self.last_generation[2] # divide to true false
        eval_inds = last_labels[np.arange(round(last_labels.size / 2)) * 2+1]
        unique_labels = set(last_labels)
        for unique_label in unique_labels:
            inds = (eval_inds == unique_label)
            lbl_predict = last_predict[inds, -1]
            self.task_to_prediction_mapping[unique_label] = lbl_predict

    def get_next_task(self):
        # warm-up
        if self.generated_classes < len(self.taskset.dataset.labels):
            self.last_generation = generate_new_classes(self.taskset.dataset, self.generated_classes, self.ways, self.shots)
            self.generated_classes += self.ways
            return self.last_generation[0], self.last_generation[1]
        else:
            label_list = list(self.task_to_prediction_mapping.keys())
            first_label = random.sample(label_list, 1)
            label_pred = self.task_to_prediction_mapping[first_label[0]]
            sim = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
            similarities = []
            with torch.no_grad():
                for label in self.task_to_prediction_mapping.keys():
                    similarities.append(sim(label_pred, self.task_to_prediction_mapping[label]).item())
            if self.similar_first:
                most_similar = np.argsort(similarities)[::-1][1:self.ways]
                labels = np.array(label_list)[most_similar]
            else:
                most_dissimilar = np.argsort(similarities)[:self.ways-1]
                labels = np.array(label_list)[most_dissimilar]
            labels = labels.tolist() + first_label
            self.last_generation = labels_to_dataset(self.taskset.dataset, labels, self.shots)
            return self.last_generation[0], self.last_generation[1]
