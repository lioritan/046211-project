import random

import torch
import numpy as np
from learn2learn.data.task_dataset import TaskDataset
from scheduler.base_schedule import BaseSchedule
from scheduler.helpers import generate_new_classes, labels_to_dataset, get_evaluation_set


class PredictionSimilaritySchedule(BaseSchedule):
    def __init__(self, taskset: TaskDataset, shots, ways, similar_first=False, memory_saving_mode=True):
        super().__init__(taskset)
        self.last_generation = None
        self.task_to_prediction_mapping = {}
        self.generated_classes = 0
        self.ways = ways
        self.shots = shots
        self.similar_first = similar_first
        # Saving the features for all tasks is pretty expensive, so we can save only a fraction and delete variables
        self.memory_saving_mode = memory_saving_mode

    def update_from_feedback(self, last_loss, last_predict=None, last_features=None):
        last_labels = self.last_generation[2]
        _, labels = get_evaluation_set(last_labels.size, None, last_labels)
        for unique_label in set(labels):
            inds = (labels == unique_label)
            lbl_features = last_features[inds, :]
            if self.memory_saving_mode:
                if unique_label > 2*len(self.taskset.dataset.labels)//3:
                    continue
                if unique_label in self.task_to_prediction_mapping:  # memory saving
                    del self.task_to_prediction_mapping[unique_label]
                torch.cuda.empty_cache()
            self.task_to_prediction_mapping[unique_label] = lbl_features

    def get_next_task(self):
        # warm-up
        if self.generated_classes < len(self.taskset.dataset.labels):
            self.last_generation = generate_new_classes(self.taskset.dataset, self.generated_classes, self.ways, self.shots)
            self.generated_classes += self.ways
            return self.last_generation[0], self.last_generation[1]
        else:
            label_list = list(self.task_to_prediction_mapping.keys())
            first_label = random.sample(label_list, 1)  # pick one randomly, the rest chosen greedily
            label_pred = self.task_to_prediction_mapping[first_label[0]]
            sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
            similarities = []
            with torch.no_grad():
                for label in self.task_to_prediction_mapping.keys():
                    new_pred = self.task_to_prediction_mapping[label]
                    similarities.append(sim(label_pred, new_pred).mean().item())
            if self.similar_first:
                most_similar = np.argsort(similarities)[::-1][1:self.ways]
                labels = np.array(label_list)[most_similar]
            else:
                most_dissimilar = np.argsort(similarities)[:self.ways-1]
                labels = np.array(label_list)[most_dissimilar]

            if self.memory_saving_mode:
                del similarities, sim
            labels = labels.tolist() + first_label
            self.last_generation = labels_to_dataset(self.taskset.dataset, labels, self.shots)
            return self.last_generation[0], self.last_generation[1]
