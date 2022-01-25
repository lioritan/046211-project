from abc import ABC
from learn2learn.data.task_dataset import TaskDataset


class BaseSchedule(ABC):
    def __init__(self, taskset: TaskDataset):
        self.taskset = taskset

    def update_from_feedback(self, last_loss, last_predict=None, last_features=None):
        raise NotImplementedError()

    def get_next_task(self):
        raise NotImplementedError()
