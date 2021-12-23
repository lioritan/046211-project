from abc import ABC
from learn2learn.data.task_dataset import TaskDataset


class BaseSchedule(ABC):
    def __init__(self, taskset: TaskDataset):
        # can use taskset.dataset.labels_to_indices
        # task size should be shots*ways (see taskset.task_transforms)
        self.taskset = taskset

    def update_from_feedback(self, last_loss, last_predict=None):
        raise NotImplementedError()

    def get_next_task(self):
        raise NotImplementedError()
