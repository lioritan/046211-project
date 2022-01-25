from learn2learn.data.task_dataset import TaskDataset

from scheduler.base_schedule import BaseSchedule


class RandomSchedule(BaseSchedule):
    def __init__(self, taskset: TaskDataset):
        super().__init__(taskset)

    def update_from_feedback(self, last_loss, last_predict=None, last_features=None):
        pass

    def get_next_task(self):
        return self.taskset.sample()
