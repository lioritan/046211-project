import learn2learn as l2l
import torch
import torch.nn as nn

from meta_learner_module import MetaLearner
from scheduler.random_schedule import RandomSchedule


def run_meta_learner(
        dataset, train_sample_size, n_test_labels, n_shots,
        per_task_lr, meta_lr, adaptation_steps, meta_batch_size,
        n_epochs):
    # shots = adaptation samples
    tasksets = l2l.vision.benchmarks.get_tasksets(dataset,
                                                  train_samples=train_sample_size,
                                                  train_ways=n_test_labels,
                                                  test_samples=2 * n_shots,
                                                  test_ways=n_test_labels,
                                                  root='~/data')

    train_schedule = RandomSchedule(tasksets.train)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = l2l.vision.models.MiniImagenetCNN(
        n_test_labels) if dataset == "mini-imagenet" else l2l.vision.models.OmniglotCNN(
        n_test_labels).to(device)

    loss = nn.CrossEntropyLoss(reduction='mean')
    meta_learner = MetaLearner(per_task_lr, meta_lr, adaptation_steps, meta_batch_size, model, loss, device)

    meta_learner.meta_train(n_epochs, train_schedule)

    meta_learner.meta_test(tasksets.test)


