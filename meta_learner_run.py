import learn2learn as l2l
import torch
import torch.nn as nn

from meta_learner_module import MetaLearner
from scheduler.batch_loss_schedule import BatchLossSchedule
from scheduler.prediction_similarity_schedule import PredictionSimilaritySchedule
from scheduler.random_schedule import RandomSchedule


def run_meta_learner(
        dataset, train_sample_size, n_test_labels, n_shots,
        per_task_lr, meta_lr, adaptation_steps, meta_batch_size,
        n_epochs):

    # shots = adaptation samples

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("get tasks")
    task_sets = l2l.vision.benchmarks.get_tasksets(
        dataset,
        train_samples=train_sample_size,
        train_ways=n_test_labels,
        test_samples=2 * n_shots,
        test_ways=n_test_labels,
        root='~/data')

    print("schedule training")
    train_schedule = RandomSchedule(task_sets.train)
    train_schedule = PredictionSimilaritySchedule(task_sets.train, shots=n_shots, ways=n_test_labels, similar_first=True)
    train_schedule = BatchLossSchedule(task_sets.train, shots=n_shots, ways=n_test_labels, hardest_first=True)

    print(f"load model (dataset is {dataset})")
    if dataset == "mini-imagenet":
        model = l2l.vision.models.MiniImagenetCNN(n_test_labels)
    else:
        model = l2l.vision.models.OmniglotCNN(n_test_labels)
    model.to(device)

    loss = nn.CrossEntropyLoss(reduction='mean')

    print(f"create meta learner")
    meta_learner = MetaLearner(per_task_lr, meta_lr, adaptation_steps, meta_batch_size, model, loss, device)

    print(f"meta learner train")
    meta_learner.meta_train(n_epochs, train_schedule)

    print(f"meta learner test")
    meta_learner.meta_test(task_sets.test)
