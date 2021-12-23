import argparse

import learn2learn as l2l
import torch
import torch.nn as nn
import numpy as np

from scheduler.random_schedule import RandomSchedule


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def calculate_meta_loss(task, learner, loss, adaptation_steps, device):
    data, labels = task
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation / evaluation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(round(data.size(0) / 2)) * 2] = True
    evaluation_indices = ~adaptation_indices

    # numpy -> torch
    evaluation_indices = torch.from_numpy(evaluation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy


def main(dataset, train_sample_size, n_test_labels, n_shots):
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

    per_task_lr = 0.5  # adaptation LR, should be high
    meta_lr = 0.005
    n_epochs = 30
    meta_batch_size = 32
    adaptation_steps = 1
    maml = l2l.algorithms.MAML(model, lr=per_task_lr, first_order=False).to(device)
    opt = torch.optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    for iteration in range(n_epochs):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0

        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone().to(device)
            batch = train_schedule.get_next_task()
            evaluation_error, evaluation_accuracy = calculate_meta_loss(batch, learner, loss, adaptation_steps, device)
            train_schedule.update_from_feedback(evaluation_error, last_predict=None)  # currently, don't return the
            # predicted result

            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

        # Average the accumulated gradients and optimize
        for p in maml.parameters(): # TODO: bad practice
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    # calculate test error
    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = tasksets.test.sample()
        evaluation_error, evaluation_accuracy = calculate_meta_loss(batch, learner, loss, adaptation_steps, device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()

    print('Meta Test Error', meta_test_error / meta_batch_size, flush=True)
    print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size, flush=True)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--multiplier', default=1, type=int,
                        help="Shot multiplier for maximum (=initial) support size.")
    parser.add_argument('--shots', default=5, type=int,
                        help="Number of training examples in the inner loop at meta-test time")
    parser.add_argument('--ways', default=5, type=int, help="Number of candidate labels at meta-test time")
    parser.add_argument('--freeze_l', action='store_true',
                        help="Should L be frozen to original value instead of matching support size?")
    parser.add_argument('--freeze_lr', action='store_true',
                        help="Static inner loop lr, or normalised by root of batch size?")
    parser.add_argument('--freeze_multiplier', action='store_true',
                        help="Freeze the support size to multiplier * shot for ablation study.")
    parser.add_argument('--dataset', default="mini-imagenet", choices=["mini-imagenet", "omniglot"],
                        help="Dataset to use.")
    parser.add_argument('--fc', action='store_true',
                        help="Use fully connected rather than convolutional back-bone. Only relevant for omniglot.")
    parser.add_argument('--resnet', action='store_true', help="Use resnet.")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(dataset=args.dataset, train_sample_size=42, n_test_labels=args.ways, n_shots=args.shots)
    pass
