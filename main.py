import argparse

import learn2learn as l2l
import torch
import torch.nn as nn

from meta_learner_module import MetaLearner
from scheduler.random_schedule import RandomSchedule


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
    n_epochs = 20
    meta_batch_size = 32
    adaptation_steps = 1

    meta_learner = MetaLearner(per_task_lr, meta_lr, adaptation_steps, meta_batch_size, model, device)
    loss = nn.CrossEntropyLoss(reduction='mean')

    meta_learner.meta_train(n_epochs, train_schedule, loss)

    meta_learner.meta_test(tasksets.test, loss)


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
    main(dataset=args.dataset, train_sample_size=50, n_test_labels=args.ways, n_shots=args.shots)
    #20 error, 0.26 acc
    # why is improving cross entropy not improving accuracy at all?
