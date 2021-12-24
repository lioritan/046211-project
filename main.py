import argparse

import learn2learn as l2l
import torch
import torch.nn as nn

from meta_learner_module import MetaLearner
from scheduler.random_schedule import RandomSchedule


def main(dataset, train_sample_size, n_test_labels, n_shots,
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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shots', default=5, type=int,
                        help="Number of training examples in the inner loop at meta-test time")
    parser.add_argument('--ways', default=5, type=int, help="Number of candidate labels (classes) at meta-test time")
    parser.add_argument('--train_size', default=80, type=int,
                        help="Number of training examples in the inner loop at meta-train time")
    parser.add_argument('--per_task_lr', default=0.5, type=int,
                        help="Per task LR for adaptation, should be high")
    parser.add_argument('--meta_lr', default=0.005, type=int,
                        help="Meta LR")
    parser.add_argument('--meta_batch_size', default=16, type=int,
                        help="Number of task gradients to average for meta-gradient step")
    parser.add_argument('--adaptation_steps', default=1, type=int,
                        help="Number of gradient steps to take during adaptation, if more than 1, consider lowering per_task_lr")
    parser.add_argument('--n_epochs', default=20, type=int,
                        help="Meta epochs for training")
    parser.add_argument('--dataset', default="mini-imagenet", choices=["mini-imagenet", "omniglot"],
                        help="Dataset to use.")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(dataset=args.dataset, train_sample_size=args.train_size, n_test_labels=args.ways, n_shots=args.shots,
         per_task_lr=args.per_task_lr, meta_lr=args.meta_lr, adaptation_steps=args.adaptation_steps, meta_batch_size=args.meta_batch_size,
         n_epochs=args.n_epochs
         )
    #20 error, 0.26 acc
    # why is improving cross entropy not improving accuracy at all?
