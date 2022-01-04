import argparse
from meta_learner_run import run_meta_learner

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shots', default=5, type=int,
                        help="Number of training examples in the inner loop at meta-test time")
    parser.add_argument('--ways', default=5, type=int, help="Number of candidate labels (classes) at meta-test time")
    parser.add_argument('--train_size', default=20, type=int,
                        help="Number of training examples in the inner loop at meta-train time")
    parser.add_argument('--per_task_lr', default=0.01, type=int,
                        help="Per task LR for adaptation, should be high")
    parser.add_argument('--meta_lr', default=0.001, type=int,
                        help="Meta LR")
    parser.add_argument('--meta_batch_size', default=4, type=int,
                        help="Number of task gradients to average for meta-gradient step")
    parser.add_argument('--adaptation_steps', default=5, type=int,
                        help="Number of gradient steps to take during adaptation, if more than 1, consider lowering per_task_lr")
    parser.add_argument('--n_epochs', default=50, type=int,
                        help="Meta epochs for training")
    parser.add_argument('--dataset', default="mini-imagenet", choices=["mini-imagenet", "omniglot"],
                        help="Dataset to use.")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    run_meta_learner(dataset=args.dataset, train_sample_size=args.train_size, n_test_labels=args.ways, n_shots=args.shots,
         per_task_lr=args.per_task_lr, meta_lr=args.meta_lr, adaptation_steps=args.adaptation_steps, meta_batch_size=args.meta_batch_size,
         n_epochs=args.n_epochs
         )
    #20 error, 0.26 acc
    # why is improving cross entropy not improving accuracy at all?
