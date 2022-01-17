import argparse
from config import defaults
from meta_learner_run import run_meta_learner

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=defaults['dataset'], choices=["mini-imagenet", "omniglot"],
                        help="Dataset to use.")
    parser.add_argument('--train_sample_size', default=defaults['train_sample_size'], type=int,
                        help="Number of training examples in the inner loop at meta-train time")
    parser.add_argument('--n_test_labels', default=defaults['n_test_labels'], type=int,
                        help="Number of candidate labels (classes) at meta-test time")
    parser.add_argument('--n_shots', default=defaults['n_shots'], type=int,
                        help="Number of training examples in the inner loop at meta-test time")
    parser.add_argument('--per_task_lr', default=defaults['per_task_lr'], type=int,
                        help="Per task LR for adaptation, should be high")
    parser.add_argument('--meta_lr', default=defaults['meta_lr'], type=int,
                        help="Meta LR")
    parser.add_argument('--train_adapt_steps', default=defaults['train_adapt_steps'], type=int,
                        help="Number of gradient steps to take during train adaptation, if more than 1, consider lowering per_task_lr")
    parser.add_argument('--test_adapt_steps', default=defaults['test_adapt_steps'], type=int,
                        help="Number of gradient steps to take during test adaptation, if more than 1, consider lowering per_task_lr")
    parser.add_argument('--meta_batch_size', default=defaults['meta_batch_size'], type=int,
                        help="Number of task gradients to average for meta-gradient step")
    parser.add_argument('--n_epochs', default=defaults['n_epochs'], type=int,
                        help="Meta epochs for training")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    run_meta_learner(
        dataset=args.dataset,
        train_sample_size=args.train_sample_size,
        n_test_labels=args.n_test_labels,
        n_shots=args.n_shots,
        per_task_lr=args.per_task_lr,
        meta_lr=args.meta_lr,
        train_adapt_steps=args.train_adapt_steps,
        test_adapt_steps=args.test_adapt_steps,
        meta_batch_size=args.meta_batch_size,
        n_epochs=args.n_epochs)

    #20 error, 0.26 acc
    # why is improving cross entropy not improving accuracy at all?
