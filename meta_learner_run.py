import learn2learn as l2l
import torch
import torch.nn as nn

from meta_learner_module import MetaLearner
from scheduler.batch_loss_schedule import BatchLossSchedule
from scheduler.input_similarity_schedule import InputSimilaritySchedule
from scheduler.learning_progress_schedule import LearningProgressSchedule
from scheduler.prediction_similarity_schedule import PredictionSimilaritySchedule
from scheduler.random_schedule import RandomSchedule
from scheduler.task_loss_schedule import TaskLossSchedule

try:
    import wandb
except Exception as e:
    pass


def init_wandb_log(dataset, train_sample_size, n_test_labels, n_shots, per_task_lr, meta_lr, train_adapt_steps,
                   test_adapt_steps, meta_batch_size, n_epochs, schedule_alg):
    try:  # if wandb is configured and working, logs the run
        wandb.init(project=f'meta-task-cl-project', entity='liorf', save_code=True)
        config = wandb.config
        config.task = dataset
        config.teacher = str(schedule_alg)
        config.ways = n_test_labels
        config.shots = n_shots
        config.train_sample = train_sample_size
        config.task_lr = per_task_lr
        config.meta_lr = meta_lr
        config.train_adapt_steps = train_adapt_steps
        config.test_adapt_steps = test_adapt_steps
        config.batch_size = meta_batch_size
    except Exception as e:
        pass


def schedule_name_to_class(schedule_name, dataset, shots, ways):
    if schedule_name == "random":
        return RandomSchedule(dataset)
    elif schedule_name == "prediction-similar":
        return PredictionSimilaritySchedule(dataset, shots=shots, ways=ways, similar_first=True)
    elif schedule_name == "prediction-far":
        return PredictionSimilaritySchedule(dataset, shots=shots, ways=ways, similar_first=False)
    elif schedule_name == "input-similar":
        return InputSimilaritySchedule(dataset, shots=shots, ways=ways, similar_first=True)
    elif schedule_name == "input-far":
        return InputSimilaritySchedule(dataset, shots=shots, ways=ways, similar_first=False)
    elif schedule_name == "batch-loss-high":
        return BatchLossSchedule(dataset, shots=shots, ways=ways, hardest_first=True)
    elif schedule_name == "batch-loss-low":
        return BatchLossSchedule(dataset, shots=shots, ways=ways, hardest_first=False)
    elif schedule_name == "task-loss-high":
        return TaskLossSchedule(dataset, shots=shots, ways=ways, hardest_first=True)
    elif schedule_name == "task-loss-low":
        return TaskLossSchedule(dataset, shots=shots, ways=ways, hardest_first=False)
    elif schedule_name == "learning-progress":
        return LearningProgressSchedule(dataset, shots=shots, ways=ways)
    raise ValueError("not supported")


def run_meta_learner(
        dataset,
        train_sample_size,
        n_test_labels,
        n_shots,
        per_task_lr,
        meta_lr,
        train_adapt_steps,
        test_adapt_steps,
        meta_batch_size,
        n_epochs,
        schedule_name="random",
        seed=1):
    init_wandb_log(dataset, train_sample_size, n_test_labels, n_shots, per_task_lr, meta_lr,
                   train_adapt_steps, test_adapt_steps, meta_batch_size, n_epochs, schedule_name)

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
    train_schedule = schedule_name_to_class(schedule_name, task_sets.train, n_shots * 2, n_test_labels)

    print(f"load model (dataset is {dataset})")
    if dataset == "mini-imagenet":
        model = l2l.vision.models.MiniImagenetCNN(n_test_labels)
    else:
        model = l2l.vision.models.OmniglotCNN(n_test_labels)
    model.to(device)

    try:
        wandb.watch(model)
    except Exception as e:
        pass

    f_loss = nn.CrossEntropyLoss(reduction='mean')

    print(f"create meta learner")
    meta_learner = MetaLearner(
        per_task_lr,
        meta_lr,
        train_adapt_steps,
        test_adapt_steps,
        meta_batch_size,
        model,
        f_loss,
        device,
        seed)

    print(f"meta learner train")
    meta_learner.meta_train(n_epochs, train_schedule)
    del train_schedule

    print(f"meta learner test")
    meta_learner.meta_test(task_sets.test)
