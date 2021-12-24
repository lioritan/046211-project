import torch
import numpy as np
import learn2learn as l2l


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class MetaLearner(object):

    def __init__(self, per_task_lr, meta_lr, adaptation_steps, meta_batch_size, nn_model, device):
        self.meta_batch_size = meta_batch_size
        self.adaptation_steps = adaptation_steps
        self.device = device
        self.maml = l2l.algorithms.MAML(nn_model, lr=per_task_lr, first_order=False).to(device)
        self.opt = torch.optim.Adam(self.maml.parameters(), meta_lr)

    def calculate_meta_loss(self, task, learner, loss):
        data, labels = task
        data, labels = data.to(self.device), labels.to(self.device)

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
        for step in range(self.adaptation_steps):
            adaptation_error = loss(learner(adaptation_data), adaptation_labels)
            learner.adapt(adaptation_error)

        # Evaluate the adapted model
        predictions = learner(evaluation_data)
        evaluation_error = loss(predictions, evaluation_labels)
        evaluation_accuracy = accuracy(predictions, evaluation_labels)
        return evaluation_error, evaluation_accuracy

    def meta_train(self, n_epochs, train_schedule, loss):
        for iteration in range(n_epochs):
            self.opt.zero_grad()
            meta_train_error = 0.0
            meta_train_accuracy = 0.0

            for task in range(self.meta_batch_size):
                # Compute meta-training loss
                learner = self.maml.clone().to(self.device)
                batch = train_schedule.get_next_task()
                evaluation_error, evaluation_accuracy = self.calculate_meta_loss(batch, learner, loss)
                train_schedule.update_from_feedback(evaluation_error, last_predict=None)  # currently, don't return the
                # predicted result

                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

            # Average the accumulated gradients and optimize
            for p in self.maml.parameters():  # TODO: bad practice
                p.grad.data.mul_(1.0 / self.meta_batch_size)
            self.opt.step()
            # TODO: train loss improves, but accuracy does not!
            print(meta_train_error / self.meta_batch_size, meta_train_accuracy / self.meta_batch_size)

    def meta_test(self, test_taskset, loss):
        # calculate test error
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(self.meta_batch_size):
            # Compute meta-testing loss
            learner = self.maml.clone()
            batch = test_taskset.sample()
            evaluation_error, evaluation_accuracy = self.calculate_meta_loss(batch, learner, loss)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        print('Meta Test Error', meta_test_error / self.meta_batch_size, flush=True)
        print('Meta Test Accuracy', meta_test_accuracy / self.meta_batch_size, flush=True)
