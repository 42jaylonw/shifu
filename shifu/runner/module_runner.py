from shifu.models.module import Module
from shifu.runner import latest_logdir, datetime_logdir

import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from shifu.utils.data import ShifuDataset


class ModuleRunner:
    dataset: [ShifuDataset, Dataset, DataLoader]

    def __init__(
            self,
            model: Module,
            lr=1e-4,
            weight_decay=1e-05,
            logdir=None,
            tensorboard_logdir=None,
            device='cuda:0',
            optimizer_class=torch.optim.Adam
    ):
        self.model = model
        self.logdir = logdir
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device

        self.start_train_time = 0
        self._len_dataset = 0

        if tensorboard_logdir is not None:
            self.logger = SummaryWriter(f"{tensorboard_logdir}/.tensorboard")

        self.model.to(device)
        self.optimizer = optimizer_class(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def train(
            self,
            dataset: [Dataset, DataLoader, ShifuDataset],
            log_interval=100,
    ):
        self.dataset = dataset
        self._len_dataset = len(dataset)
        self.start_train_time = time.time()

        self.model.train()
        for step, (data, label) in enumerate(dataset):
            self.optimizer.zero_grad()
            pred = self.model(data)
            loss, loss_logs = self.model.loss_func(pred, label)
            loss.backward()
            self.optimizer.step()

            # log
            if step % log_interval == 0:
                self.log('Train', step, loss_logs, data, label, pred)
                self.log_figure(step, loss_logs, data, label, pred)
                self.save()

    def play(
            self,
            dataset: [Dataset, DataLoader, ShifuDataset],
            log=True,
            log_figure=False
    ):
        self._len_dataset = len(dataset)
        self.start_train_time = time.time()
        self.load()
        for step, (data, label) in enumerate(dataset):
            pred = self.model(data)
            loss, loss_logs = self.model.loss_func(pred, label)
            if log:
                self.log('Eval', step, loss_logs, data, label, pred)
            if log_figure:
                self.log_figure(step, loss_logs, data, label, pred)

    def log(self, run_mode, step, loss_logs, data, label, pred, width=42):
        step_str = f"Iteration step: {step} / {self._len_dataset}"
        iteration_time = time.time() - self.start_train_time
        total_time = iteration_time / (step + 1) * self._len_dataset
        eta = total_time - iteration_time

        log_string = (f"{'#' * width}\n"
                      f"{step_str.center(width, ' ')}\n\n")

        log_string += f"    {run_mode}/\n"
        for loss_name, loss_val in loss_logs.items():
            log_string += f"        {loss_name} :{loss_val}\n"
            self.logger.add_scalar(f"{run_mode}/{loss_name}", loss_val, global_step=step, walltime=time.time())
        log_string += f"    \n"
        log_string += (f"   Iteration time: {iteration_time: .2f} s\n"
                       f"   Total time: {total_time: .2f} s\n"
                       f"   ETA: {eta: .2f} s\n")
        print(log_string)

    def save(self):
        self.model.save(self.logdir)

    def load(self):
        self.model.load(self.logdir)

    def log_figure(self, step, loss_logs, data, label, predicted):
        pass

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()


def run_module(
        run_mode,
        model,
        dataset,
        model_name,

        # train parm
        train_log_interval=100,

        # play param
        play_log=True,
        play_log_figure=False,

        log_root="./logs",
        runner_class=ModuleRunner,
        lr=1e-4,
        weight_decay=1e-05,
        device='cuda:0'

):
    print(f"\n\nRunning Model {model_name}\n")
    if run_mode == 'train':
        log_dir = datetime_logdir(log_root, model_name)
        runner = runner_class(model,
                              logdir=log_dir,
                              lr=lr,
                              weight_decay=weight_decay,
                              tensorboard_logdir=log_dir,
                              device=device)
        runner.train(dataset, train_log_interval)
    elif run_mode == 'play':
        play_log_root = f"{log_root}_play/{model_name}"
        load_logdir = latest_logdir(log_root, model_name)
        play_logdir = datetime_logdir(play_log_root, model_name)
        runner = runner_class(model,
                              logdir=load_logdir,
                              tensorboard_logdir=play_logdir,
                              device=device)
        runner.play(dataset,
                    log=play_log,
                    log_figure=play_log_figure)
    else:
        raise NotImplementedError

    runner.destroy()
