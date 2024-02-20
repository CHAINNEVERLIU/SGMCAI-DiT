import random
import torch

import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss


class TrainBase(object):
    def __init__(self,
                 epochs,
                 train_loader,
                 optimizer,
                 loss_function,
                 lr,
                 device,
                 patience=None,
                 val_loader=None,
                 val_epochs=1,
                 checkpoint_path=None,
                 train_log_path=None):

        self.epoches = epochs
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.lr = lr
        self.device = device
        self.loss_function = loss_function
        self.val_epoches = val_epochs

        # 如果输入了patience参数就表示使用提前停止策略
        self.patience = patience
        if patience is not None:
            self.early_stopping = EarlyStopping(patience=patience, verbose=False)
            self.val_loader = val_loader
            self.checkpoint_path = checkpoint_path

            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)

        # 如果输入了train_log_path就表示使用tensorboard记录训练过程信息
        self.train_log_path = train_log_path
        if self.train_log_path is not None:
            if not os.path.exists(self.train_log_path):
                os.makedirs(self.train_log_path)

            self.Writer = SummaryWriter(self.train_log_path)

    @staticmethod
    def save_best_model(model, path):
        torch.save(model.state_dict(), path + '/' + 'best_model.pth')
        print("成功将此次训练模型存储(储存格式为.pth)至:" + str(path))

    def fit(self, model, *args, **kwargs):
        # 创建学习率调度器
        p1 = int(0.75 * self.epoches)
        p2 = int(0.9 * self.epoches)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[p1, p2], gamma=0.1
        )

        loop = tqdm(range(self.epoches), total=self.epoches)
        for epoch in loop:
            train_batch_loss = self._get_train_batch_loss(model, *args, **kwargs)

            # 接下来判断是否要使用tensorboard记录训练过程
            if hasattr(self, 'Writer'):
                self.Writer.add_scalar('Epoch batch loss/train', train_batch_loss,
                                       global_step=epoch, walltime=None)

            # 接下来判断是否要使用提前停止策略
            if self.patience is not None:
                if (epoch + 1) % self.val_epoches == 0:
                    val_batch_loss = self._get_val_batch_loss(model, *args, **kwargs)

                    if hasattr(self, 'Writer'):
                        self.Writer.add_scalar('Epoch batch loss/val', val_batch_loss,
                                               global_step=epoch, walltime=None)

                    self.early_stopping(val_batch_loss, model, self.checkpoint_path)
                    early_stop_information = '{}/{}'.format(self.early_stopping.counter, self.patience)
                    loop.set_postfix(loss=train_batch_loss,
                                     val_loss=val_batch_loss,
                                     early_stop=early_stop_information)

                    if self.early_stopping.early_stop:
                        break
            else:
                loop.set_postfix(loss=train_batch_loss)

            lr_scheduler.step()

        # 关闭SummaryWriter对象
        if hasattr(self, 'Writer'):
            self.Writer.close()

        if self.patience is not None:
            best_model_path = self.checkpoint_path + '/' + 'checkpoint.pth'
            model.load_state_dict(torch.load(best_model_path))

        return model

    def _get_train_batch_loss(self, model, *args, **kwargs):
        raise NotImplementedError(f'请先实现_get_train_batch_loss方法!')

    def _get_val_batch_loss(self, model, *args, **kwargs):
        raise NotImplementedError(f'请先实现_get_val_batch_loss方法!')


