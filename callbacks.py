import time


from .utils import AverageMeter


class Callback(object):
    def __init__(self):
        pass

    def on_begin(self, start_epoch=None, end_epoch=None, metrics_name=None):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_step_begin(self, size, eval_model=False):
        pass

    def on_batch_begin(self, batch, batch_size):
        pass

    def on_batch_end(self, metrics=None):
        pass

    def on_step_end(self):
        pass

    def on_epoch_end(self, msg, metrics=None):
        pass

    def on_end(self):
        pass

    def set_params(self, arch=None, optimizer=None, criterion=None):
        self.arch = arch
        self.optimizer = optimizer
        self.criterion = criterion


class Compose(Callback):
    def __init__(self, callbacks=[]):
        if len(callbacks) and not all([isinstance(c, Callback) for c in callbacks]):
            raise ValueError("All callbacks must be an instance of Callback")

        self.callbacks = callbacks

    def on_begin(self, start_epoch=None, end_epoch=None, metrics_name=None):
        for callback in self.callbacks:
            callback.on_begin(start_epoch, end_epoch, metrics_name)

    def on_end(self):
        for callback in self.callbacks:
            callback.on_end()

    def on_step_begin(self, size, mode=""):
        for callback in self.callbacks:
            callback.on_step_begin(size, mode=mode)

    def on_step_end(self):
        for callback in self.callbacks:
            callback.on_step_end()

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, metrics):
        for callback in self.callbacks:
            callback.on_epoch_end(metrics)

    def on_batch_begin(self, batch, batch_size):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, batch_size)

    def on_batch_end(self, metrics):
        for callback in self.callbacks:
            callback.on_batch_end(metrics)

    def set_params(self, arch=None, optimizer=None, criterion=None):
        for callback in self.callbacks:
            callback.set_params(arch, optimizer, criterion)


class Timemeasure(Callback):
    def __init__(self, dataset, print_freq=0):
        self.print_freq = print_freq
        self.train_time = AverageMeter()
        self.dataset = dataset

    def on_begin(self, start_epoch=None, end_epoch=None, metrics_name=None):
        self.end_epoch = end_epoch
        self.data_time = AverageMeter()
        self.train_end = time.time()

    def on_epoch_begin(self, epoch):
        self.epoch = epoch
        self.epoch_time = AverageMeter()

    def on_step_begin(self, size, eval_model=False):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.end = time.time()
        self.eval_model = eval_model

    def on_batch_begin(self, batch, batch_size):
        self.batch = batch
        self.data_time.update(time.time() - self.end)

    def on_batch_end(self, metrics=None):
        # measure elapsed time
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()

    def on_step_end(self):
        # reset timer to measure elapsed time only for evaluation
        self.end = time.time()

    def on_epoch_end(self, msg, metrics=None):
        # measure elapsed time
        if self.eval_model is True:
            self.eval_time.update(time.time() - self.end)
            self.end = time.time()
            msg += ["Eval time: {self.eval_time.avg:.3f})"]

        if self.epoch % self.print_freq == 0:
            msg += ["Batch time: {0.avg:.3f} ({0.sum:.3f}) ".format(self.batch_time)]
            msg += ["Data Time: {0.avg:.3f} ({0.sum:.3f}) ".format(self.data_time)]

            print("".join(msg))

    def on_end(self):
        self.train_time.update(time.time() - self.train_end)
        print(
            "{0} train time {1.val:.3f} ({1.avg:.3f})\t".format(
                self.dataset, self.train_time
            )
        )

