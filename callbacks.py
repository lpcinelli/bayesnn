import datetime
import io
import json
import time
import uuid

from .utils import AverageMeter

# import telegram


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

    def set_params(self, **kwargs):
        self.params = dict(kwargs)
        # self.arch = arch
        # self.optimizer = optimizer
        # self.criterion = criterion


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
            msg += ["Eval time: {:.3f}".format(time.time() - self.end)]
        self.end = time.time()

        if self.epoch % self.print_freq == 0:
            msg += ["Batch time: {0.avg:.3f} ({0.sum:.3f}) ".format(self.batch_time)]
            msg += ["Data time: {0.avg:.3f} ({0.sum:.3f}) ".format(self.data_time)]

            print("".join(msg))

    def on_end(self):
        self.train_time.update(time.time() - self.train_end)
        print(
            "{0} train time {1.val:.3f} ({1.avg:.3f})\t".format(
                self.dataset, self.train_time
            )
        )


class Monitor(Callback):
    """Monitor retrieve and continuously update the state a Keras model into the `self.state` attribute.

    Parameters
    ----------

    - plot_history : int
        Plot an history graph (logs against epoch) every nth epochs. No plot if None.
    - date_format = str
        Format of the date and time to display.
    """

    def __init__(self, name, plot_history=None, date_format="%Y-%m-%d %H:%M"):

        super().__init__()

        self.can_plot = False

        self.date_format = date_format
        self.plot_history = plot_history

        self.state = {}
        self.state["epochs"] = []

        # Add/generate these attributes in keras.models.Model ?
        self.state["name"] = name
        self.state["model_id"] = str(uuid.uuid4())
        self.state["training_id"] = str(uuid.uuid4())

    def on_train_begin(self, logs={}):

        message = (
            "Monitor initialized.\n"
            'Name of the model is "{}"\n'
            "Model ID is {}\n"
            "Training ID is {}"
        )
        self.notify(
            message.format(
                self.state["name"], self.state["model_id"], self.state["training_id"]
            )
        )

        # Add the model to the state
        self.state["model_json"] = self.model.to_json()

        self.state["params"] = self.params

        self.state["train_start_time"] = datetime.now()

        message = "Training of {} (N={}) started at {} for {} epochs."
        self.notify(
            message.format(
                self.params["dataset"],
                self.params["samples"],
                self.state["train_start_time"].strftime(self.date_format),
                self.params["epochs"],
            )
        )

    def on_train_end(self, logs={}):

        self.state["train_end_time"] = datetime.now()
        self.state["train_duration"] = (
            self.state["train_end_time"] - self.state["train_start_time"]
        )

        # In hours
        duration = (
            self.state["train_end_time"] - self.state["train_start_time"]
        ).total_seconds()
        self.state["train_duration"] = duration

        message = "Training is done at {} for a duration of {}s."
        self.notify(
            message.format(
                self.state["train_end_time"].strftime(self.date_format),
                self.state["train_duration"],
            )
        )

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

    def notify(self, message, parse_mode=None):
        pass

    def notify_image(self, fig):
        pass


class PrintMonitor(Monitor):
    """This monitor only print messages with the classic `print` function
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.can_plot = False

    def notify(self, message):
        print(message)

    def notify_image(self, fig):
        pass


class TelegramMonitor(Monitor):
    """This monitor send messages to a Telegram chat ID with a bot.
    """

    def __init__(self, api_token, chat_id, **kwargs):

        super().__init__(**kwargs)

        self.can_plot = True

        self.bot = telegram.Bot(token=api_token)
        self.chat_id = chat_id

    def notify(self, message, parse_mode=None):

        try:
            ret = self.bot.send_message(
                chat_id=self.chat_id, text=message, parse_mode=parse_mode
            )
            return ret
        except NewConnectionError:
            print("Error with notify() in TelegramMonitor.")
            return None

    def notify_image(self, fig):
        bf = io.BytesIO()
        fig.savefig(bf, format="png")
        bf.seek(0)

        try:
            self.bot.sendPhoto(chat_id=self.chat_id, photo=bf)
        except NewConnectionError:
            print("Error with notify_image() in TelegramMonitor.")
            return None


class FileMonitor(Monitor):
    """This monitor write a JSON file every time a message is sent. The JSON file contains all
        the state of the current training.
    """

    def __init__(self, filepath, **kwargs):

        super().__init__(**kwargs)

        self.can_plot = (
            False
        )  # we could save the figure as base64 and put it in JSON ...

        self.filepath = filepath

    def notify(self, message, parse_mode=None):

        with open(self.filepath, "w") as f:
            f.write(json.dumps(self.state, indent=4, sort_keys=True))

    def notify_image(self, fig):
        pass
