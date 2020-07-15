import time
import numpy as np
from utils.helperFunctions import count_num_correct
from sklearn.metrics import confusion_matrix
import torch
import datetime


# Decorators
def tracking_decorator_outer(func):
    """  """
    def _tracking_decorator_outer(self, *args, **kwargs):
        """ Inter """

        # Reset running values for epoch
        self.tracker.running_loss, self.tracker.epoch_correct = 0.0, 0
        self.tracker.lr = self.optimizer.param_groups[0]['lr']
        self.tracker.epoch = self.epoch
        self.tracker.print_epoch_update()

        func(self, *args, **kwargs)

        # Update & Print
        self.tracker.outer_update(val_preds=self.val_preds, val_labs=self.val_labs)
        self.tracker.print_outer_update(val_flag=self.flags['val'], cm_flag=self.flags['print_cm_flag'],
                                        val_preds=self.val_preds, val_labels=self.val_labs)

    return _tracking_decorator_outer


def tracking_decorator_inner(func):
    """  """
    def _tracking_decorator_inner(self, *args, **kwargs):

        # Intra-epoch pre & post updates
        self.tracker.t_end_load = time.monotonic()
        self.tracker.t_start_inner = time.monotonic()
        self.tracker.batch_num = args[0]

        func(self, *args, **kwargs)

        self.tracker.inner_update(self.outputs, self.labels, self.loss)
        self.tracker.print_inner_update()
        # This is positioned as the very last action before cpu work on next
        # batch begins for maximum accuracy
        self.tracker.t_start_load = time.monotonic()

    return _tracking_decorator_inner


class DeepMetaTracker:

    """Base class for all of your deep learning tracking & counting needs"""

    def __init__(self):

        # Per epoch loss, made up of running loss components
        self.losses = []
        self.running_loss = 0

        self.data_train = None
        self.eval_data = None

        self.num_correct_epoch = {'train': 0, 'val': 0}
        self.accuracy = {'train': [], 'val': []}

        # Use these for heavy dude per element tracking
        self.labels = {'train': [], 'val': []}
        self.predictions = {'train': [], 'val': []}

        self.epoch_losses = []
        self.learning_rates = []

    def inner_update(self):
        """
        Update inner counters
        """

    def outer_update(self):
        """
        Update outer counters
        :return:
        """

    def print_outer_update(self):
        """
        Print an update on the console within the current epoch
        """

    def print_inner_update(self):
        """
        Print an update on the console after each epoch
        """

    def print_epoch_update(self):
        """
        :return:
        """
        pass

    def to_df(self, col_keys=None):
        pass
        # print(np.array([self.accuracy['train'], self.accuracy['val'], self.losses, self.learning_rates]))


class BasicMetaTracker(DeepMetaTracker):
    """A fairly specific TrainingMetaTracker for the current type of problems I am working on"""
    def __init__(self, train_len=50000, num_classes=10, val_len=2000, batch_size=64, n=10,  max_epochs=None):
        super(BasicMetaTracker, self).__init__()

        self.cm_flag = False
        self.num_classes = num_classes
        self.epoch_correct = 0
        self.max_epochs = max_epochs

        self.epoch_losses = []
        self.learning_rates = []

        self.train_len = train_len
        self.val_len = val_len
        self.batch_size = batch_size
        self.n = n
        self.modVal = int((train_len / self.batch_size) / n)

        self.epoch = 0
        self.lr = 0.0
        self.t_start = 0.0          # Over starting time for training

        self.t_start_inner = 0      # batch forward & back-prop time
        self.t_start_load = 0.0     # CPU time start
        self.t_end_load = 0.0       # CPU time end
        self.t_load = 0.0
        self.t_total = 0.0

        self.inner_time = 0         # GPU time
        self.loss_delta = 0.0       # Change in loss from previous
        self.val_correct = 0        # number of correct predictions

        # Confusion Matrix Stuff
        if self.cm_flag:
            self.cm_current, cm_previous = [], []
            self.cm_row_accuracies = [[] for i in range(self.num_classes)]
            self.cm_previous = [[0 for i in range(self.num_classes)] for i in range(self.num_classes)]
            self.cm_delta = [[0 for i in range(self.num_classes)] for i in range(self.num_classes)]

    def inner_update(self, outputs=None, labels=None, loss=None, lr=None):
        """
        Update counters after running a single batch
        """
        self.inner_time += time.monotonic() - self.t_start_inner
        batch_preds = [torch.argmax(output) for output in outputs]
        self.epoch_correct += len([pred for p, pred in enumerate(batch_preds) if pred == labels[p]])
        self.running_loss += loss.item()
        self.t_start_load = time.monotonic()
        self.t_load += self.t_start_load - self.t_end_load
        self.t_total = time.monotonic() - self.t_start

        # Uncomment below for item-wise prediction tracking
        # self.all_predictions[self.epoch].extend(batch_predictions)
        # self.all_labels[self.epoch].extend(self.labels.tolist())

    def outer_update(self, val_preds=None, val_labs=None):
        """
        Post-epoch update of counters
        """
        if (val_labs is not None) and (val_preds is not None):
            self.val_correct = count_num_correct(val_labs, val_preds)
            self.accuracy['val'].append(self.val_correct / self.val_len)

        self.accuracy['train'].append(self.epoch_correct / self.train_len)
        self.losses.append(np.mean(self.epoch_losses))
        self.learning_rates.append(self.lr)
        if self.epoch > 0:
            # Compute change in loss
            self.loss_delta = self.losses[self.epoch] - self.losses[self.epoch-1]

    def print_epoch_update(self):
        """ Top Banner update for each new epoch"""
        hours, mins, secs = [float(i) for i in str(datetime.timedelta(seconds=time.monotonic() - self.t_start)).split(':')]
        out_str = "--------< EPOCH %d/%d ... lr: %5.5f ... Elapsed: %2.0fh %2.0fm %2.0fs >-------" % (
            self.epoch + 1, self.max_epochs, self.lr, hours, mins, secs)
        print(' ' + '=' * len(out_str) + '\n ' + out_str +'\n' + ' ' + '=' * len(out_str))

    def print_inner_update(self):
        if (self.batch_num+1) % self.modVal == self.modVal-1:
            # duration = time.monotonic() - self.t_start
            duration_num_images = (self.batch_num+1) * self.batch_size + self.epoch * self.train_len
            ips_total = int(duration_num_images / self.t_total)
            ips_gpu = int(duration_num_images / self.inner_time)
            ips_cpu_portion = int(duration_num_images / self.t_load)
            print('\t> %6d/%6d Images, L = %.4f,  IMG/s: %4d(total), %4d(cpu), %4d(gpu),' %
                  ((self.batch_num + 1) * self.batch_size, self.train_len, self.running_loss / self.n,
                   ips_total, ips_cpu_portion, ips_gpu))
            # Reset running values
            self.t_update = time.monotonic()
            self.epoch_losses.append(self.running_loss / self.n)
            self.running_loss = 0.0

    def print_outer_update(self, val_flag=False, cm_flag=False, val_preds=None, val_labels=None):

        strngs = ('Mean Epoch Loss', 'Training Accuracy', 'Validation Accuracy')
        # 1. PRINT ACCURACIES & LOSS Δ's
        print("\t\t%-20s: %4.4f,\tΔ = %+4.4f" % (strngs[0], np.mean(self.epoch_losses), self.loss_delta))
        print("\t\t%-20s: %4.4f,\t%5d / %5d" % (strngs[1], self.epoch_correct / self.train_len, self.epoch_correct, self.train_len))
        if val_flag:
            print("\t\t%-20s: %4.4f,\t%5d / %5d" % (strngs[2], (self.val_correct) / self.val_len,  self.val_correct, self.val_len))

            self.val_correct = count_num_correct(val_labels, val_preds)
        # 2. PRINT CONFUSION MATRIX
        # if cm_flag:
        #     self.print_confusion_matrix(val_labels, val_preds)
            # self.update_results_plot()

        self.epoch_losses.clear()

    def print_confusion_matrix(self, labs, preds):
        """
        Display a confusion matrix for the current epoch
        WARNING: as n classes increases, memory may blow up, it's not exactly time-efficient to calculate a
        new confusion matrix every epoch around

        :param labs: ground truth labels
        :param preds: model predictions
        :return:
        """

        if len(labs) == len(preds):
            # Standard confusion matrix
            self.cm_current = confusion_matrix([str(i) for i in labs], [str(i) for i in preds])
            # Difference from previous confustion matrix
            self.cm_delta = np.array(self.cm_current) - np.array(self.cm_previous)
            self.cm_previous = self.cm_current
            # max_len1 = len(str(max([max(row) for row in self.cm_current])))
            row_accuracy = 0.0
            print("\t\t   %s" % ' '.join(["{:^10s}".format(self.classes[i]) for i in self.classNumbers]))
            for r, row in enumerate(self.cm_current):
                row_accuracy = row[r]/np.sum(row)
                self.cm_row_accuracies[r].append(row_accuracy)
                row_strings = ['%4s(%+4d)' % (str(item), self.cm_delta[r][i]) for i, item in enumerate(row)]

                # row_items = ['%4s' % (str(item)) for item in row]
                # delta_items = ['%+4d' % (item) for item in self.cm_delta[r]]

                print("\t%5s %s |   %s%4.2f" % (self.classes[r], ' '.join(row_strings), r'%', 100*row_accuracy))
                # print("\t\t  %s" % (' '.join(delta_items)))
            # print('\t\t\t' + '_' * (len(' '.join(row_strings))))
            # recalls = ['%10s' % ('%3.3f' % item) for item in compute_precision_vals(self.cm_current)]
            # print('\t\t%s' % ' '.join(recalls))
        else:
            print('Input length mismatch, len(labs) %d != len(preds) %d' % (len(labs), len(preds)))
            pass

    def to_df(self, col_keys=None):
        import pandas as pd

        if col_keys is None:
            col_keys = ('train_acc', 'val_acc', 'loss', 'lr')
        outputData = np.array([self.accuracy['train'], self.accuracy['val'],
                               self.losses, self.learning_rates]).T

        df = pd.DataFrame(outputData, columns=col_keys)
        return df
