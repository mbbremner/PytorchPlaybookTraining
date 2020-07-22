"""
A Class for handling hyper-parameters & arguments with convenient JSON playbooks
Author: Michael Bremner
"""
import torch

from torch.utils import data
import torchvision.transforms as transforms
import torchvision
from torch.backends import cudnn
import importlib
cudnn.benchmark = True
from torchsummary import summary
import utils.helperFunctions as hf
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
from DeepMetaTracker import DeepMetaTracker

import utils.Selectors as Selectors
from models.MyNets import FcBlock



def select_module(module, attr_):
    """ Select a function (attr_) from a module obj (module)"""
    if not hasattr(module, attr_):
        raise TypeError('Bad attr. key: %s, not in the %s' % (attr_, str(module)))
    return getattr(module, attr_)

def module_select(keys):
    """
    Given a list of keys select a package in the following way:
    ['torch', 'nn', 'init'] returns torch.nn.init

    :param keys: list of keys to join / traverse
    :return: The desired module as shown in the example above
    """
    ret_val = None
    if len(keys) > 0:
        keys = ['torch', 'nn', 'init']
        attr_ = importlib.import_module(keys.pop(0))
        while len(keys) > 0:
            key = keys.pop(0)
            attr_ = getattr(attr_, key)
        ret_val = attr_
    return ret_val

def dynamic_select(args):
    if type(args) is list:
        pass
    elif type(args) is dict:
        pass
    elif type(args) is 'path':
        pass
    elif type(args) is str:
        pass

class NNetTrainer:

    def __init__(self):
        """Definition of fundamental instance variables"""
        self.tags = []      # List of obj attribute selections
        self.args = {}      # Dict of argument args associated with tags i.e. args[tag] = tag_args
        self.kwargs = {}    # Dict of argument kwargs associated with tags i.e. args[tag] = tag_kwargs

        self.inputs = []
        self.outputs = []
        self.labels = []
        self.batch_data = []

        self.epochs = 0         # Number of epochs to run
        self.e = 0              # Current Epoch

        self.dataset = None
        self.loader = None
        self.model = nn.Sequential()
        self.optimizer = None
        self.scheduler = None
        self.loss_function = None
        self.init_func = None

        self.tracker = DeepMetaTracker()            # Track thing such as accuracies and losses

    def train(self):

        # Training Loop
        input, labels = None, None
        for self.e in list(range(50)):
            epoch_loss = 0
            num_correct = 0
            total_predicted = 0
            for i, self.batch_data in enumerate(self.loader, 0):

                self.inputs, self.labels = self.batch_data[0].cuda(), self.batch_data[1].cuda()
                self.optimizer.zero_grad()
                self.outputs = self.model(self.inputs)
                loss = self.loss_function(self.outputs, self.labels)
                loss.backward()
                self.optimizer.step()

                pred_labels = [torch.argmax(output, dim=0).item() for output in self.outputs]
                # pred_labels = torch.argmax(outputs, axis=1)
                num_correct += len([(a, b) for a, b in zip(self.labels, pred_labels) if a == b])
                total_predicted += len(self.inputs)
                epoch_loss += loss

                if i % 50 == 5:
                    lr = [i['lr'] for i in self.optimizer.param_groups]
                    print('%s, Batch %s, Accuracy: %4.3f' % (lr, str(i), num_correct / total_predicted))

            print("Epoch Loss: %4.4f" % epoch_loss)
            self.scheduler.step()

    def probe_learning_rate(self, lr_init=0.0001, lr_max=1.0, multiply_by=2, n=10):

        """ Explore loss behavior as a function of learning rate to establish initial lr before training """
        losses = []
        current_lr = lr_init
        self.running_loss = 0.0

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_init

        for i, self.batch_data in enumerate(self.loader, 0):

            self.inputs = self.batch_data[0].cuda()
            self.labels = self.batch_data[1].cuda()

            # Forward, backward (side to side)
            self.optimizer.zero_grad()
            self.outputs = self.model(self.inputs)
            loss = self.loss_function(self.outputs, self.labels)
            loss.backward()
            self.optimizer.step()
            self.running_loss += loss.item()

            # Update learning rate every n epochs
            if i % n == n - 1:
                print("%3d. Loss: %5.5f, lr: %9.9f" % (i, self.running_loss / n, current_lr))
                losses.append((current_lr, self.running_loss / n))
                self.running_loss = 0
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= multiply_by
                    current_lr = param_group['lr']

            if current_lr > lr_max:
                print("\tMaximum learning rate encountered, breaking...")
                break

        # Display Results of lr adjustments
        def plot_lr(data):
            """ Plots the learning rate after probing"""
            fig = plt.figure()
            ax1 = plt.subplot(111)
            lr_vals = [l[0] for l in data]
            loss_vals = [l[1] for l in data]

            ax1.plot(lr_vals, loss_vals)
            plt.ylabel("loss")
            plt.xlabel("learning rate (log scale)")
            plt.xticks(np.arange(0, max(lr_vals), 0.005))
            plt.xscale('log', basex=10)
            plt.show()

        plot_lr(losses)


class PlaybookTrainer(NNetTrainer):

    def __init__(self, playbook):

        super(PlaybookTrainer, self).__init__()
        self.epochs = 50
        print('Initializing Playbook Trainer')
        self.update_playbook(playbook)

        print("%s, %s" % (self.args['optimizer'], self.kwargs['optimizer']))

        # Process transform arguments from the playbook dictionary into transform objects
        # self.kwargs['dataset']['transform'] = \
        #     transforms.Compose([select_module(module=transforms, attr_=key)(*args, **kwargs)
        #         for key, args, kwargs in self.kwargs['dataset']['transform']])
        #
        # print('Initialization Complete')

    def initialize_training(self):
        """
        Initializations are all in proper order and arguments are taken from
        the self's object attributes.
        :return: True upon successful completion
        """

        self.select_model()
        self.init_model_features()
        self.select_optim()
        self.select_dataset()
        self.select_scheduler()
        self.select_loader()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.model.classifier[-1].out_features = len(self.dataset.classes)
        self.model.cuda()
        return True


    def update_playbook(self, playbook):

        self.tags = {k: vals[0] for k, vals in playbook.items()}
        self.args = {k: vals[1] for k, vals in playbook.items()}
        self.kwargs = {k: vals[2] for k, vals in playbook.items()}

        self.kwargs['dataset']['transform'] = \
            transforms.Compose([select_module(module=transforms, attr_=key)(*args, **kwargs)
                for key, args, kwargs in self.kwargs['dataset']['transform']])


    def dynamic_training_init(self):
        """
        Initialize necessary training objeccts dynamically with names/tags
        :return: True upon successful completion
        """

        init_keys_in_order = ['model', 'optimizer', 'dataset', 'scheduler', 'loader']
        module_list = [torchvision.models, torch.optim, torchvision.datasets, torch.optim.lr_scheduler, data, torch.nn.init]
        modules = dict(zip(init_keys_in_order, module_list))

        arg_updates ={'optimizer': {'params': self.model.parameters},
                      'scheduler': {'optimizer': self.optimizer}
                      }

        for key in init_keys_in_order:
            if key == 'optimizer':
                self.kwargs['optimizer']['params'] = self.model.parameters()
            elif key == 'scheduler':
                self.kwargs['scheduler']['optimizer'] = self.optimizer
            elif key == 'loader':
                self.args['loader'].insert(0, self.dataset)
            setattr(self, key, select_module(module=modules[key], attr_=self.tags[key])
                                            (*self.args[key], **self.kwargs[key]))

        self.model.classifier[-1].out_features = len(self.dataset.classes)

        # Init loss function
        setattr(self, 'loss_function', torch.nn.CrossEntropyLoss())
        # Initialize model features
        if 'pretrained' in self.kwargs['model'] and self.kwargs['model']['pretrained'] is False:
            print("\t Iniitializing model %s with: %s" % (self.tags['model'], self.tags['init_func']))
            self.init_func = select_module(nn.init, self.tags['init_func'])
            Selectors.init_model_features(self.model, self.init, *self.args['init_func'], **self.kwargs['init_func'])

        self.model.cuda()
        print('Complete.')
        return True

    def select_model(self):
        self.model = select_module(torchvision.models, self.tags['model'])(*self.args['model'], **self.kwargs['model'])
        self.model.cuda()

    def select_optim(self):
        self.optimizer = select_module(torch.optim, self.tags['optimizer'])(
            self.model.parameters(), *self.args['optimizer'], **self.kwargs['optimizer'])

    def select_dataset(self):

        self.dataset = select_module(torchvision.datasets, self.tags['dataset'])(
            *self.args['dataset'], **self.kwargs['dataset'])

    def select_loader(self):
        self.loader = data.DataLoader(self.dataset, *self.args['loader'], **self.kwargs['loader'])

    def select_scheduler(self):

        self.scheduler = select_module(torch.optim.lr_scheduler, self.tags['scheduler'])(
            self.optimizer, *self.args['scheduler'], **self.kwargs['scheduler'])

    def init_model_features(self, model=None):
        """
        If not using pretrained features, apply init function
        :return: optional, return the model object
        """
        if model is None:
            model=self.model

        # 2. Select init func & initialize model params
        if 'pretrained' in self.kwargs['model'] and self.kwargs['model']['pretrained'] is False:
            print("\t Iniitializing model %s with: %s" % (self.tags['model'], self.tags['init_func']))
            for param in model.parameters():
                if len(param.shape) > 1:
                    self.init(param, *self.args['init_func'], **self.kwargs['init_func'])
        return self.model


def main():

    pt = PlaybookTrainer(playbook=hf.jsonLoad('playbooks/PlaybookTemplate.json'))
    pt.simple_training_init()
    print(len(list(pt.model.parameters())))
    pt.train()

    # pt.model = ResNet50_CLSA()
    # summary(model.cuda(), input_size=(3, 32, 32))
    exit()

    # print(pt.model[0].features)
    # pt.model = nn.Sequential(pt.model[0].features, FcBlock(outputs=10, sizes=[1024, 512, 64])).cuda()

    # pt.probe_learning_rate(lr_init=0.00001)
    summary(pt.model, input_size=(3, 32, 32))
    exit()
    pass


if __name__ == "__main__":
    main()

# self.model = Selectors.select_model(self.tags['model'], *self.args['model'], **self.kwargs['model'])
# self.optimizer = Selectors.select_optim(self.tags['optimizer'], self.model,
#                                         *self.args['optimizer'], **self.kwargs['optimizer'])
#
# self.dataset = Selectors.select_dataset(self.tags['dataset'],
#                                         *self.args['dataset'], **self.kwargs['dataset'])
# self.scheduler = Selectors.select_scheduler(tag=self.tags['scheduler'], optimizer=self.optimizer,
#                                             *self.args['scheduler'], **self.kwargs['scheduler'])
# self.loader = Selectors.select_loader(self.dataset,
#                                       *self.args['loader'], **self.kwargs['loader'])
# self.loss_function = torch.nn.CrossEntropyLoss()

# setattr(self, 'model', Selectors.select_model(self.tags['model'], *self.args['model'], **self.kwargs['model']))
# self.kwargs['optimizer']['params'] = self.model.parameters()
#
# setattr(self, 'optimizer', Selectors.select_optim(self.tags['optimizer'], *self.args['optimizer'], **self.kwargs['optimizer']))
# setattr(self, 'dataset', Selectors.select_dataset(self.tags['dataset'], *self.args['dataset'], **self.kwargs['dataset']))
#
# self.kwargs['scheduler']['optimizer'] = self.optimizer
# setattr(self, 'scheduler', Selectors.select_scheduler(self.tags['scheduler'], *self.args['scheduler'], **self.kwargs['scheduler']))
#
# self.args['loader'].insert(0, self.dataset)
# setattr(self, 'loader', Selectors.select_loader(self.tags['loader'], *self.args['loader'], **self.kwargs['loader']))

# exit()
# self.select_model()
# # self.init_model_features()
# self.select_optim()
# self.select_dataset()
# self.select_scheduler()
# self.select_loader()
# self.loss_function = torch.nn.CrossEntropyLoss()
# self.model.cuda()