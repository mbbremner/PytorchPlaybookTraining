"""
Load training playbooks and go get some tea
Author: Michael Bremner, Feb. 2020
"""
# ------------------------------------------------------------------------
# -------------< Neural Net Training, Top Level Script >------------------
# ------------------------------------------------------------------------
from PlaybookTrainer import PlaybookTrainer
from NNetController import NNetController
from utils.helperFunctions import jsonLoad
from torch.backends import cudnn
cudnn.benchmark = True
import argparse


class DeepTrainingApplication:
    """
    Top level application for handling training use-cases
    """
    def __init__(self, pb):
        """
        Model, Controller, and Use Cases. No view class, sry nubs
        """
        # MODEL & CONTROLLER
        self.model = PlaybookTrainer(pb)
        self.controller = NNetController(self.model)
        self.USE_CASES = {'train_one': self.controller.UC_train_single_model,
                          'train_many': self.controller.UC_train_many,
                          'summarize': self.controller.UC_summarize_model,
                          'find_lr': self.controller.UC_explore_learning_rate
                         }

    def execute_use_case(self, uc_key, *args, **kwargs):
        """ User selects a use case with uc_key """

        if uc_key not in self.USE_CASES:
            raise KeyError("Invalid use case key. Valid keys: {}".format(self.USE_CASES.keys()))
        self.USE_CASES[uc_key](*args, **kwargs)
        return print('Use case %s complete' % uc_key)

    def parse_arguments(self, path_default):
        """
        If given through command line, select -pb,
            else return the default path
        """
        # playbook_path = self.parse_arguments(path)         # Check argparser first
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('-pb', nargs=1, type=str, default=[path_default])
        opts = arg_parser.parse_args()
        return opts.pb[0]

def main():

    playbooks = {
                    'cifar10': {
                                'vgg19': 'playbooks/playbook-cifar10-vgg19.json',
                                'res50': 'playbooks/playbook-res50-cifar10.json',
                                'res50clsa': 'playbooks/playbook-res50clsa-cifar10.json'},

                    'mnist':{'vgg19': 'playbooks/playbook-mnist-vgg19.json'},
                    'debug': {'vgg19': 'playbooks/PlaybookTemplate.json'}
                }

    pb_select = jsonLoad(playbooks['debug']['vgg19'])
    App_ = DeepTrainingApplication(pb_select)
    App_.execute_use_case('train_one', playbook=pb_select)
    # DeepLearner.execute_use_case('summarize', playbook=pb_selection)
    # DeepLearner.execute_use_case('find_lr', lr_init=0.00001, lr_max=0.5, multiply_by=1.35)

if __name__ == "__main__":
    main()


