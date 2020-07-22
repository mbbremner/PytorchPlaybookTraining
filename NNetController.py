""" Build out use cases for the NNTrainer object & other utils """

from torchsummary import summary


class NNetController:
    """
    Controller class, manages use cases for application.py
    """
    def __init__(self, trainer):
        """
        :param trainer: Model object passed from application level
        """
        self.trainer = trainer

    def UC_train_single_model(self, playbook):
        """
        Execute a training session for a single model
        :param playbook: Dictionary of training parameters
        :return: no return
        """
        self.trainer.update_playbook(playbook)
        self.trainer.initialize_training()
        self.trainer.train()

    def UC_train_many(self, playbook_list):
        """
        Execute many playbooks consecutively.
        If one sessions fails, move to the next one
        :param playbook_list: List of .json files
        :return: no return
        """
        for playbook in playbook_list:
            try:
                self.UC_train_single_model(playbook)
            except:
                print("Training failed, moving to next model")

    def UC_explore_learning_rate(self, lr_init, lr_max, multiply_by=2):
        self.trainer.probe_learning_rate(lr_init, lr_max, multiply_by)

    def UC_summarize_model(self, playbook):
        self.trainer.update_params(playbook)
        self.trainer.initialize_training()
        # shape = self.trainer.data_shape[1:]
        shape = [32, 32]
        summary(self.trainer.model, (3, shape[0], shape[1]))