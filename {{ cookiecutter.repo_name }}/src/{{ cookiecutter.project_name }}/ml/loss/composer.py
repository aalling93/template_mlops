import torch.nn as nn
from typing import Union, Dict
import torch


class LossComposer(nn.Module):
    """ "
    #     WeightedSpeedLoss2DWeighted,
    used to compose the total loss using a combination of different losses..
    # target has size B x 11 x F x F, where F is e.g. 32
    # prediction has size B x (N) x F x F, where N depends on the model..
    """

    def __init__(
        self,
        weight1: float = 1,
        weight2: float = 1,
    ):
        super(LossComposer, self).__init__()


        # loss functions


        # loss internel regularizations

        # loss wights
        self.weight1 = weight1  # nn.Parameter(torch.tensor(weight1, dtype=torch.float32))
        self.weight2 = weight2  # nn.Parameter(torch.tensor(weight2, dtype=torch.float32))


        # for logging purposes only
        self.loss_dict = {
            "list_of_losses": [
                str(self.loss1),
                str(self.loss2),
            ],
            "weight1": self.weight1,
            "weight2": self.weight2,

        }

        self.outputs = {
            "loss": 0,
            "loss1": 0,
            "loss2": 0,
        }

        self.eps = 1e-6


    def forward(self, prediction, target):
        """

        """
        

        pass
