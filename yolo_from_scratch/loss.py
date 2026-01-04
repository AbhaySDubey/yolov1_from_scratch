import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # before sending the inputs through the model, we need to reshape the input to (S,S,30)
        predictions = predictions.reshape(-1, self.S, self.S, self.C+(self.B*5))

        # note:
        # predictions[..., 0:20] are the class probabilities
        # predictions[..., 20] is the confidence score for bbox1
        # predictions[..., 21:25] is the co-ordinates for bbox1 
        # predictions[..., 25] is the confidence score for bbox2
        # predictions[..., 26:30] is the co-ordinates for bbox2
        
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        # this line first of all converts the iou tensor of shape (batch_size,S,S) to aa tensor of shape (1,batch_size,S,S)
        # further, when we concatenate them using torch.cat() along dim=0, we get a tensor of shape (2,batch_size,S,S)
        # this allows us to later apply torch.argmax(tensor, dim=0) to find the best bounding box (0 or 1) for each image along each grid cell
        # as a tensor of shape (batch_size,S,S) 
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        bestbox = torch.argmax(ious, dim=0)
        # iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3) # Iobj_i -> it tells us whether an object exists in that cell (0 or 1)

        # ======================= #
        #   FOR BOX COORDINATES   #
        # ======================= #
        # bestbox can be either 0 (box1 was correct) or 1 (box2 was correct)
        box_predictions = exists_box*(
            (
                bestbox*predictions[..., 26:30]+(1-bestbox)*predictions[..., 21:25]
            )
        )
        box_targets = exists_box*target[..., 21:25]

        # i've added 1e-6 post taking the absolute value for now, will change it to do it before taking the absolute value later
        # box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4])*torch.sqrt(
        #     torch.abs(box_predictions[..., 2:4])+1e-6
        # )

        # box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_predictions = torch.cat(
            [
                box_predictions[..., :2],
                torch.sign(box_predictions[..., 2:4]) *
                torch.sqrt(torch.abs(box_predictions[..., 2:4]) + 1e-6)
            ],
            dim=-1
        )

        box_targets = torch.cat(
            [
                box_targets[..., :2],
                torch.sqrt(box_targets[..., 2:4])
            ],
            dim=-1
        )

        # the reason behind doing end_dim=-2 (end_dim defines the last dimension to flatten)
        # which is the opposite of start_dim (that defines the first dimension to flatten)
        # is to ensure that we get the following shape conversion:
        # shape(original_tensor) => (N,S,S,4) -> (box_predictions or box_targets) 
        # shape(flattened_tensor) => (N*S*S,4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )


        # ======================= #
        #     FOR OBJECT LOSS     #
        # ======================= #
        pred_box = (
            bestbox*predictions[..., 25:26]
            +(1-bestbox)*predictions[..., 20:21]
        )

        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box*pred_box),
            torch.flatten(exists_box*target[..., 20:21])
        )
        
        
        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        # let's see which one works

        # start_dim=1 as the original implementation results in (N,S*S) tensor
        no_object_loss = self.mse(
            torch.flatten((1-exists_box)*predictions[..., 20:21], start_dim=1),
            torch.flatten((1-exists_box)*target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1-exists_box)*predictions[..., 25:26], start_dim=1),
            torch.flatten((1-exists_box)*target[..., 20:21], start_dim=1)
        )

        # end_dim=-2 results in (N*S*S) tensor
        # no_object_loss = self.mse(
        #     torch.flatten((1-exists_box)*predictions[..., 20:21], end_dim=-2),
        #     torch.flatten((1-exists_box)*target[..., 20:21], end_dim=-2)
        # )

        # no_object_loss += self.mse(
        #     torch.flatten((1-exists_box)*predictions[..., 25:26], end_dim=-2),
        #     torch.flatten((1-exists_box)*target[..., 20:21], end_dim=-2)
        # )

        # ======================= #
        #      FOR CLASS LOSS     #
        # ======================= #
        # the tensor is converted as follows:
        # shape(original_tensor) := (N,S,S,20) -> for 20 classes
        # shape(flattened_tensor) := (N*S*S,20) -> for 20 classes
        class_loss = self.mse(
            torch.flatten(exists_box*predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box*target[..., :20], end_dim=-2)
        )

        # actual loss, as implemented in the paper
        loss = (
            self.lambda_coord*box_loss  # first 2 rows of paper, the loss obtained from box co-ordinates
            +object_loss    # 3rd line of the paper, the object loss; it considers the identity multiplied by the probability that there exists an object in the cell
            +self.lambda_noobj*no_object_loss   # 4th row of the paper
            +class_loss  # 5th row of paper
        )

        return loss
