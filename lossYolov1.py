import torch.nn as nn
import torch


def IOU(box1, box2, format='midpoint'):
    assert (box1.shape == box2.shape)
    if format == 'centre':
        x1_box1 = box1[..., 0:1]
        y1_box1 = box1[..., 1:2]
        x2_box1 = box1[..., 2:3]
        y2_box1 = box1[..., 3:4]
        x1_box2 = box2[..., 0:1]
        y1_box2 = box2[..., 1:2]
        x2_box2 = box2[..., 2:3]
        y2_box2 = box2[..., 3:4]


    elif format == 'midpoint':
        box1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        box1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        box1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        box1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2
        box2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        box2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        box2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        box2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

    else:
        print("sai format")
    x1 = torch.max(x1_box1, x1_box2)
    y1 = torch.max(y1_box1, y1_box2)
    x2 = torch.min(x2_box1, x2_box2)
    y2 = torch.min(y2_box1, y2_box2)
    insection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    s1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    s2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
    return insection / (s1 + s2 - insection + 1e-6)


class lossYolo(nn.Module):
    def __init__(self, num_box=2, split_size=7, num_class=20, lamda_coord=5, lamda_no_object=0.5):
        super(lossYolo, self).__init__()
        self.num_box = num_box
        self.split_size = split_size
        self.num_class = num_class
        self.mse = nn.MSELoss(reduction='sum')
        self.lamda_coord = lamda_coord
        self.lamda_no_object = lamda_no_object

    def forward(self, predict, target):
        predict = predict.reshape(-1, self.split_size, self.split_size, self.num_box * 5 + self.num_class)

        iou1 = torch.unsqueeze(IOU(predict[..., 21:25], target[..., 21:25]), dim=0)
        iou2 = torch.unsqueeze(IOU(predict[..., 26:30], target[..., 21:25]), dim=0)

        max_iou, max_index = torch.cat((iou1, iou2)).max(dim=0)

        exis_object = target[..., 20:21]

        box_predict_target = max_index * predict[..., 25:30] + (1 - max_index) * predict[..., 20:25]
        class_predict_target = max_index * predict[..., :self.num_class] + (1 - max_index) * predict[...,
                                                                                             :self.num_class]

        box_predict_exis_object = box_predict_target * exis_object
        loss_midpoit = self.mse(box_predict_exis_object[..., 1:3], target[..., 21:23])
        box_weight_height = torch.sign(box_predict_exis_object[..., 3:5]) * torch.sqrt(
            torch.abs(box_predict_exis_object[..., 3:5]))

        loss_weight_height = self.mse(torch.sqrt(target[..., 23:25]), box_weight_height)
        loss_object = self.mse(target[..., 20:21], box_predict_exis_object[..., 0:1])
        loss_no_object = self.mse((1 - exis_object) * predict[..., 20:21], target[..., 20:21])

        loss_no_object += self.mse((1 - exis_object) * predict[..., 25:26], target[..., 20:21])
        loss_class = self.mse(target[..., :20], predict[..., :20])
        return self.lamda_coord * loss_midpoit + self.lamda_coord * loss_weight_height + self.lamda_no_object * loss_no_object + loss_class+loss_object

