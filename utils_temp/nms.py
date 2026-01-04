import torch
from iou import intersection_over_union

def non_max_suppresssion(
        predictions, # predictions should be bboxes
        iou_threshold, # to be used to filter out multiple overlapping boxes for the same object
        threshold, # to be used to filter out boxes that have low probability of containing an object
        box_format="corners", # can be midpoint (cxcywh) or corners (xyxy) format
):
    # e.g. 
    # predictions = [[1,0.9,x1,y1,x2,y2]] -> note here, the box format considered is xyxy

    # assert type(
    bboxes = [box for box in predictions if box[1] > threshold] # filter out boxes based on confidence score
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        curr_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != curr_box[0]
            or intersection_over_union(torch.tensor(curr_box[2:]), torch.tensor(box[2:]), box_format=box_format) < iou_threshold
        ]

        bboxes_after_nms.append(curr_box)

    return bboxes_after_nms

