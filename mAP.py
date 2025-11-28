import torch
from collections import Counter
from iou import intersection_over_union

def average_precision(detections, ground_truths, iou_threshold, box_format="corners"):
    # this'll count the number of bounding boxes for the image_idx (gt[0])
    # say, if img_0 has 3 bboxes (for the current class)
    # and, img_1 has 5 bboxes
    # then amt_bboxes = {0:3, 1:5}
    amt_bboxes = Counter([gt[0] for gt in ground_truths])
    epsilon = 1e-6  # just for numerical stability (primarily used to avoid div. by 0)

    # this creates a tensor of zeros 
    # with the same size as the number of bboxes in the image (of the curr. class)
    # so, for the above example, we'll have:
    # amt_bboxes = {0:torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}
    # this is intended to record the first bbox per object (the one with the highest score)
    # the primary idea is to elimnate any dupalicate values or predictions  
    for key, val in amt_bboxes.items():
        amt_bboxes[key] = torch.zeros(val)

    detections.sort(key=lambda x: x[2], reverse=True)
    TP = torch.zeros(len(detections))
    FP = torch.zeros(len(detections))
    total_true_bboxes = len(ground_truths)

    if total_true_bboxes == 0:
        return -1
    
    for det_idx, pred in enumerate(detections):
        gt_img_bboxes = [
            bbox for bbox in ground_truths if bbox[0] == pred[0]
        ]

        num_gts = len(gt_img_bboxes)
        best_iou = 0

        for idx, gt in enumerate(gt_img_bboxes):
            iou = intersection_over_union(
                torch.tensor(gt[3:]), torch.tensor(pred[3:]), box_format=box_format
            )
            if iou > best_iou:
                best_iou = iou
                best_iou_idx = idx

        # we've to check if the bounding box is actually valid
        # to do so, we need to:
        # 1. check if the best_iou > iou_threshold
        # 2. check if this index has not been encountered before (this is isolating the bbox for a singular object) 
        if best_iou > iou_threshold and amt_bboxes[pred[0]][best_iou_idx] == 0:
            TP[det_idx] = 1
            amt_bboxes[pred[0]][best_iou_idx] = 1
        else:
            FP[det_idx] = 1

    # torch.cumsum(torch.tensor(), dim=dim) -> calculates the prefix_sum for the torch.tensor() passed in along the specified dimension
    # i.e. if we have array := [1,0,1,0,1] then torch.cumsum(array, dim=0) := [1,1,2,2,3]
    TP_aggr = torch.cumsum(TP, dim=0)
    FP_aggr = torch.cumsum(FP, dim=0)
    recalls = TP_aggr/(total_true_bboxes+epsilon)
    precisions = TP_aggr/(TP_aggr+FP_aggr+epsilon)

    # this is done to facilitate numerical integration,
    # the idea is that we need a point along the y axis to start the integration 
    # the integration gives the area of the precision-recall curve which is the average precision
    # in the precision-recall curve, the precision is plotted along Y-axis and recall along X-axis 
    # and, hence to calculate the avg. precision (area under this curve) where precision is plotted as a function of recall
    # we require the precision to start at 1 and recall at 0 
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))

    return torch.trapz(precisions, recalls)

def mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=0.5,
        box_format="corners",
        num_classes=20
):
    # pred_boxes (list): [[train_idx, class_pred, prob_score, x1, y1, x2, y2]] -> train_idx (image_idx)
    # true_boxes (list): [[train_idx, class_pred, prob_score, x1, y1, x2, y2]] -> train_idx (image_idx)
    average_precisions = []

    for c in range(num_classes):
        detections = []
        ground_truths = []

        detections = [det for det in pred_boxes if det[1] == c]
        ground_truths = [gt for gt in true_boxes if gt[1] == c]
        avg_precision = average_precision(detections=detections, ground_truths=ground_truths, iou_threshold=iou_threshold, box_format=box_format)
        if avg_precision != -1:
            average_precisions.append(average_precision(detections, ground_truths, iou_threshold, box_format=box_format))

    return sum(average_precisions)/len(average_precisions)
