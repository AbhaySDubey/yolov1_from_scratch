import torch
import cv2
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# general implementation
# good for bboxes with shape (1,4)
def calculate_iou(bbox1, bbox2):
    # intersection area
    x1 = torch.max(bbox1[0], bbox2[0])
    x2 = torch.min(bbox1[2], bbox2[2])
    y1 = torch.max(bbox1[1], bbox2[1])
    y2 = torch.max(bbox1[3], bbox2[3])

    inter_area = (torch.abs(x2-x1)*torch.abs(y2-y1))

    # union area
    area1 = torch.abs(bbox1[2]-bbox1[0])*torch.abs(bbox1[3]-bbox1[1])
    area2 = torch.abs(bbox2[2]-bbox2[0])*torch.abs(bbox2[3]-bbox2[1])

    union_area = area1+area2-inter_area

    iou = inter_area/(union_area+(1e-6))

    return iou

# implementation that can be used in implementing yolo
# yolo uses the cxcywh format (midpoint format)
# where the co-ordinates for the center of the box and the height and width of the box are provided
def intersection_over_union(bboxes_preds, bboxes_truths, box_format="midpoint"):
    # bboxes_preds.shape is (N,4)
    # bboxes_truths.shape is (N,4)
    # here, N is the number of bboxes (batch-size)

    # in YOLO algorithm, the shape would be something like (N,S,S,4)

    if box_format == "midpoint":
        bbox1_x1 = bboxes_preds[..., 0:1]-(bboxes_preds[..., 2:3]/2)
        bbox1_y1 = bboxes_preds[..., 1:2]-(bboxes_preds[..., 3:4]/2)
        bbox1_x2 = bboxes_preds[..., 0:1]+(bboxes_preds[..., 2:3]/2)
        bbox1_y2 = bboxes_preds[..., 1:2]+(bboxes_preds[..., 3:4]/2)
        bbox2_x1 = bboxes_truths[..., 0:1]-(bboxes_truths[..., 2:3]/2)
        bbox2_y1 = bboxes_truths[..., 1:2]-(bboxes_truths[..., 3:4]/2)
        bbox2_x2 = bboxes_truths[..., 0:1]+(bboxes_truths[..., 2:3]/2)
        bbox2_y2 = bboxes_truths[..., 1:2]+(bboxes_truths[..., 3:4]/2)


    if box_format == "corners":
        bbox1_x1 = bboxes_preds[..., 0:1]
        bbox1_y1 = bboxes_preds[..., 1:2]
        bbox1_x2 = bboxes_preds[..., 2:3]
        bbox1_y2 = bboxes_preds[..., 3:4]
        bbox2_x1 = bboxes_truths[..., 0:1]
        bbox2_y1 = bboxes_truths[..., 1:2]
        bbox2_x2 = bboxes_truths[..., 2:3]
        bbox2_y2 = bboxes_truths[..., 3:4]

    x1 = torch.max(bbox1_x1, bbox2_x1)
    y1 = torch.max(bbox1_y1, bbox2_y1)
    x2 = torch.min(bbox1_x2, bbox2_x2)
    y2 = torch.min(bbox1_y2, bbox2_y2)

    # intersection area
    inter_area = (x2-x1).clamp(0) * (y2-y1).clamp(0)

    # union area
    area1 = abs((bbox1_x2-bbox1_x1)*(bbox1_y2-bbox1_y1))
    area2 = abs((bbox2_x2-bbox2_x1)*(bbox2_y2-bbox2_y1))

    union_area = area1+area2-inter_area

    return (inter_area/(union_area+1e-6))


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


# saving and loading model    
def save_checkpoint(state, filename="model_checkpoint.pth"):
    print("Saving Checkpoint...")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("Loading Checkpoint...")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])



# converting cellboxes relative to cells -> relative to image
def convert_cellboxes(predictions, S=7):
    """
        Converts cellbox predictions created relative to a grid cell (total SxS cells) to relative to image. To do this, we follow:
        x_img = (x_cells+cell_idx)/S
    """
    predictions = predictions.to('cpu')
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size,S,S,30)
    
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1*(1-best_box)+best_box*bboxes2
    
    # creates a tensor of shape (batch_size,S,S,1)
    cell_indices = torch.arange(S).repeat(batch_size,S,1).unsqueeze(-1)
    x = (best_boxes[..., :1]+cell_indices)
    y = (best_boxes[..., 1:2]+cell_indices.permute(0,2,1,3))
    w_h = (best_boxes[..., 2:4]+cell_indices)
    converted_bboxes = torch.cat((x,y,w_h), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
    
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )
    
    return converted_preds
    
def cellboxes_to_bboxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S*S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []
    
    for ex_idx in range(out.shape[0]):
        bboxes = []
        for bbox_idx in range(S*S):
            bboxes.append([x.item() for x in converted_pred[ex_idx,bbox_idx,:]])
        all_bboxes.append(bboxes)
    
    return all_bboxes

def get_bboxes(
    loader, model,
    iou_threshold, threshold,
    box_format="midpoint", 
    device="cuda"
):
    all_pred_bboxes = []
    all_true_bboxes = []
    
    model.eval()
    t_idx = 0
    
    for batch_idx, (x,labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            predictions = model(x)
            
        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_bboxes(labels)
        pred_bboxes = cellboxes_to_bboxes(predictions)
        
        for idx in range(batch_size):
            nms_boxes = non_max_suppresssion(
                pred_bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format
            )
            
            for nms_box in nms_boxes:
                all_pred_bboxes.append([t_idx]+nms_box)
                
            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_bboxes.append([t_idx]+box)
            
            t_idx += 1
            
    model.train()
    return all_pred_bboxes, all_true_bboxes


# plots bounding box to the image (current implementation uses matplotlib patches)
# would add cv2.rectangle method next - it's easier anyways
def plot_image(image, bboxes):
    img = np.array(image)
    h, w, _ = img.shape
    
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    for box in bboxes:
        box = box[2:]
        assert len(box)==4, "Box has more than 4 values. Required 4 values are (x,y,w,h)"
        tl_x = box[0]-(box[2]/2)
        tl_y = box[1]-(box[3]/2)
        
        rect = patches.Rectangle(
            (tl_x*w, tl_y*h),
            box[2]*w, box[3]*h,
            linewidth=1,
            edgecolor='b',
            facecolor='none'
        )
        
        ax.add_patch(rect)
        
    plt.show()