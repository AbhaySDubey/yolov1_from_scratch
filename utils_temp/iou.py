import torch

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


# if __name__ == "__main__":
    # bbox1 = []
    # calculate_iou