
def IntersectionOverUnion(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    area_intersection = (x2 - x1) * (y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[2])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[2])
    area_union = area_box1 + area_box2 - area_intersection

    iou = area_intersection / area_union
    return iou
