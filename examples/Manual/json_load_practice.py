import cv2
import json
import os

from collections import namedtuple


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ground Truth Structure
'''
|--- [Image File Name Dict]
    |--- [Object Type Dict]
        |--- [Coordinate List]
            |--- [xmin, xmax, ymin, ymax]
'''
def ParseGT():
    json_data = open('/home/jihunjung/Downloads/ship_detection_scenes.json').read()
    data_dict = json.loads(json_data)

    ground_truth = dict()

    for idx, data in enumerate(data_dict):
        for obj_type in data['Label'].keys():
            coord_list = []
            obj_dict = dict()
            for coordinate in data['Label'][obj_type]:
                boxes = []
                min_value = min(coordinate['geometry'])
                max_value = max(coordinate['geometry'])
                boxes.append(min_value['x'])  # xmin(0)
                boxes.append(min_value['y'])  # ymin(1)
                boxes.append(max_value['x'])  # xmax(2)
                boxes.append(max_value['y'])  # ymax(3)
                coord_list.append(boxes)
            obj_dict[obj_type] = coord_list
        ground_truth[data['External ID']] = obj_dict
    return ground_truth

def DisplayGT(gt_data):
    scenes_dir = '/home/jihunjung/ship_detection/data/scenes'
    for gt in gt_data.keys():
        img = cv2.imread(os.path.join(scenes_dir, gt))
        b, g, r = cv2.split(img)
        rgb_img = cv2.merge([r, g, b])

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(rgb_img)
        for obj in gt_data[gt].keys():
            for boxes in gt_data[gt][obj]:
                x = boxes[0]
                y = boxes[1]
                w = boxes[2] - boxes[0]
                h = boxes[3] - boxes[1]
                rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=1)
                ax.add_patch(rect)
                '''
                rect = plt.Rectangle((boxes[0], boxes[1]), width, height, fill=False, edgecolor=(1, 0, 0), linewidth=1)
                plt.gca().add_patch(rect)
                '''
    plt.show()

def main():
    # detection = namedtuple("Detection", ["image_path", "gt", "pred"])
    gt_data = ParseGT()
    DisplayGT(gt_data)
    return

if __name__ == '__main__':
    main()
