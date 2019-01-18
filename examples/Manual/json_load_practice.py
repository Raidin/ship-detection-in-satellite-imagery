import cv2
import json
import os
from matplotlib import pyplot as plt

# Ground Truth Structure
'''
|--- [Image File Name Dict]
    |--- [Object Type Dict]
        |--- [Coordinate List Dict]
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
                min_value = min(coordinate['geometry'])
                max_value = max(coordinate['geometry'])
                coord_dict = {'xmin': min_value['x'],
                                'ymin': min_value['y'],
                                'xmax': max_value['x'],
                                'ymax': max_value['y']}
                coord_list.append(coord_dict)
            obj_dict[obj_type] = coord_list
        ground_truth[data['External ID']] = obj_dict

    return ground_truth

def DisplayGT(gt_data):
    scenes_dir = '/home/jihunjung/ship_detection/data/scenes'
    for gt in gt_data.keys():
        img = cv2.imread(os.path.join(scenes_dir, gt))
        b, g, r = cv2.split(img)
        rgb_img = cv2.merge([r, g, b])
        plt.figure(figsize=(5, 5))
        plt.imshow(rgb_img)
        for obj in gt_data[gt].keys():
            for coord in gt_data[gt][obj]:
                width = coord['xmax'] - coord['xmin']
                height = coord['ymax'] - coord['ymin']
                rect = plt.Rectangle((coord['xmin'], coord['ymin']), width, height, fill=False, edgecolor=(1, 0, 0), linewidth=1)
                plt.gca().add_patch(rect)
    plt.show()

def main():
    gt_data = ParseGT()
    # DisplayGT(gt_data)
    return


if __name__ == '__main__':
    main()
