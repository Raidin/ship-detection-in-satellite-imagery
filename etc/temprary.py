import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

def ReadImage(image_path, idx):
    image = Image.open(image_path)
    pix = image.load()
    n_spectrum = 3
    width = image.size[0]
    height = image.size[1]
    # creat vector
    picture_vector = []
    for chanel in range(n_spectrum):
        for y in range(height):
            for x in range(width):
                picture_vector.append(pix[x, y][chanel])

    picture_vector = np.array(picture_vector).astype('uint8')
    picture_tensor = picture_vector.reshape([n_spectrum, height, width]).transpose(1, 2, 0)
    picture_tensor = picture_tensor.transpose(2, 0, 1)
    return picture_tensor

def main():
    data_dir = 'data/scenes'
    print os.listdir(data_dir)

    for idx, img_file in tqdm(enumerate(os.listdir(data_dir))):
        input_image = os.path.join(data_dir, img_file)
        ReadImage(input_image, idx)

def NumpyConcatenateExample():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[9, 8, 7], [6, 5, 4], [2, 5, 5]])
    c = np.concatenate((a, b))
    print a.shape
    print b.shape
    print c.shape
    print c

def DisplayLabelUsingCV2(picture_tensor):
    picture_tensor = cv2.cvtColor(picture_tensor, cv2.COLOR_RGB2BGR)
    display_text = '{0:.4f}'.format(0.1234)
    position = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
    text = dict(width=position[0][0], height=position[0][1], baseline=position[1])
    cv2.putText(picture_tensor, display_text, (100, 100 - text['baseline']), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
    picture_tensor = cv2.cvtColor(picture_tensor, cv2.COLOR_BGR2RGB)

    # Transpose to Image Type
    result_image = Image.fromarray(picture_tensor)
    save_str = 'zzzzzz.png'
    result_image.save(save_str, 'PNG')

def DisplayLabelUsingPlt():
    # rect = plt.Rectangle((e[0][0], e[0][1]),
    #                       80,
    #                       80, fill=False,
    #                       edgecolor=(1, 0, 0), linewidth=2.5)
    # plt.gca().add_patch(rect)
    # plt.gca().text(e[0][0], e[0][1], 'prob ::  {:.3f}'.format(e[1][0][1]),
    #                     bbox=dict(facecolor=(1, 0, 0), alpha=0.5), fontsize=9, color='white')
    return 0


if __name__ == '__main__':
    # main()
    NumpyConcatenateExample()
