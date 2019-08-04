from yolo import YOLO
from PIL import Image
import os

classes = ['coconut']


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image, _ = yolo.detect_image(image)
            r_image.show()


def detect_folder(folder):
    yolo = YOLO(score=0.1)
    with open('result.txt', 'w') as f:
        for file in os.listdir(folder):
            image = Image.open(os.path.join(folder, file))
            r_image, box_detected = yolo.detect_image(image)

            str_to_write = os.path.join(folder, file)

            for box in box_detected:
                str_to_write = str_to_write + " " + ','.join([str(b) for b in box])

            str_to_write += '\n'

            f.write(str_to_write)


if __name__ == '__main__':
    # detect_img(YOLO())
    detect_folder('./datasets/images')
