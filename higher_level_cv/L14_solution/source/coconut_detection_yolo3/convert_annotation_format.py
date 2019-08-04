import xml.etree.ElementTree as ET
import os
import random

ANN_PATH = './datasets/annotations'
IMG_PATH = './datasets/images'

TRAIN_ANN_FILE = './datasets/train_annotation.txt'
TEST_ANN_FILE = './datasets/test_annotation.txt'

TRAIN_RATE = 0.8
TEST_RATE = 0.2

CLASSES = ['coconut']


def convert_annotation(xml_file, file_to_write):
    in_file = open('./datasets/annotations/{}'.format(xml_file))
    tree = ET.parse(in_file)
    root = tree.getroot()

    image_path = root.find('path').text
    file_to_write.write(image_path)

    for obj in root.iter('object'):
        cls = obj.find('name').text

        if cls not in CLASSES:
            continue

        cls_id = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')

        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        file_to_write.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    file_to_write.write('\n')


list_xml = os.listdir(ANN_PATH)
random.shuffle(list_xml)
num_data = len(list_xml)
num_train = int(TRAIN_RATE * num_data)
num_test = num_data - num_train

with open(TRAIN_ANN_FILE, 'w') as f:
    for file in list_xml[:num_train]:
        convert_annotation(file, f)

with open(TEST_ANN_FILE, 'w') as f:
    for file in list_xml[num_train:]:
        convert_annotation(file, f)
