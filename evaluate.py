import tqdm
import os
import glob
import json
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def detect(yolo,filename):
    img_raw = tf.image.decode_image(open(filename, 'rb').read(), channels = 3)
    img = tf.expand_dims(img_raw,0)
    img = transform_images(img,FLAGS.size)
    boxes,scores,_,nums = yolo(img)

    return boxes,scores,nums


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')
    
    files = []
    
    with open("data/kitti_cars/final_test.txt",'r') as f:
        for line in f:
            line  = line.rstrip('\n')
            files.append("/home/favre49/Sources/yolov3-tf2/data/kitti_cars/training/image_2/"+line+".png")

    otherfiles = glob.glob("/home/favre49/Sources/yolov3-tf2/data/CDACData/vehicleDetection/testing/*/*.png")
    files = files + otherfiles

    pred_boxes = {}

    for filename in tqdm.tqdm(files):
        boxes,scores,nums = detect(yolo,filename)
        boxlist = [x for x in np.array(boxes[0,:]).tolist() if x != [0.0,0.0,0.0,0.0]]
        scorelist = [x for x in np.array(scores[0,:]).tolist() if x > 0.0001 ]

        pred_boxes[os.path.basename(filename)] = {'boxes':boxlist,'scores':scorelist}

    with open('pred_boxes.json','w', encoding='utf-8') as f:
        json.dump(pred_boxes,f,ensure_ascii=False,indent=4)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

