import tensorflow as tf
from absl import app,flags,logging
from absl.flags import FLAGS
import numpy as np
import cv2
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-trt-fp16-416', 'path to output')
flags.DEFINE_integer('input_size', 416, 'path to output')
flags.DEFINE_string('quantize_mode', 'float16', 'quantize mode (int8, float16)')
flags.DEFINE_string('dataset', "./coco_dataset/coco/5k.txt", 'path to dataset')
flags.DEFINE_integer('loop', 10, 'loop')

def save_trt():
    if FLAGS.quantize_mode == 'float16':
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                precision_mode=trt.TrtPrecisionMode.FP16,
                max_workspace_size_bytes=8000000000,
                max_batch_size=16)
        converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=FLAGS.weights, conversion_params=conversion_params)
        converter.convert()
    else:
        pass

    converter.save(output_saved_model_dir=FLAGS.output)
    print("Converted to TensorRT!")

    saved_model_loaded = tf.saved_model.load(FLAGS.output)
    graph_func = saved_model_loaded.signatures[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    trt_graph = graph_funv.graph.as_graph_def()
    for n in trt_graph.node:
        print(n.op)
        if n.op == "TRTEngineOp":
            print("Node: %s, %s" % (n.op, n.name.replace("/", "_")))
        else:
            print("Exclude Node: %s, %s" % (n.op, n.name.replace("/","_")))
    logging.info("model saved to: {}".format(FLAGS.output))

    trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
    print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
    all_nodes = len([1 for n in trt_graph.node])
    print("numb. of all_nodes in TensorRT graph:", all_nodes)


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    save_trt()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass







