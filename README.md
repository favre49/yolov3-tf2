# YoloV3 Implemented in TensorFlow 2.0

## Detecting with loaded model:

```
python detect_from_model.py --tiny \
--image test.png \
--model ./serving/yolov3/1 \
--classes ./kitti.names
```

## Detecting with loaded weights:

```
python detect.py --tiny \
--classes kitti.names \
--num_classes 1 \
--weights ./models/final/yolov3_train_12.tf \
--image test.png \
--tiny
```

## Detecting video with loaded weights:

```
python detect_video.py \
--classes kitti.names \
--num_classes 1 \
--weights ./models/final/yolov3_train_12.tf \
--video input.mp4 \
--output output.mp4 \
--tiny
```

## Detecting video with loaded model

```
python detect_video_from_model.py --tiny \
--model ./serving/yolov3/1 \
--video input.mp4 \
--classes ./kitti.names \
--output output.mp4
```

## Changing IOU threshold

See line 24 on yolov3_tf2/utils.py
