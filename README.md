# PaddleInference-generic-demo
Classic demo provided for AI accelerators adapted in Paddle Inference.
## Model test
```
cd model_test/shell
```
### CPU
- Arm CPU (Android)
  ```
  ./build.sh android arm64-v8a cpu
  ./run_with_adb.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh conv_bn_relu_224_int8_per_channel 1,3,224,224 float32 float32 android arm64-v8a cpu UQG0220A15000356

  ./build.sh android armeabi-v7a cpu
  ./run_with_adb.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh conv_bn_relu_224_int8_per_channel 1,3,224,224 float32 float32 android armeabi-v7a cpu UQG0220A15000356
  ```
- Arm CPU (Linux)
  ```
  ./build.sh linux arm64 cpu
  ./run.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux arm64 cpu
  ./run.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 linux arm64 cpu
  ./run.sh conv_bn_relu_224_int8_per_channel 1,3,224,224 float32 float32 linux arm64 cpu
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh conv_bn_relu_224_int8_per_channel 1,3,224,224 float32 float32 linux arm64 cpu 192.168.100.30 22 khadas khadas

  ./build.sh linux armhf cpu
  ./run.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux armhf cpu
  ./run.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 linux armhf cpu
  ./run.sh conv_bn_relu_224_int8_per_channel 1,3,224,224 float32 float32 linux armhf cpu
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux armhf cpu 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 linux armhf cpu 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh conv_bn_relu_224_int8_per_channel 1,3,224,224 float32 float32 linux armhf cpu 192.168.100.13 22 root rockchip
  ```
- x86 CPU (Linux)
  ```
  ./build.sh linux amd64 cpu
  ./run.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux amd64 cpu
  ./run.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 linux amd64 cpu
  ./run.sh conv_bn_relu_224_int8_per_channel 1,3,224,224 float32 float32 linux amd64 cpu
  ./run_with_ssh.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh conv_add_144_192_int8_per_layer 1,3,192,144 float32 float32 linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh conv_bn_relu_224_int8_per_channel 1,3,224,224 float32 float32 linux amd64 cpu localhost 9031 root root
  ```
### Kunlunxin XPU with XDNN
- x86 CPU + Kunlunxin K100 (Ubuntu)
  ```
  ./build.sh linux amd64 xpu
  ./run.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux amd64 xpu
  ```
- Arm CPU + Kunlunxin K200 (KylinOS)
  ```
  ./build.sh linux arm64 xpu
  ./run.sh conv_bn_relu_224_fp32 1,3,224,224 float32 float32 linux arm64 xpu
  ```

## Image classification demo based on MobileNet, ResNet etc.
```
cd image_classification_demo/shell
```
### CPU
- Arm CPU (Android)
  ```
  ./build.sh android arm64-v8a cpu
  ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh mobilenet_v1_int8_224_per_channel imagenet_224.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh resnet50_int8_224_per_layer imagenet_224.txt test android arm64-v8a cpu UQG0220A15000356

  ./build.sh android armeabi-v7a cpu
  ./run_with_adb.sh mobilenet_v1_fp32_224 imagenet_224.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh mobilenet_v1_int8_224_per_channel imagenet_224.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh resnet50_fp32_224 imagenet_224.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh resnet50_int8_224_per_layer imagenet_224.txt test android armeabi-v7a cpu UQG0220A15000356
  ```
- Arm CPU (Linux)
  ```
  ./build.sh linux arm64 cpu
  ./run.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux arm64 cpu
  ./run.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux arm64 cpu
  ./run.sh mobilenet_v1_int8_224_per_channel imagenet_224.txt test linux arm64 cpu
  ./run.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 cpu
  ./run.sh resnet50_int8_224_per_layer imagenet_224.txt test linux arm64 cpu
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh mobilenet_v1_int8_224_per_channel imagenet_224.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh resnet50_int8_224_per_layer imagenet_224.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas

  ./build.sh linux armhf cpu
  ./run.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux armhf cpu
  ./run.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux armhf cpu
  ./run.sh mobilenet_v1_int8_224_per_channel imagenet_224.txt test linux armhf cpu
  ./run.sh resnet50_fp32_224 imagenet_224.txt test linux armhf cpu
  ./run.sh resnet50_int8_224_per_layer imagenet_224.txt test linux armhf cpu
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh mobilenet_v1_int8_224_per_channel imagenet_224.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh resnet50_int8_224_per_layer imagenet_224.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ```
- x86 CPU (Linux)
  ```
  ./build.sh linux amd64 cpu
  ./run.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux amd64 cpu
  ./run.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux amd64 cpu
  ./run.sh mobilenet_v1_int8_224_per_channel imagenet_224.txt test linux amd64 cpu
  ./run.sh resnet50_fp32_224 imagenet_224.txt test linux amd64 cpu
  ./run.sh resnet50_int8_224_per_layer imagenet_224.txt test linux amd64 cpu
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh mobilenet_v1_int8_224_per_layer imagenet_224.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh mobilenet_v1_int8_224_per_channel imagenet_224.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh resnet50_int8_224_per_layer imagenet_224.txt test linux amd64 cpu localhost 9031 root root
  ```
### Kunlunxin XPU with XDNN
- x86 CPU + Kunlunxin K100 (Ubuntu)
  ```
  ./build.sh linux amd64 xpu
  ./run.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux amd64 xpu
  ./run.sh resnet50_fp32_224 imagenet_224.txt test linux amd64 xpu
  ./run_with_ssh.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux amd64 xpu localhost 9031 root root
  ./run_with_ssh.sh resnet50_fp32_224 imagenet_224.txt test linux amd64 xpu localhost 9031 root root
  ```
- Arm CPU + Kunlunxin K200 (KylinOS)
  ```
  ./build.sh linux arm64 xpu
  ./run.sh mobilenet_v1_fp32_224 imagenet_224.txt test linux arm64 xpu
  ./run.sh resnet50_fp32_224 imagenet_224.txt test linux arm64 xpu
  ```

## Object detection demo based on SSD, YOLO etc.
```
cd object_detection_demo/shell
```
### CPU
- Arm CPU (Android)
  ```
  ./build.sh android arm64-v8a cpu
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test android arm64-v8a cpu UQG0220A15000356

  ./build.sh android armeabi-v7a cpu
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test android armeabi-v7a cpu UQG0220A15000356
  ```
- Arm CPU (Linux)
  ```
  ./build.sh linux arm64 cpu
  ./run.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux arm64 cpu
  ./run.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test linux arm64 cpu
  ./run.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux arm64 cpu
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas

  ./build.sh linux armhf cpu
  ./run.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux armhf cpu
  ./run.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test linux armhf cpu
  ./run.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux armhf cpu
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ```
- x86 CPU (Linux)
  ```
  ./build.sh linux amd64 cpu
  ./run.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux amd64 cpu
  ./run.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test linux amd64 cpu
  ./run.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux amd64 cpu
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh ssd_mobilenet_v1_relu_voc_int8_300_per_layer ssd_voc_300.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux amd64 cpu localhost 9031 root root
  ```
### Kunlunxin XPU with XDNN
- x86 CPU + Kunlunxin K100 (Ubuntu)
  ```
  ./build.sh linux amd64 xpu
  ./run.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux amd64 xpu
  ./run.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux amd64 xpu
  ```
- Arm CPU + Kunlunxin K200 (KylinOS)
  ```
  ./build.sh linux arm64 xpu
  ./run.sh ssd_mobilenet_v1_relu_voc_fp32_300 ssd_voc_300.txt test linux arm64 xpu
  ./run.sh yolov3_mobilenet_v1_270e_coco_fp32_608 yolov3_coco_608.txt test linux arm64 xpu
  ```

## Keypoint detection demo based on PP-TinyPose etc.
```
cd keypoint_detection_demo/shell
```
### CPU
- Arm CPU (Android)
  ```
  ./build.sh android arm64-v8a cpu
  ./run_with_adb.sh tinypose_fp32_128_96 tinypose_128_96.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test android arm64-v8a cpu UQG0220A15000356

  ./build.sh android armeabi-v7a cpu
  ./run_with_adb.sh tinypose_fp32_128_96 tinypose_128_96.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test android armeabi-v7a cpu UQG0220A15000356
  ```
- Arm CPU (Linux)
  ```
  ./build.sh linux arm64 cpu
  ./run.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux arm64 cpu
  ./run.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test linux arm64 cpu
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas

  ./build.sh linux armhf cpu
  ./run.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux armhf cpu
  ./run.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test linux armhf cpu
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ./run_with_ssh.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test linux armhf cpu 192.168.100.13 22 root rockchip
  ```
- x86 CPU (Linux)
  ```
  ./build.sh linux amd64 cpu
  ./run.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux amd64 cpu
  ./run.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test linux amd64 cpu
  ./run_with_ssh.sh tinypose_fp32_128_96 tinypose_128_96.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh tinypose_int8_128_96_per_channel tinypose_128_96.txt test linux amd64 cpu localhost 9031 root root
  ```

## Semantic segmentation demo based on PP-LiteSeg/PP-HumanSeg etc.
```
cd semantic_segmentation_demo/shell
```
### CPU
- Arm CPU (Android)
  ```
  ./build.sh android arm64-v8a cpu
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test android arm64-v8a cpu UQG0220A15000356
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test android arm64-v8a cpu UQG0220A15000356

  ./build.sh android armeabi-v7a cpu
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test android armeabi-v7a cpu UQG0220A15000356
  ./run_with_adb.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test android armeabi-v7a cpu UQG0220A15000356
  ```
- Arm CPU (Linux)
  ```
  ./build.sh linux arm64 cpu
  ./run.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux arm64 cpu
  ./run.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test linux arm64 cpu
  ./run.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test linux arm64 cpu
  ./run.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux arm64 cpu
  ./run.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test linux arm64 cpu
  ./run.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test linux arm64 cpu
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test linux arm64 cpu 192.168.100.30 22 khadas khadas

  ./build.sh linux armhf cpu
  ./run.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux armhf cpu
  ./run.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test linux armhf cpu
  ./run.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test linux armhf cpu
  ./run.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux armhf cpu
  ./run.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test linux armhf cpu
  ./run.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test linux armhf cpu
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux armhf cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test linux armhf cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test linux armhf cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux armhf cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test linux armhf cpu 192.168.100.30 22 khadas khadas
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test linux armhf cpu 192.168.100.30 22 khadas khadas
  ```
- x86 CPU (Linux)
  ```
  ./build.sh linux amd64 cpu
  ./run.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux amd64 cpu
  ./run.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test linux amd64 cpu
  ./run.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test linux amd64 cpu
  ./run.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux amd64 cpu
  ./run.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test linux amd64 cpu
  ./run.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test linux amd64 cpu
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_layer cityscapes_512_1024.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_int8_512_1024_per_channel cityscapes_512_1024.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_layer human_224_398.txt test linux amd64 cpu localhost 9031 root root
  ./run_with_ssh.sh portrait_pp_humansegv1_lite_398x224_with_softmax_int8_224_398_per_channel human_224_398.txt test linux amd64 cpu localhost 9031 root root
  ```
### Kunlunxin XPU with XDNN
- x86 CPU + Kunlunxin K100 (Ubuntu)
  ```
  ./build.sh linux amd64 xpu
  ./run.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux amd64 xpu
  ./run.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux amd64 xpu
  ```
- Arm CPU + Kunlunxin K200 (KylinOS)
  ```
  ./build.sh linux arm64 xpu
  ./run.sh pp_liteseg_stdc1_cityscapes_1024x512_scale_1_0_160k_with_argmax_fp32_512_1024 cityscapes_512_1024.txt test linux arm64 xpu
  ./run.sh portrait_pp_humansegv1_lite_398x224_with_softmax_fp32_224_398 human_224_398.txt test linux arm64 xpu
  ```
