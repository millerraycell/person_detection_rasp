# Person detection on Raspberry Pi 4

Using the coco_ssd_mobilenet_v1_1.0 tensorflow light model and the python libs tflite and openCV 

The python script will try to take pictures and if it recognizes it'll save to a detected_frames folder, the 5 pictures with a 1 second timeout

# How to run

- Setup the detection model
```
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d PersonDetectionModel
```

- Install the dependencies
```
python3 -m venv venv
source venv/bin/activate
pip install
```

- To run the detection
```
python person_detection.py
```

## Outputs

The 5 checked images will be located on the detected_frames folder
