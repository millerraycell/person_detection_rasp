import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import os

MODEL_PATH = 'PersonDetectionModel/detect.tflite'
LABELMAP_PATH = 'PersonDetectionModel/labelmap.txt'
CONFIDENCE_THRESHOLD = 0.5

OUTPUT_DIR = 'detected_frames'
FILENAME_PREFIX = 'person_detection'

def parse_label_map():
    with open(LABELMAP_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    if labels[0] == '???':
        del(labels[0])
    return labels

if __name__ == "__main__":
    labels = parse_label_map()
    frame_count = 0

    # Load tflite model
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Couldn't find camera.")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("No frame to check")
            break
        
        input_data = cv2.resize(frame, (input_width, input_height))
        input_data = np.expand_dims(input_data, axis=0).astype(np.uint8)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])

        # --- Draw bounding boxes for detected people ---
        for i in range(num_detections):
            if scores[i] > CONFIDENCE_THRESHOLD:
                # Check if the detected class is 'person' (class ID 0 in COCO models)
                class_id = int(classes[i])
                if class_id < len(labels) and labels[class_id] == 'person':
                    # Get bounding box coordinates (they are normalized between 0 and 1)
                    ymin, xmin, ymax, xmax = boxes[i]
                    # Denormalize and scale to the original frame size
                    h, w, _ = frame.shape
                    xmin = int(xmin * w)
                    xmax = int(xmax * w)
                    ymin = int(ymin * h)
                    ymax = int(ymax * h)

                    # Draw the rectangle and label
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    label = f"Person: {int(scores[i]*100)}%"
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{FILENAME_PREFIX}_{timestamp}_{frame_count}.jpg"

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        filepath = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(filepath, frame)
        frame_count += 1

        time.sleep(1)  # Adjust the sleep time as needed

        if frame_count > 5:
            break

    cap.release()