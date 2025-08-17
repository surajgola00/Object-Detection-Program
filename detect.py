import cv2
import numpy as np

# Use absolute path
cfg_path = r"C:\waste_detection\yolo\yolov4.cfg"
weights_path = r"C:\waste_detection\yolo\yolov4.weights"

# Load YOLO
net = cv2.dnn.readNet(weights_path, cfg_path)

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels (COCO dataset)
with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize colors for different classes
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def detect_objects(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Run inference
    outputs = net.forward(output_layers)

    # Process results
    boxes, confidences, class_ids = [], [], []
    confidence_threshold = 0.5

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x, center_y, w, h = detection[:4] * [width, height, width, height]
                x, y = int(center_x - w/2), int(center_y - h/2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    # Draw boxes
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        color = colors[class_ids[i]]
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, f"{classes[class_ids[i]]}: {confidences[i]:.2f}", 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Waste Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run detection
detect_objects("test.jpg")
