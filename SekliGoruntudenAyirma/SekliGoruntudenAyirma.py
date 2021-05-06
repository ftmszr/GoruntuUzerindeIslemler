import cv2 
import numpy as np
import glob
import random

egitim = cv2.dnn.readNet("sekli_ayirma.cfg","sekli_ayirma.weights")#eğittiğin ağ yolo ile 

# nesnenin adı
classes = []
with open("sekil.txt", "r") as f:
    classes = f.read().splitlines()

images_path = glob.glob("Goruntu.jpg")#taranacak resimlerin yolunu belirtme

layer_names = egitim.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in egitim.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Insert here the path of your images
random.shuffle(images_path)
# loop through all the images
for img_path in images_path:
    # Loading image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=2, fy=2)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    egitim.setInput(blob)
    outs = egitim.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            #color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), (100,100,100), 1)
            cropped_img = img[y:y+h,x:x+w] 
            cv2.imwrite('sekil.jpg',cropped_img)
            cv2.imshow("sekil", cropped_img)
    cv2.imshow("Image", img)
    key = cv2.waitKey(0)
cv2.destroyAllWindows()