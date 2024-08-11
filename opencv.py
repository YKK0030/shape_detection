import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model.h5')

img_size = 180

# Function to classify images using the trained model
def classify_image(image):
    resized_image = cv2.resize(image, (img_size, img_size))
    input_image_array = np.array(resized_image, dtype=np.float32)
    input_image_array = np.expand_dims(input_image_array, axis=0)  # Add batch dimension
    input_image_array /= 255.0  # Normalize the image

    predictions = model.predict(input_image_array)
    result = tf.nn.softmax(predictions[0])
    class_name = flower_names[np.argmax(result)]
    confidence_score = np.max(result) * 100

    return class_name, confidence_score

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Classify the captured frame
    class_name, confidence_score = classify_image(frame)

    # Display the results on the frame
    cv2.putText(frame, f"{class_name}: {confidence_score:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('shape Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
