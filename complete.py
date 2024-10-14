import time
import io
import cv2
from google.cloud import vision

# Initialize the Google Cloud Vision client
client = vision.ImageAnnotatorClient()

def capture_and_upload_image():
    # Open webcam feed
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the webcam feed
        cv2.imshow('Webcam', frame)

        # Save the frame as a temporary image
        image_path = 'data/webcam_capture.jpg'
        cv2.imwrite(image_path, frame)

        # Send the image to Google Vision API for analysis
        upload_image(image_path)

        # Wait for 5 seconds before capturing the next image
        time.sleep(5)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()

def upload_image(image_path):
    """Uploads an image to Google Cloud Vision API and retrieves labels."""
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Call the Vision API to detect labels in the image
    response = client.label_detection(image=image)
    
    # Process the response
    labels = response.label_annotations
    print("Labels detected:")
    for label in labels:
        print(f"{label.description} (confidence: {label.score})")

    if response.error.message:
        raise Exception(f'Error from Vision API: {response.error.message}')

# Run the function to capture images and send them to Google Vision API
capture_and_upload_image()
