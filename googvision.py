import os
from google.cloud import vision
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set the path to your API key JSON file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

def detect_labels(path):
    """Detects labels in the file."""
    client = vision.ImageAnnotatorClient()

    with open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations

    print('Labels:')
    for label in labels:
        print(label.description)

    if response.error.message:
        raise Exception(f'{response.error.message}')

if __name__ == '__main__':
    # Path to the image file
    image_path = 'data/face.png'
    
    # Detect labels in the image
    detect_labels(image_path)