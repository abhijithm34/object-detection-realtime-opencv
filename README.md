# Real-Time Object Detection with OpenCV

A Python-based real-time object detection system using OpenCV and MobileNet SSD. This project demonstrates how to implement real-time object detection using your computer's webcam.

## Features

- Real-time object detection using webcam feed
- Support for 20 different object classes
- High-performance detection using MobileNet SSD
- Visual bounding boxes and labels for detected objects
- FPS counter for performance monitoring

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/abhijithm34/object-detection-realtime-opencv.git
cd object-detection-realtime-opencv
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure you have a working webcam connected to your computer.

2. Run the object detection script:
```bash
python object_detector.py
```

3. The program will:
   - Open your webcam
   - Start detecting objects in real-time
   - Display bounding boxes around detected objects
   - Show the FPS counter in the top-left corner

4. Press 'q' to quit the application.

## How It Works

The system uses a pre-trained MobileNet SSD model to detect objects in real-time. The model is optimized for mobile and embedded vision applications, providing a good balance between speed and accuracy.

## Project Structure

- `object_detector.py`: Main script for object detection
- `model_config.prototxt`: Model architecture definition
- `model_weights.caffemodel`: Pre-trained model weights
- `requirements.txt`: Python package dependencies

## Troubleshooting

1. If you get a webcam error:
   - Make sure your webcam is properly connected
   - Check if another application is using the webcam
   - Try changing the camera index in `object_detector.py` (change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`)

2. If the model files are missing:
   - Make sure you have downloaded both `model_config.prototxt` and `model_weights.caffemodel`
   - Place them in the same directory as `object_detector.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Abhijith M
