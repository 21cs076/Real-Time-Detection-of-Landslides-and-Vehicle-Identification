# Real-Time Detection of Landslides and Vehicle Identification

This project leverages computer vision and machine learning techniques to detect landslides and identify vehicles in real-time. The system uses YOLO (You Only Look Once) models trained on specific datasets for landslides and vehicles, and provides real-time alerts via SMS using Twilio.

## Features

- **Real-Time Detection**: Utilizes YOLO models to detect landslides and vehicles in video frames with high accuracy.
- **GUI Interface**: A user-friendly graphical interface built with Tkinter displays the video feed and detection results.
- **SMS Alerts**: Sends SMS alerts when a landslide is detected, including a summary of the vehicles affected.
- **Detection History**: Maintains a history of detections for a specified number of frames to visualize past detections.

## Dependencies

- OpenCV
- NumPy
- PyYAML
- Tkinter
- Pillow (PIL)
- Ultralytics YOLO
- Twilio

## Installation

```sh
pip install opencv-python-headless numpy pyyaml pillow twilio ultralytics
```

## Usage

1. Clone the repository and navigate to the project directory.
2. Place your trained YOLO models and corresponding `data.yaml` files in the specified paths.
3. Update the Twilio credentials in the `app.py` file.
4. Run the application:

```sh
python app.py
```

## File Structure

- **app.py**: Main application file that sets up the GUI, loads models, processes video frames, and sends SMS alerts.
- **models/**: Directory containing the YOLO model files.
- **yaml/**: Directory containing the `data.yaml` files for model class names.
