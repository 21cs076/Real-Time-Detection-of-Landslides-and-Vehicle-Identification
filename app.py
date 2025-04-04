import cv2
import numpy as np
import yaml
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
from collections import deque

# Paths to YOLO models and data.yaml files
landslide_model_path = "models/Models/l67s.pt"
landslide_yaml_path = "yaml/landslide_data.yaml"
vehicle_model_path = "models/Models/v55s.pt"
vehicle_yaml_path = "yaml/vehicle_data.yaml"

# Load YOLO models
landslide_model = YOLO(landslide_model_path)
vehicle_model = YOLO(vehicle_model_path)

# Load Class Names
def load_classes(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data["names"]

landslide_classes = load_classes(landslide_yaml_path)
vehicle_classes = load_classes(vehicle_yaml_path)

# Video input
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = 3 * frame_rate
frame_count = 0
vehicle_counts = deque([{}] * 5, maxlen=5)
last_vehicle_count = 0
detection_history = deque(maxlen=10)

# GUI Dimensions
max_width, max_height = 960, 540
scale = min(max_width / frame_width, max_height / frame_height)
scaled_width = int(frame_width * scale)
scaled_height = int(frame_height * scale)

# GUI Setup
root = tk.Tk()
root.title("Real Time Landslides and Vehicle Detection")
root.geometry(f"{max_width+20}x{max_height+160}")
root.configure(bg="#F0F0F0")

style = ttk.Style()
style.configure("TLabel", font=("Arial", 12))
style.configure("Title.TLabel", font=("Arial", 18, "bold"), foreground="#333333")
style.configure("Error.TLabel", font=("Arial", 12), foreground="red")
style.configure("Success.TLabel", font=("Arial", 12), foreground="green")
style.configure("LargeBlue.TButton", background="#007BFF", foreground="#000000", font=("Arial", 14, "bold"), padding=10)

title_label = ttk.Label(root, text="Real Time Landslides and Vehicle Detection", style="Title.TLabel")
title_label.pack(pady=10)

video_frame = tk.Frame(root, bg="#FFFFFF", width=scaled_width, height=scaled_height)
video_frame.pack(pady=10)
video_frame.pack_propagate(False)
video_label = tk.Label(video_frame, bg="#000000")
video_label.pack(expand=True, fill="both")

status_label = ttk.Label(root, text="Status: Waiting", style="TLabel")
status_label.pack(pady=5)

vehicle_count_label = ttk.Label(root, text="Vehicles in Frame: 0", style="TLabel")
vehicle_count_label.pack(pady=5)

def format_vehicle_counts():
    total_counts = {}
    for frame_count in vehicle_counts:
        for vehicle, count in frame_count.items():
            total_counts[vehicle] = total_counts.get(vehicle, 0) + count

    formatted_output = ", ".join(
        f"{count} {vehicle}s" if count > 1 else f"{count} {vehicle}"
        for vehicle, count in total_counts.items() if count > 0
    )
    return formatted_output if formatted_output else "No vehicles detected"

def process_frame():
    global frame_count, last_vehicle_count

    ret, frame = cap.read()
    if not ret:
        cap.release()
        status_label.config(text="Processing Done ✅", style="Success.TLabel")
        return

    frame_data = []
    current_vehicle_count = {}

    if frame_count % frame_interval == 0:
        landslide_results = landslide_model(frame, imgsz=640, conf=0.5)
        landslide_detected = any(
            int(box.cls[0]) == 1 for result in landslide_results for box in result.boxes
        )

        for result in landslide_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label_text = landslide_classes[int(box.cls[0])]
                frame_data.append((x1, y1, x2, y2, label_text, (0, 0, 255)))  # Red box

        vehicle_results = vehicle_model(frame, imgsz=640, conf=0.5)
        for result in vehicle_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_label = vehicle_classes[int(box.cls[0])]
                frame_data.append((x1, y1, x2, y2, vehicle_label, (255, 0, 0)))  # Blue box
                current_vehicle_count[vehicle_label] = current_vehicle_count.get(vehicle_label, 0) + 1

        last_vehicle_count = sum(current_vehicle_count.values()) if current_vehicle_count else last_vehicle_count
        vehicle_counts.append(current_vehicle_count)

        if landslide_detected:
            status_label.config(text="\U0001F6A8 Landslide Detected!", style="Error.TLabel")

    detection_history.append(frame_data)

    for past_detections in detection_history:
        for x1, y1, x2, y2, label_text, color in past_detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    vehicle_count_label.config(text=f"Vehicles in Frame: {last_vehicle_count}")

    frame = cv2.resize(frame, (scaled_width, scaled_height))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    frame_count += 1
    root.after(30, process_frame)

start_button = ttk.Button(root, text="Start Processing", command=process_frame, style="LargeBlue.TButton")
start_button.pack(pady=10)

root.mainloop()
