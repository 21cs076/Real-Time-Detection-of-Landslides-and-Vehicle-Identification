import cv2
import numpy as np
import yaml
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
from twilio.rest import Client
from collections import deque
from sentinelhub import WmsRequest, MimeType, CRS, BBox, SHConfig
import datetime

# ====================== CONFIGURATION ======================
# Twilio Credentials
ACCOUNT_SID = " "
AUTH_TOKEN = " "
TWILIO_PHONE_NUMBER = " "
TO_PHONE_NUMBER = " "

# Sentinel Hub (Replace with your Instance ID)
INSTANCE_ID = " "
AREA_BBOX = [ , , , ]  #bounding box
RESOLUTION = 1024  # Image width/height in pixels

# YOLO Model Paths
landslide_model_path = "models/Models/l67s.pt"
landslide_yaml_path = "yaml/landslide_data.yaml"
vehicle_model_path = "models/Models/v55s.pt"
vehicle_yaml_path = "yaml/vehicle_data.yaml"

# GUI Settings
MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT = 960, 540
REFRESH_INTERVAL = 300000  # 5 minutes (300000ms)

# ====================== INITIALIZATION ======================
# Configure Sentinel Hub
config = SHConfig()
config.instance_id = INSTANCE_ID
config.save()

# Load YOLO Models
landslide_model = YOLO(landslide_model_path)
vehicle_model = YOLO(vehicle_model_path)

# Load Class Names
def load_classes(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data["names"]

landslide_classes = load_classes(landslide_yaml_path)
vehicle_classes = load_classes(vehicle_yaml_path)

# Detection Tracking
vehicle_counts = deque([{}] * 5, maxlen=5)
sms_sent = False
last_vehicle_count = 0
detection_history = deque(maxlen=10)
last_image_time = None

# ====================== SATELLITE IMAGE HANDLING ======================
def get_satellite_image():
    """Fetches latest satellite image from Sentinel Hub"""
    global last_image_time
    
    bbox = BBox(AREA_BBOX, crs=CRS.WGS84)
    request = WmsRequest(
        layer='FALSE-COLOR-URBAN',  # Best for geological features
        bbox=bbox,
        time='latest',
        width=RESOLUTION,
        height=RESOLUTION,
        image_format=MimeType.PNG,
        config=config
    )
    image_data = request.get_data()
    last_image_time = datetime.datetime.now()
    return cv2.cvtColor(image_data[0], cv2.COLOR_RGB2BGR)

# ====================== GUI SETUP ======================
root = tk.Tk()
root.title("Satellite Landslide & Vehicle Detection")
root.geometry(f"{MAX_DISPLAY_WIDTH+20}x{MAX_DISPLAY_HEIGHT+180}")
root.configure(bg="#F0F0F0")

style = ttk.Style()
style.configure("TLabel", font=("Arial", 12))
style.configure("Title.TLabel", font=("Arial", 18, "bold"), foreground="#333333")
style.configure("Error.TLabel", font=("Arial", 12), foreground="red")
style.configure("Success.TLabel", font=("Arial", 12), foreground="green")
style.configure("LargeBlue.TButton", background="#007BFF", foreground="#000000", 
                font=("Arial", 14, "bold"), padding=10)

# GUI Components
title_label = ttk.Label(root, text="Real-Time Satellite Monitoring System", style="Title.TLabel")
title_label.pack(pady=10)

video_frame = tk.Frame(root, bg="#FFFFFF", width=MAX_DISPLAY_WIDTH, height=MAX_DISPLAY_HEIGHT)
video_frame.pack(pady=10)
video_frame.pack_propagate(False)
video_label = tk.Label(video_frame, bg="#000000")
video_label.pack(expand=True, fill="both")

status_label = ttk.Label(root, text="Status: Waiting for first image...", style="TLabel")
status_label.pack(pady=5)

vehicle_count_label = ttk.Label(root, text="Vehicles in Area: 0", style="TLabel")
vehicle_count_label.pack(pady=5)

image_time_label = ttk.Label(root, text="Last Image: Never", style="TLabel")
image_time_label.pack(pady=5)

sms_status_label = ttk.Label(root, text="SMS Status: Not Sent", style="TLabel")
sms_status_label.pack(pady=5)

# ====================== CORE FUNCTIONALITY ======================
def format_vehicle_counts():
    """Formats vehicle counts from last 5 detections"""
    total_counts = {}
    for frame_count in vehicle_counts:
        for vehicle, count in frame_count.items():
            total_counts[vehicle] = total_counts.get(vehicle, 0) + count

    return ", ".join(
        f"{count} {vehicle}s" if count > 1 else f"{count} {vehicle}"
        for vehicle, count in total_counts.items() if count > 0
    ) or "No vehicles detected"

def send_sms(message):
    """Sends SMS alert via Twilio"""
    global sms_sent
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        msg = client.messages.create(
            body=message, from_=TWILIO_PHONE_NUMBER, to=TO_PHONE_NUMBER
        )
        sms_status_label.config(text="SMS Status: Sent Successfully", style="Success.TLabel")
        sms_sent = True
        return True
    except Exception as e:
        print(f"SMS Error: {e}")
        sms_status_label.config(text=f"SMS Failed: {str(e)}", style="Error.TLabel")
        return False

def process_satellite_image():
    """Main processing pipeline for satellite images"""
    global last_vehicle_count, sms_sent, last_image_time
    
    try:
        status_label.config(text="Status: Downloading satellite image...")
        root.update()
        
        # 1. Get fresh satellite image
        frame = get_satellite_image()
        image_time_label.config(text=f"Last Image: {last_image_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 2. Process detections
        frame_data = []
        current_vehicle_count = {}
        
        # Landslide detection (using SWIR/NIR bands)
        landslide_results = landslide_model(frame, imgsz=640, conf=0.5)
        landslide_detected = any(
            int(box.cls[0]) == 1 for result in landslide_results for box in result.boxes
        )
        
        # Vehicle detection (using RGB bands)
        vehicle_results = vehicle_model(frame, imgsz=640, conf=0.5)
        
        # Process results
        for result in landslide_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = landslide_classes[int(box.cls[0])]
                frame_data.append((x1, y1, x2, y2, label, (0, 0, 255)))  # Red
        
        for result in vehicle_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = vehicle_classes[int(box.cls[0])]
                frame_data.append((x1, y1, x2, y2, label, (255, 0, 0)))  # Blue
                current_vehicle_count[label] = current_vehicle_count.get(label, 0) + 1
        
        # Update tracking
        last_vehicle_count = sum(current_vehicle_count.values()) if current_vehicle_count else last_vehicle_count
        vehicle_counts.append(current_vehicle_count)
        detection_history.append(frame_data)
        
        # 3. Draw detections
        display_frame = frame.copy()
        for detections in detection_history:
            for x1, y1, x2, y2, label, color in detections:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 4. Trigger alerts
        if landslide_detected and not sms_sent:
            status_label.config(text="\U0001F6A8 Landslide Detected!", style="Error.TLabel")
            vehicle_summary = format_vehicle_counts()
            message = (f"ALERT: Landslide detected at coordinates {AREA_BBOX}.\n"
                      f"Potential vehicles affected: {vehicle_summary}.\n"
                      f"Timestamp: {last_image_time}")
            if send_sms(message):
                sms_sent = True
        else:
            status_label.config(text="Status: Monitoring Active", style="TLabel")
        
        # 5. Update GUI
        display_frame = cv2.resize(display_frame, (MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT))
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        vehicle_count_label.config(text=f"Vehicles in Area: {last_vehicle_count}")
        
    except Exception as e:
        status_label.config(text=f"Error: {str(e)}", style="Error.TLabel")
        print(f"Processing Error: {e}")
    
    finally:
        # Schedule next update
        root.after(REFRESH_INTERVAL, process_satellite_image)

# ====================== START BUTTON ======================
start_button = ttk.Button(
    root, 
    text="Start Satellite Monitoring", 
    command=process_satellite_image, 
    style="LargeBlue.TButton"
)
start_button.pack(pady=10)

root.mainloop()