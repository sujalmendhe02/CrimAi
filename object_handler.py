import csv
import os
from object_detection import ObjectDetector

def save_object_detection_data(object_name, detection_time, video_name=""):
    """Save object detection data to CSV"""
    field = ['Object Name', 'Detection Time', 'Video/Image Source']
    filename = "detected_objects.csv"
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(field)
        csvwriter.writerow([object_name, detection_time, video_name])

def get_object_detection_history():
    """Get object detection history from CSV"""
    try:
        with open("detected_objects.csv", 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip header
            return list(csvreader)
    except FileNotFoundError:
        return []

def register_new_object(object_data):
    """Register a new criminal object"""
    object_name = object_data['Object Name']
    description = object_data['Description']
    danger_level = object_data['Danger Level']
    
    # Save object metadata
    field = ['Object Name', 'Description', 'Danger Level', 'Registration Date']
    filename = "criminal_objects.csv"
    
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(field)
        
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csvwriter.writerow([object_name, description, danger_level, current_date])

def get_registered_objects():
    """Get list of registered criminal objects"""
    try:
        with open("criminal_objects.csv", 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip header
            return list(csvreader)
    except FileNotFoundError:
        return []