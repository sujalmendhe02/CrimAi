import cv2
import numpy as np
import os
import pickle
from sklearn.cluster import DBSCAN
import glob

class ObjectDetector:
    def __init__(self):
        self.object_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        self.object_database = {}
        self.load_object_database()
    
    def extract_features(self, image):
        """Extract SIFT features from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def add_object_to_database(self, object_name, image_paths):
        """Add a new criminal object to the database"""
        all_descriptors = []
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                keypoints, descriptors = self.extract_features(img)
                if descriptors is not None:
                    all_descriptors.append(descriptors)
        
        if all_descriptors:
            # Combine all descriptors for this object
            combined_descriptors = np.vstack(all_descriptors)
            self.object_database[object_name] = combined_descriptors
            self.save_object_database()
            return True
        return False
    
    def detect_objects_in_image(self, image):
        """Detect criminal objects in a single image"""
        detected_objects = []
        keypoints, descriptors = self.extract_features(image)
        
        if descriptors is None:
            return detected_objects
        
        for object_name, stored_descriptors in self.object_database.items():
            matches = self.matcher.knnMatch(descriptors, stored_descriptors, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # If enough good matches found, object is detected
            if len(good_matches) > 10:  # Threshold for detection
                detected_objects.append((object_name, len(good_matches)))
        
        return detected_objects
    
    def save_object_database(self):
        """Save object database to file"""
        if not os.path.exists('object_data'):
            os.makedirs('object_data')
        
        with open('object_data/object_database.pkl', 'wb') as f:
            pickle.dump(self.object_database, f)
    
    def load_object_database(self):
        """Load object database from file"""
        try:
            with open('object_data/object_database.pkl', 'rb') as f:
                self.object_database = pickle.load(f)
        except FileNotFoundError:
            self.object_database = {}
    
    def get_registered_objects(self):
        """Get list of registered criminal objects"""
        return list(self.object_database.keys())

def register_criminal_object(object_name, image_paths):
    """Register a new criminal object"""
    detector = ObjectDetector()
    return detector.add_object_to_database(object_name, image_paths)

def detect_objects_in_frame(frame):
    """Detect criminal objects in a video frame"""
    detector = ObjectDetector()
    return detector.detect_objects_in_image(frame)