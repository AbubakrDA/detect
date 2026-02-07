import cv2
import numpy as np
import time

# Load pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    """Detect faces in a frame and draw bounding boxes"""
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Draw enhanced bounding boxes around detected faces
    for (x, y, w, h) in faces:
        # Draw thicker bounding box with rounded corners
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # Draw corner markers for style
        corner_size = 15
        # Top-left corner
        cv2.line(frame, (x, y), (x+corner_size, y), (0, 255, 0), 2)
        cv2.line(frame, (x, y), (x, y+corner_size), (0, 255, 0), 2)
        # Top-right corner
        cv2.line(frame, (x+w, y), (x+w-corner_size, y), (0, 255, 0), 2)
        cv2.line(frame, (x+w, y), (x+w, y+corner_size), (0, 255, 0), 2)
        # Bottom-left corner
        cv2.line(frame, (x, y+h), (x+corner_size, y+h), (0, 255, 0), 2)
        cv2.line(frame, (x, y+h), (x, y+h-corner_size), (0, 255, 0), 2)
        # Bottom-right corner
        cv2.line(frame, (x+w, y+h), (x+w-corner_size, y+h), (0, 255, 0), 2)
        cv2.line(frame, (x+w, y+h), (x+w, y+h-corner_size), (0, 255, 0), 2)
        
        # Draw background for text
        cv2.rectangle(frame, (x, y-30), (x+70, y), (0, 255, 0), -1)
        cv2.putText(frame, f'Face', (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame, len(faces)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set higher resolution for better detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("🎯 Enhanced Face Detection App Started!")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("Press 'c' to change detection mode")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Detect faces
        processed_frame, face_count = detect_faces(frame)
        
        # Display face count
        cv2.putText(processed_frame, f'Faces: {face_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('Face Detection', processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('s'):  # Save frame
            cv2.imwrite(f'face_detection_{int(time.time())}.jpg', processed_frame)
            print("Frame saved!")
        elif key == ord('c'):  # Change detection mode
            global face_cascade
            if face_cascade == cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'):
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
                print("Switched to profile face detection")
            else:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                print("Switched to frontal face detection")
    
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Face Detection App Closed")

if __name__ == "__main__":
    main()