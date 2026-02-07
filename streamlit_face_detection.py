import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Face Detection App",
    page_icon="🎯",
    layout="wide"
)

# App title and description
st.title("🎯 Real-Time Face Detection")
st.markdown("""
### Upload an image or use your webcam to detect faces!
This app uses OpenCV's Haar Cascade classifier for face detection.
""")

# Initialize multiple face cascades for better detection
@st.cache_resource
def load_face_cascades():
    cascades = {
        'frontal': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
        'profile': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml'),
        'frontal_alt': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'),
        'frontal_alt2': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'),
    }
    return cascades

face_cascades = load_face_cascades()

def detect_faces(image):
    """Detect faces in an image using multiple cascades for better accuracy"""
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    
    # Detect faces using multiple cascades
    all_faces = []
    
    # Try different cascade classifiers with optimized parameters
    cascade_params = [
        (face_cascades['frontal'], 1.05, 6, (30, 30)),  # Default frontal
        (face_cascades['frontal_alt'], 1.05, 5, (25, 25)),  # Alternative frontal
        (face_cascades['frontal_alt2'], 1.05, 4, (20, 20)),  # Second alternative
        (face_cascades['profile'], 1.05, 5, (30, 30)),  # Profile faces
    ]
    
    for cascade, scale_factor, min_neighbors, min_size in cascade_params:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_faces.extend(faces)
    
    # Remove duplicate faces using non-maximum suppression
    def non_max_suppression(boxes, overlap_thresh=0.3):
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        area = boxes[:, 2] * boxes[:, 3]
        idxs = np.argsort(y2)
        
        pick = []
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
        
        return boxes[pick].astype("int")
    
    # Apply non-maximum suppression to remove duplicates
    faces = non_max_suppression(all_faces)
    
    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        # Draw bounding box
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # Draw corner markers
        corner_size = 15
        cv2.line(img_array, (x, y), (x+corner_size, y), (0, 255, 0), 2)
        cv2.line(img_array, (x, y), (x, y+corner_size), (0, 255, 0), 2)
        cv2.line(img_array, (x+w, y), (x+w-corner_size, y), (0, 255, 0), 2)
        cv2.line(img_array, (x+w, y), (x+w, y+corner_size), (0, 255, 0), 2)
        cv2.line(img_array, (x, y+h), (x+corner_size, y+h), (0, 255, 0), 2)
        cv2.line(img_array, (x, y+h), (x, y+h-corner_size), (0, 255, 0), 2)
        cv2.line(img_array, (x+w, y+h), (x+w-corner_size, y+h), (0, 255, 0), 2)
        cv2.line(img_array, (x+w, y+h), (x+w, y+h-corner_size), (0, 255, 0), 2)
        
        # Draw label background and text
        cv2.rectangle(img_array, (x, y-30), (x+70, y), (0, 255, 0), -1)
        cv2.putText(img_array, 'Face', (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Convert back to RGB for display
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(img_array), len(faces)

# Sidebar for options
st.sidebar.header("Options")

# Choose input method
input_method = st.sidebar.radio(
    "Choose input method:",
    ["Upload Image", "Webcam"],
    index=0
)

if input_method == "Upload Image":
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image...",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        
        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        # Detect faces
        processed_image, face_count = detect_faces(image)
        
        # Display processed image
        with col2:
            st.subheader("Processed Image")
            st.image(processed_image, use_column_width=True)
            st.success(f"🎯 Detected {face_count} face(s)!")
        
        # Download button for processed image
        if face_count > 0:
            # Convert to bytes for download
            buffered = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            processed_image.save(buffered, format="JPEG")
            buffered.seek(0)
            
            st.sidebar.download_button(
                label="📥 Download Processed Image",
                data=buffered.read(),
                file_name="face_detection_result.jpg",
                mime="image/jpeg"
            )
            buffered.close()
            os.unlink(buffered.name)

else:  # Webcam mode
    st.sidebar.markdown("### Webcam Settings")
    
    # Webcam controls
    if st.sidebar.button("🎥 Start Webcam", key="start_webcam"):
        st.session_state.webcam_active = True
    
    if st.sidebar.button("⏹️ Stop Webcam", key="stop_webcam"):
        if 'webcam_active' in st.session_state:
            st.session_state.webcam_active = False
    
    # Initialize webcam state
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    
    if st.session_state.webcam_active:
        st.sidebar.success("Webcam is active!")
        
        # Webcam feed
        webcam_placeholder = st.empty()
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            st.error("❌ Could not access webcam. Please check your camera settings.")
        else:
            st.info("🔴 Live Webcam Feed - Press 'Stop Webcam' to end")
            
            # Process webcam frames
            while st.session_state.webcam_active and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Convert frame to PIL Image for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Detect faces
                processed_image, face_count = detect_faces(pil_image)
                
                # Display processed frame
                webcam_placeholder.image(
                    processed_image, 
                    caption=f"Detected {face_count} face(s)",
                    use_column_width=True
                )
                
                # Small delay to prevent high CPU usage
                import time
                time.sleep(0.03)
            
            # Release webcam when done
            cap.release()
            st.sidebar.info("Webcam stopped")
    else:
        st.info("👆 Click 'Start Webcam' to begin live face detection")

# Add some styling
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This face detection app uses OpenCV's Haar Cascade classifier 
to detect faces in images with real-time processing.
""")

# Footer
st.markdown("---")
st.caption("Built with Streamlit & OpenCV | Face Detection App")