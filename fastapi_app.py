from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
import json
from typing import List, Dict
import base64

app = FastAPI(title="Face Detection API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize face cascades
def load_face_cascades():
    cascades = {
        'frontal': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
        'profile': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml'),
        'frontal_alt': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'),
        'frontal_alt2': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'),
    }
    return cascades

face_cascades = load_face_cascades()

def detect_faces_api(image_array: np.ndarray) -> List[Dict]:
    """Detect faces and return coordinates"""
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # Detect faces using multiple cascades
    all_faces = []
    
    cascade_params = [
        (face_cascades['frontal'], 1.05, 6, (30, 30)),
        (face_cascades['frontal_alt'], 1.05, 5, (25, 25)),
        (face_cascades['frontal_alt2'], 1.05, 4, (20, 20)),
        (face_cascades['profile'], 1.05, 5, (30, 30)),
    ]
    
    for cascade, scale_factor, min_neighbors, min_size in cascade_params:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_faces.extend([(x, y, w, h) for (x, y, w, h) in faces])
    
    # Non-maximum suppression
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
    
    faces = non_max_suppression(all_faces)
    
    # Convert to list of dictionaries
    result = []
    for (x, y, w, h) in faces:
        result.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "confidence": 0.95  # Placeholder confidence score
        })
    
    return result

@app.get("/")
async def root():
    return {"message": "Face Detection API", "version": "1.0.0"}

@app.post("/detect-faces")
async def detect_faces_endpoint(file: UploadFile = File(...)):
    """Detect faces in uploaded image"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = detect_faces_api(img_array)
        
        return {
            "filename": file.filename,
            "faces_detected": len(faces),
            "faces": faces,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect-faces-with-image")
async def detect_faces_with_image(file: UploadFile = File(...)):
    """Detect faces and return annotated image"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = detect_faces_api(img_array)
        
        # Draw bounding boxes
        for face in faces:
            x, y, w, h = face["x"], face["y"], face["width"], face["height"]
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Add corner markers
            corner_size = 15
            cv2.line(img_array, (x, y), (x + corner_size, y), (0, 255, 0), 2)
            cv2.line(img_array, (x, y), (x, y + corner_size), (0, 255, 0), 2)
            
            # Add text background
            cv2.rectangle(img_array, (x, y - 30), (x + 70, y), (0, 255, 0), -1)
            cv2.putText(img_array, 'Face', (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Convert back to bytes
        _, img_encoded = cv2.imencode('.jpg', img_array)
        img_bytes = img_encoded.tobytes()
        
        return StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"attachment; filename=detected_{file.filename}",
                "X-Faces-Detected": str(len(faces))
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "face-detection-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)