# Face Detection Application

A comprehensive face detection system with multiple interfaces including Streamlit web app, FastAPI REST API, and Docker containerization.

## 🚀 Features

- **Real-time Face Detection**: Advanced Haar Cascade with multiple classifiers
- **Web Interface**: Streamlit-based GUI with image upload and webcam support
- **REST API**: FastAPI endpoints for programmatic access
- **Docker Support**: Containerized deployment
- **High Accuracy**: Multiple cascade classifiers with non-maximum suppression
- **Cross-platform**: Works on Windows, Linux, and macOS

## 📦 Installation

### Prerequisites
- Python 3.9+
- OpenCV dependencies
- Webcam (for live detection)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd detect
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit app**
   ```bash
   streamlit run streamlit_face_detection.py
   ```

4. **Run FastAPI server**
   ```bash
   python fastapi_app.py
   ```

## 🎯 Usage

### Web Interface (Streamlit)

Access the web app at `http://localhost:8501`:

- **Upload Image**: Select an image file for face detection
- **Webcam**: Use live webcam for real-time detection
- **Download**: Get processed images with detected faces

### API Endpoints (FastAPI)

Access API docs at `http://localhost:8000/docs`:

- **POST /detect-faces**: Detect faces and return coordinates
- **POST /detect-faces-with-image**: Detect faces and return annotated image
- **GET /health**: Health check endpoint

#### Example API Usage

```bash
# Detect faces in an image
curl -X POST "http://localhost:8000/detect-faces" \
     -H "accept: application/json" \
     -F "file=@your_image.jpg"

# Get annotated image
curl -X POST "http://localhost:8000/detect-faces-with-image" \
     -H "accept: image/jpeg" \
     -F "file=@your_image.jpg" \
     --output detected_image.jpg
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t face-detection-app .

# Run the container
docker run -p 8501:8501 -p 8000:8000 face-detection-app
```

### Docker Compose

```yaml
version: '3.8'
services:
  face-detection:
    build: .
    ports:
      - "8501:8501"  # Streamlit
      - "8000:8000"  # FastAPI
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

## 🏗️ Project Structure

```
detect/
├── streamlit_face_detection.py  # Web interface
├── fastapi_app.py              # REST API
├── Dockerfile                  # Container configuration
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🔧 Technical Details

### Face Detection Algorithm

- **Multiple Cascade Classifiers**: Uses 4 different Haar Cascade models
- **Non-Maximum Suppression**: Removes duplicate detections
- **Histogram Equalization**: Improves contrast for better detection
- **Optimized Parameters**: Fine-tuned scale factors and neighbor settings

### Performance

- **Real-time Processing**: ~15-30 FPS on standard hardware
- **High Accuracy**: Detects frontal and profile faces
- **Low Resource Usage**: Efficient memory and CPU utilization

## 📊 API Response Format

### JSON Response Example

```json
{
  "filename": "image.jpg",
  "faces_detected": 3,
  "faces": [
    {
      "x": 100,
      "y": 150,
      "width": 120,
      "height": 120,
      "confidence": 0.95
    }
  ],
  "success": true
}
```

## 🚦 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest -v
```

### Code Style

```bash
# Format code
pip install black
black .
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV for face detection algorithms
- Streamlit for web interface framework
- FastAPI for REST API framework
- Docker for containerization

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review API examples

---

**Happy Face Detecting!** 🎭