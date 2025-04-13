## 🔍 Overview
This project implements a pothole detection system using the YOLOv5 object detection model. It can identify potholes in both static images and video streams, making it useful for road maintenance authorities, autonomous vehicles, and transportation safety applications.

## ✨ Features

- Real-time pothole detection in video streams
- Static image pothole detection
- Visual bounding boxes around detected potholes
- Confidence score display
- Pre-trained model for immediate use
- Custom training workflow for improved accuracy

---

## 📦 Requirements
- cvzone==1.5.6
- ultralytics==8.0.26
- hydra-core>=1.2.0
- matplotlib>=3.2.2
- numpy>=1.18.5
- opencv-python==4.5.4.60
- Pillow>=7.1.2
- PyYAML>=5.3.1
- requests>=2.23.0
- scipy>=1.4.1
- torch>=1.7.0
- torchvision>=0.8.1
- tqdm>=4.64.0
- filterpy==1.4.5
- scikit-image==0.19.3
- lap==0.4.0

---

## 💻 Installation

- Clone the repository:
     git clone [https://github.com/yourusername/pothole-detection.git](https://github.com/satishkanti/Real-time-Object-detection-using-YOLOv5.git)
     cd pothole-detection

- Create a virtual environment (optional but recommended):
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate

- Install the required packages:
     pip install -r requirements.txt

- Download the pre-trained model:
  The model file "pothole.pt" should be placed in the project root directory

---

## 🚀 Usage
Image Detection
- To detect potholes in a single image:
     from ultralytics import YOLO
     import cv2

##### Load the model
model = YOLO('pothole.pt')

##### Perform detection
   results = model("path/to/your/image.jpg", show=True)
   cv2.waitKey(0)
- Video Detection
   To detect potholes in a video stream:
      from ultralytics import YOLO
      import cv2
      import cvzone
      import math

##### Load the model
   cap = cv2.VideoCapture("path/to/your/video.mp4")
   model = YOLO("pothole.pt")
   classNames = ["pothole"]

while True:
    success, img = cap.read()
    if not success:
        break
        
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            
            # Draw rectangle
            cvzone.cornerRect(img, (x1, y1, w, h))
            
            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            
            # Class name
            cls = int(box.cls[0])
            
            # Display class and confidence
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', 
                              (max(0, x1), max(35, y1)), 
                              scale=1, thickness=1)
    
    cv2.imshow("Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

---

## 🧠 Model Training
The model was trained using YOLOv5 with the following configuration:

1. Base model: YOLOv5l (large)
2. Epochs: 50
2. Image size: 640x640
4. Dataset: Custom pothole dataset

To train your own model:

1. Prepare your dataset following YOLO format
2. Create a data.yaml file with dataset configuration
3. Run the training:
   yolo task=detect mode=train model=yolov5l.pt data=/path/to/data.yaml epochs=50 imgsz=640

---

## 📊 Results
Our model achieves the following performance metrics:
- mAP50: 0.806
- mAP50-95: 0.527
- Precision: 0.853
- Recall: 0.716

Training progression:
- Initial mAP50: 0.264
- Final mAP50: 0.780
- Best mAP50-95: 0.527 (Epoch 43/50)

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add some amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
