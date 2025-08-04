#                                                                     **Computer-Vision (CV)**

**Beginner Level** - **Intermediate Level:** - **Advanced Level**

=========================Beginner Level==================================

Welcome to **Computer Vision (CV)**  This guide will take you from **absolute beginner** to building your first CV projects.  

---

## **Step 1: Learn the Basics**  

### **1️ Understand Images & Pixels**  
- An **image** is a grid of pixels (e.g., 640x480 resolution = 640 columns × 480 rows).  
- Each pixel has **color values** (RGB: Red, Green, Blue).  
- **Grayscale** = single value (0=black, 255=white).  

🔹 **Try this:**  
```python
import cv2  
img = cv2.imread("dog.png")  
print(img.shape)  # (height, width, channels)  
```
<img width="979" height="37" alt="image" src="https://github.com/user-attachments/assets/86716b81-dd8c-4535-ac80-d86acf8b6bc8" />

### **2️ Install Key Tools**  
- **Python** (3.8+)  
- **OpenCV** (`pip install opencv-python`)  
- **Matplotlib** (`pip install matplotlib`)  

---

## ** Step 2: Basic Image Processing**  

### **1️ Load & Display Images**  
```python
import cv2  
img = cv2.imread("dog.png")  
cv2.imshow("Dog Image", img)  
cv2.waitKey(0)  # Press any key to close  
```
<img width="894" height="742" alt="image" src="https://github.com/user-attachments/assets/6a40b76e-767e-4e2f-ab25-3acad6c3cee5" />

### **2️ Convert to Grayscale**  
```python
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
cv2.imshow("Gray Dog", gray_img)  
cv2.waitKey(0)  
```
<img width="898" height="730" alt="image" src="https://github.com/user-attachments/assets/9e65dff3-361a-4608-843f-e2da74b67b21" />

### **3️ Edge Detection (Canny Edge)**  
```python
edges = cv2.Canny(gray_img, 100, 200)  # Min & Max thresholds  
cv2.imshow("Edges", edges)  
cv2.waitKey(0)  
```
<img width="901" height="744" alt="image" src="https://github.com/user-attachments/assets/252a2e2c-91e5-41f0-9b36-b2d6c5efd511" />

### **4️ Face Detection (Haar Cascades)**  
```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)  

for (x, y, w, h) in faces:  
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  

cv2.imshow("Faces Detected", img)  
cv2.waitKey(0)  
```
<img width="900" height="750" alt="image" src="https://github.com/user-attachments/assets/9122d8cf-a6f6-4805-8aba-339ed23c1caa" />

---

## ** Step 3: Simple Machine Learning for CV**  

### **1️ Train a Digit Classifier (MNIST Dataset)**  
```python
from sklearn.datasets import load_digits  
from sklearn.svm import SVC  

digits = load_digits()  
X, y = digits.data, digits.target  

model = SVC()  
model.fit(X, y)  

# Predict a digit  
prediction = model.predict([X[0]])  
print("Predicted:", prediction)  
```

### **2️ Handwritten Digit Recognition (OpenCV + SVM)**  
- Draw digits on a whiteboard → Detect & Predict.  

---

## ** Step 4: Beginner Projects**  

### **🔹 Project 1: Live Face Detection**  
 **Goal:** Use your webcam to detect faces in real-time.  
```python
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class FaceDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return img

st.title("Live Face Detection with Streamlit")

webrtc_streamer(
    key="face-detect",
    video_transformer_factory=FaceDetectionTransformer,
    media_stream_constraints={"video": True, "audio": False}
)
```
<img width="1912" height="973" alt="Screenshot 2025-08-04 182044" src="https://github.com/user-attachments/assets/0efb1bfb-7436-4571-8e74-8fa8fec0b616" />

### **🔹 Project 2: Image Filters App**  
**Goal:** Apply filters (blur, edge, grayscale) to an image. 
```python
import cv2
import numpy as np
import streamlit as st

st.title("Image Filters App")
st.write("Upload an image and apply filters (blur, edge, grayscale).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Original Image', use_column_width=True)

    filter_type = st.selectbox("Choose a filter", ["None", "Grayscale", "Blur", "Edge Detection"])

    if filter_type == "Grayscale":
        filtered = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        st.image(filtered, caption='Grayscale Image', use_column_width=True, channels="GRAY")
    elif filter_type == "Blur":
        filtered = cv2.GaussianBlur(img, (15, 15), 0)
        st.image(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB), caption='Blurred Image', use_column_width=True)
    elif filter_type == "Edge Detection":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        st.image(edges, caption='Edge Detection', use_column_width=True, channels="GRAY")
```
<img width="1912" height="972" alt="image" src="https://github.com/user-attachments/assets/d72d98b6-a94c-46aa-898f-91e18881d545" />
<img width="1919" height="967" alt="image" src="https://github.com/user-attachments/assets/598e221a-87cc-40e5-b8fb-b0d38ea71a69" />
<img width="1919" height="960" alt="image" src="https://github.com/user-attachments/assets/a7aea9a1-73fc-4865-8bbd-7d6c7363caaa" />
<img width="1916" height="970" alt="image" src="https://github.com/user-attachments/assets/b990cffa-10bc-4a95-902e-99592f77147c" />

### **🔹 Project 3: Simple Motion Detector**  
**Goal:** Detect movement using background subtraction.  
```python
import cv2
import numpy as np
import streamlit as st

st.title("Simple Motion Detector")
st.write("Upload a video to detect motion using background subtraction.")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = open("temp_video.mp4", "wb")
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture("temp_video.mp4")
    fgbg = cv2.createBackgroundSubtractorMOG2()

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fgmask = fgbg.apply(frame)
        # Highlight motion in red on the original frame
        motion = cv2.bitwise_and(frame, frame, mask=fgmask)
        display = cv2.addWeighted(frame, 0.7, motion, 0.7, 0)
        stframe.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
    cap.release()
```
<img width="1919" height="726" alt="image" src="https://github.com/user-attachments/assets/e492850d-f7d7-47b3-9ecd-017749e943db" />

---

## **Step 5: Next Steps**  
**Master OpenCV basics** (Geometric transforms, contours).  
**Learn NumPy for image manipulation**.  
**Move to CNN-based models** (Next: **Intermediate Level**).  

---

## **Summary: Beginner CV Roadmap**  
| **Topic**               | **What You’ll Learn**          | **Tools Used**       |  
|-------------------------|-------------------------------|----------------------|  
| **Image Basics**        | Pixels, RGB, Grayscale        | OpenCV, Matplotlib   |  
| **Image Processing**    | Filters, Edge Detection       | OpenCV               |  
| **Face Detection**      | Haar Cascades                 | OpenCV               |  
| **Simple ML for CV**    | SVM for digit classification  | scikit-learn         |  
| **Beginner Projects**   | Live face detection, filters  | OpenCV + Webcam      |  

---

### What’s Next?
- **Intermediate Level:** CNNs, YOLO, Object Detection.  
- **Advanced Level:** Transformers, GANs, Deployment.  

=========================Intermediate Level==============================


-
-

## About Me

**Vishal Kumar**
- [GitHub](https://github.com/VishalKumar-GitHub)

📫 **Follow me** on [Xing](https://www.xing.com/profile/Vishal_Kumar055381/web_profiles?expandNeffi=true) | [LinkedIn](https://www.linkedin.com/in/vishal-kumar-819585275/)
