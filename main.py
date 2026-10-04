import streamlit as st
import torch
import torch.nn as nn
import cv2
import pandas as pd
import datetime
import numpy as np
from PIL import Image
from torchvision import models, transforms
from database import create, fill, display

st.set_page_config(page_title="Deepfake Classification", page_icon="🖼️", layout="centered")
create()

st.title("Deepfake Image Classification System")
st.info("v 1.1")

IMG_SIZE = 224
classes = ["Fake", "Real"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


@st.cache_resource
def load_model():
    model = build_model()
    state_dict = torch.load("weights/best_real_fake_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_face_detector():
    detector = cv2.FaceDetectorYN.create("weights/face_detection_yunet_2026may.onnx", "", (320, 320), 0.6, 0.3, 5000)
    return detector


model = load_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.224, 0.224, 0.225]),
])


def detect_face(image):
    img = np.array(image)
    detector = load_face_detector()
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is None:
        return None
    face = max(faces, key=lambda f: f[2] * f[3])
    x, y, fw, fh = face[:4]
    x = int(x)
    y = int(y)
    fw = int(fw)
    fh = int(fh)
    padding = 0.25
    pad_w = int(fw * padding)
    pad_h = int(fh * padding)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + fw + pad_w)
    y2 = min(h, y + fh + pad_h)
    return image.crop((x1, y1, x2, y2))


upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if upload:
    file_name = upload.name
    image = Image.open(upload).convert("RGB")
    st.image(image, caption="Uploaded image")

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            face = detect_face(image)

            if face is None:
                st.warning("No face detected")
            else:
                date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                image_tensor = transform(face).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(image_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0, predicted_class].item()

                predicted_label = classes[predicted_class]

                if predicted_label == "Real":
                    st.success(f"Predicted class: {predicted_label}")
                else:
                    st.error(f"Predicted class: {predicted_label}")

                st.info(f"Confidence: {confidence * 100:.2f}%")
                st.write(f"Fake: {probabilities[0][0].item() * 100:.2f}%")
                st.write(f"Real: {probabilities[0][1].item() * 100:.2f}%")

                fill(file_name, date, predicted_label, confidence)

with st.sidebar:
    st.title("System History")
    data = display()
    if data:
        df = pd.DataFrame(data, columns=["ID", "File Name", "Date", "Prediction", "Confidence"])
        df["Confidence"] = (df["Confidence"] * 100).round(2)
        st.dataframe(df, hide_index=True, use_container_width=True)