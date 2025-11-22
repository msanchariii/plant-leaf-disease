import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision
from torchvision import transforms
import timm
import json

# ---------------- Load class names ----------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load ResNet50 ----------------
def load_model_resnet():
    model = torchvision.models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load("resnet50.pth", map_location=device))
    return model.to(device).eval()

# ---------------- Load EfficientNet-B0 (timm) ----------------
def load_model_efficientnet():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(class_names))
    model.load_state_dict(torch.load("b0.pth", map_location=device))
    return model.to(device).eval()

# Load both models
resnet = load_model_resnet()
try:
    effnet = load_model_efficientnet()
    print("✅ EfficientNet-B0 (timm) loaded.")
except Exception as e:
    print("❌ EfficientNet load failed:", e)
    effnet = None

# ---------------- Preprocessing ----------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- Prediction ----------------
def predict_ensemble(img):
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        r_prob = F.softmax(resnet(x), dim=1)

        if effnet is not None:
            e_prob = F.softmax(effnet(x), dim=1)
            ens_prob = (r_prob + e_prob) / 2
        else:
            e_prob = None
            ens_prob = r_prob

    def decode(prob):
        conf, idx = torch.max(prob, dim=1)
        return class_names[idx.item()], conf.item() * 100

    res = decode(r_prob)
    eff = decode(e_prob) if e_prob is not None else ("Not available", 0.0)
    ens = decode(ens_prob)

    return res, eff, ens

# ---------------- UI ----------------
st.title("🍅 Tomato Leaf Disease Detector")
st.write("Upload a tomato leaf image to identify disease using AI Ensemble")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Leaf", width="stretch")

    res, eff, ens = predict_ensemble(img)

    st.markdown("### 🔍 Prediction Results")
    if effnet is not None:
        table = {
            "Model": ["ResNet50", "EfficientNet-B0", "Ensemble (Final)"],
            "Prediction": [res[0], eff[0], ens[0]],
            "Confidence (%)": [f"{res[1]:.2f}", f"{eff[1]:.2f}", f"{ens[1]:.2f}"],
        }
    else:
        table = {
            "Model": ["ResNet50", "Ensemble (Final)"],
            "Prediction": [res[0], ens[0]],
            "Confidence (%)": [f"{res[1]:.2f}", f"{ens[1]:.2f}"],
        }

    st.table(table)
