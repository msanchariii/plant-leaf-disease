import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision
from torchvision import transforms
import timm
import json
import os

# ---------------- Config ----------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

RESNET_PATH = os.path.join(MODEL_DIR, "resnet50.pth")
EFFNET_PATH = os.path.join(MODEL_DIR, "b0.pth")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 Weights based on performance
RESNET_WEIGHT = 0.3
EFFNET_WEIGHT = 0.7

CONF_THRESHOLD = 0.60   # 60% minimum confidence
GAP_THRESHOLD = 0.10    # 10% gap required

# ---------------- Load class names ----------------
with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

# ---------------- Load Models ----------------
@st.cache_resource
def load_model_resnet():
    model = torchvision.models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()

@st.cache_resource
def load_model_efficientnet():
    try:
        model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=len(class_names)
        )
        model.load_state_dict(torch.load(EFFNET_PATH, map_location=DEVICE))
        return model.to(DEVICE).eval()
    except Exception as e:
        print("EfficientNet load failed:", e)
        return None

resnet = load_model_resnet()
effnet = load_model_efficientnet()

# ---------------- Preprocessing ----------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------------- Top-3 Helper ----------------
def decode_top3(prob):
    top_probs, top_idxs = torch.topk(prob, 3)

    results = []
    for i in range(3):
        class_name = class_names[top_idxs[0][i].item()]
        confidence = top_probs[0][i].item() * 100
        results.append((class_name, confidence))

    return results

# ---------------- Ensemble Logic ----------------
def weighted_ensemble(r_prob, e_prob):
    # Weighted voting
    ens = (RESNET_WEIGHT * r_prob + EFFNET_WEIGHT * e_prob)

    # 🔥 Agreement Boost
    r_top = torch.argmax(r_prob)
    e_top = torch.argmax(e_prob)

    if r_top == e_top:
        ens = ens * 1.1  # boost confidence if both agree

    # Normalize again (important after scaling)
    ens = ens / ens.sum(dim=1, keepdim=True)

    return ens

# ---------------- Prediction ----------------
def predict_ensemble(img):
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        r_prob = F.softmax(resnet(x), dim=1)

        if effnet is not None:
            e_prob = F.softmax(effnet(x), dim=1)
            ens_prob = weighted_ensemble(r_prob, e_prob)
        else:
            e_prob = None
            ens_prob = r_prob

    res = decode_top3(r_prob)
    eff = decode_top3(e_prob) if e_prob is not None else []
    ens = decode_top3(ens_prob)

    return res, eff, ens, ens_prob

# ---------------- Unknown Detection ----------------
def is_valid_prediction(ens_prob):
    top_probs, _ = torch.topk(ens_prob, 2)

    top1 = top_probs[0][0].item()
    top2 = top_probs[0][1].item()

    if top1 < CONF_THRESHOLD:
        return False

    if (top1 - top2) < GAP_THRESHOLD:
        return False

    return True

# ---------------- UI ----------------
st.title("🍅 Tomato Leaf Disease Detection")
st.write(
    "Upload a tomato leaf image to identify diseases using an ensemble of deep learning models."
)

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width="stretch")

    res, eff, ens, ens_prob = predict_ensemble(img)

    # ---------------- Table ----------------
    def format_top3(results):
        return "\n".join([
            f"{i+1}. {cls} ({conf:.2f}%)"
            for i, (cls, conf) in enumerate(results)
        ])

    st.markdown("### 🔍 Top-3 Prediction Results")

    if effnet is not None:
        table = {
            "Model": [
                "ResNet50",
                "EfficientNet-B0",
                "Ensemble (Final)"
            ],
            "Top-3 Predictions": [
                format_top3(res),
                format_top3(eff),
                format_top3(ens)
            ]
        }
    else:
        table = {
            "Model": ["ResNet50", "Ensemble (Final)"],
            "Top-3 Predictions": [
                format_top3(res),
                format_top3(ens)
            ]
        }

    st.table(table)

    # ---------------- Final Prediction ----------------
    st.markdown("### Final Prediction")

    if not is_valid_prediction(ens_prob):
        st.error("The image is not a tomato leaf or prediction is uncertain.")
    else:
        final_class, final_conf = ens[0]
        st.success(f"{final_class} ({final_conf:.2f}% confidence)")

    # ---------------- Info ----------------
    st.info(
        "This system uses weighted ensemble learning with confidence thresholding "
        "to improve reliability and reduce incorrect predictions."
    )