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

# Ensemble weights (you can tune this)
RESNET_WEIGHT = 0.4
EFFNET_WEIGHT = 0.6

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

# ---------------- Helper: Top-3 ----------------
def decode_top3(prob):
    top_probs, top_idxs = torch.topk(prob, 3)

    results = []
    for i in range(3):
        class_name = class_names[top_idxs[0][i].item()]
        confidence = top_probs[0][i].item() * 100
        results.append((class_name, confidence))

    return results

# ---------------- Prediction ----------------
def predict_ensemble(img):
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        r_prob = F.softmax(resnet(x), dim=1)

        if effnet is not None:
            e_prob = F.softmax(effnet(x), dim=1)

            # 🔥 Weighted Ensemble (better than simple average)
            ens_prob = (RESNET_WEIGHT * r_prob + EFFNET_WEIGHT * e_prob) / (
                RESNET_WEIGHT + EFFNET_WEIGHT
            )
        else:
            e_prob = None
            ens_prob = r_prob

    res = decode_top3(r_prob)
    eff = decode_top3(e_prob) if e_prob is not None else []
    ens = decode_top3(ens_prob)

    return res, eff, ens


# for unknown objects
def is_valid_prediction(top3, conf_threshold=60, gap_threshold=10):
    top1_conf = top3[0][1]
    top2_conf = top3[1][1]

    if top1_conf < conf_threshold:
        return False

    if (top1_conf - top2_conf) < gap_threshold:
        return False

    return True

# ---------------- UI ----------------
st.title("🍅 Tomato Leaf Disease Detection")
st.write(
    "Upload a tomato leaf image to identify possible diseases using an ensemble of deep learning models."
)

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width="stretch")

    res, eff, ens = predict_ensemble(img)

    # ---------------- Format Table ----------------
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
                "Ensemble (Final Prediction)"
            ],
            "Top-3 Predictions": [
                format_top3(res),
                format_top3(eff),
                format_top3(ens)
            ]
        }
    else:
        table = {
            "Model": [
                "ResNet50",
                "Ensemble (Final Prediction)"
            ],
            "Top-3 Predictions": [
                format_top3(res),
                format_top3(ens)
            ]
        }

    st.table(table)

    # ---------------- Highlight Final Prediction ----------------
    st.markdown("### Final Prediction")
    # final_class, final_conf = ens[0]
    # st.success(f"{final_class} ({final_conf:.2f}% confidence)")
    is_valid = is_valid_prediction(ens)

    if not is_valid:
        st.error("The uploaded image does not appear to be a tomato leaf or the model is uncertain.")
    else:
        final_class, final_conf = ens[0]
        st.success(f"{final_class} ({final_conf:.2f}% confidence)")

    # ---------------- Note ----------------
    st.info(
        "Note: The system is trained specifically on tomato leaf diseases. "
        "Predictions for non-tomato images may be unreliable."
    )