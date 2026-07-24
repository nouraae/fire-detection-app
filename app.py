import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image

# =========================================================
# KONFIGURASI -- harus SAMA PERSIS dengan waktu training
# =========================================================
CLASS_NAMES = ["fire", "normal", "smoke"]  # urutan sesuai output notebook (cek "Classes: [...]")
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = "model_resnet50_augmented.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# LOAD MODEL -- di-cache supaya cuma di-load sekali, bukan tiap kali user upload gambar
# =========================================================
@st.cache_resource
def load_model():
    model = torchvision.models.resnet50(weights=None)  # weights=None, karena mau load bobot sendiri
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# =========================================================
# PREPROCESSING -- harus SAMA PERSIS dengan waktu training (base_tf, tanpa augmentasi)
# =========================================================
def preprocess_image(image: Image.Image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return transform(image).unsqueeze(0)  # tambah batch dimension


# =========================================================
# UI STREAMLIT
# =========================================================
st.set_page_config(page_title="Fire/Smoke/Normal Classifier", page_icon="🔥")

st.title("🔥 Fire / Smoke / Normal Classifier")
st.write("Upload gambar untuk diklasifikasikan sebagai **fire**, **smoke**, atau **normal**.")

model = load_model()

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang di-upload", use_container_width=True)

    with st.spinner("Menganalisis gambar..."):
        input_tensor = preprocess_image(image).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()

    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item() * 100

    st.success(f"**Prediksi: {pred_class.upper()}** ({confidence:.1f}% confidence)")

    # Tampilkan probabilitas semua kelas
    st.write("Detail probabilitas per kelas:")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"- {class_name}: {probs[i].item()*100:.1f}%")
        st.progress(probs[i].item())
