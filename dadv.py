import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS
import speech_recognition as sr
import tempfile
import os
import asyncio

# OCR & Image
import easyocr
from PIL import Image
import cv2
import numpy as np

# Accuracy & Visualization
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Multi-Mode Translator", layout="centered")
st.title("🌐 Voice • Text • Camera Translator")

# -----------------------------
# LOAD SIMILARITY MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# SUPPORTED LANGUAGES
# -----------------------------
languages = {
    "English": "en",
    "Tamil": "ta",
    "Kannada": "kn",
    "Hindi": "hi",
    "Telugu": "te",
    "Malayalam": "ml",
    "Urdu": "ur"
}

easyocr_langs = {
    "English": ["en"],
    "Tamil": ["ta"],
    "Kannada": ["kn"],
    "Hindi": ["hi"],
    "Telugu": ["te"],
    "Malayalam": ["ml"],
    "Urdu": ["ur"]
}

# -----------------------------
# MODE SELECTION
# -----------------------------
mode = st.selectbox(
    "Select Mode",
    ["Text Translator", "Voice Translator", "Camera Translator"]
)

# -----------------------------
# LANGUAGE SELECTION
# -----------------------------
input_lang = st.selectbox("Select Input Language", list(languages.keys()))
target_lang = st.selectbox("Select Target Language", list(languages.keys()))

src_code = languages[input_lang]
tgt_code = languages[target_lang]

st.markdown("### 🌐 Language Configuration")
st.write(f"Input Language: *{input_lang}*")
st.write(f"Target Language: *{target_lang}*")
st.divider()

# -----------------------------
# TRANSLATION FUNCTIONS
# -----------------------------
def translate(text, src, tgt):
    return GoogleTranslator(source=src, target=tgt).translate(text)

def compute_scores(original, translated):
    # Back translation
    back_text = translate(translated, tgt_code, src_code)

    # Embeddings
    emb_orig = model.encode(original, convert_to_tensor=True)
    emb_trans = model.encode(translated, convert_to_tensor=True)
    emb_back = model.encode(back_text, convert_to_tensor=True)

    # Scores
    forward_score = util.cos_sim(emb_orig, emb_trans).item() * 100
    back_score = util.cos_sim(emb_trans, emb_back).item() * 100
    semantic_score = util.cos_sim(emb_orig, emb_back).item() * 100

    return back_text, round(forward_score, 2), round(back_score, 2), round(semantic_score, 2)

# -----------------------------
# VISUALIZATION (SIDE-BY-SIDE BAR CHART)
# -----------------------------
def show_visualization(fwd, back, semantic):
    st.markdown("### 📊 Translation Quality Analysis")

    df = pd.DataFrame({
        "Stage": [
            "Forward Translation",
            "Back Translation",
            "Semantic Similarity"
        ],
        "Score (%)": [fwd, back, semantic]
    })

    fig = px.bar(
        df,
        x="Stage",
        y="Score (%)",
        text="Score (%)",
        range_y=[0, 100],
        title="Side-by-Side Translation Quality Comparison"
    )

    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    if semantic >= 80:
        st.success("High quality translation – meaning preserved")
    elif semantic >= 50:
        st.warning("Moderate translation quality")
    else:
        st.error("Low translation quality – meaning loss detected")

# -----------------------------
# TEXT TO SPEECH
# -----------------------------
async def speak(text, lang):
    tts = gTTS(text=text, lang=lang)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# -----------------------------
# CAMERA CROP
# -----------------------------
def crop_center(img, percent=0.6):
    h, w, _ = img.shape
    ch, cw = int(h * percent), int(w * percent)
    y1 = (h - ch) // 2
    x1 = (w - cw) // 2
    return img[y1:y1 + ch, x1:x1 + cw]

# =====================================================
# MODE 1: TEXT TRANSLATOR
# =====================================================
if mode == "Text Translator":
    st.subheader("✍️ Text Translator")
    text = st.text_area("Enter text")

    if st.button("Translate Text"):
        if text.strip():
            translated = translate(text, src_code, tgt_code)

            st.markdown("### 🗂 Input Text")
            st.write(text)

            st.markdown("### 🌍 Translated Text")
            st.write(translated)

            back_text, fwd, back, semantic = compute_scores(text, translated)

            st.markdown("### 🔁 Back Translated Text")
            st.write(back_text)

            show_visualization(fwd, back, semantic)

            audio = asyncio.run(speak(translated, tgt_code))
            st.audio(audio)
            os.remove(audio)
        else:
            st.warning("Please enter text.")

# =====================================================
# MODE 2: VOICE TRANSLATOR
# =====================================================
elif mode == "Voice Translator":
    st.subheader("🎤 Voice Translator (Local System Only)")

    if st.button("Start Voice Input"):
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("Listening...")
                audio = r.listen(source)
                text = r.recognize_google(audio, language=src_code)

            translated = translate(text, src_code, tgt_code)
            back_text, fwd, back, semantic = compute_scores(text, translated)

            st.markdown("### 🗂 Recognized Text")
            st.write(text)

            st.markdown("### 🌍 Translated Text")
            st.write(translated)

            st.markdown("### 🔁 Back Translated Text")
            st.write(back_text)

            show_visualization(fwd, back, semantic)

            audio_out = asyncio.run(speak(translated, tgt_code))
            st.audio(audio_out)
            os.remove(audio_out)

        except Exception as e:
            st.error(e)

# =====================================================
# MODE 3: CAMERA TRANSLATOR
# =====================================================
elif mode == "Camera Translator":
    st.subheader("📷 Camera Text Scanner")

    img_file = st.camera_input("Capture image")

    if img_file:
        image = Image.open(img_file)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_cv = crop_center(img_cv)

        st.image(img_cv, caption="Scanned Area", channels="BGR")

        reader = easyocr.Reader(easyocr_langs[input_lang], gpu=False)
        results = reader.readtext(img_cv, detail=0)
        text = " ".join(results)

        if text.strip():
            translated = translate(text, src_code, tgt_code)
            back_text, fwd, back, semantic = compute_scores(text, translated)

            st.markdown("### 🗂 Detected Text")
            st.write(text)

            st.markdown("### 🌍 Translated Text")
            st.write(translated)

            st.markdown("### 🔁 Back Translated Text")
            st.write(back_text)

            show_visualization(fwd, back, semantic)

            audio = asyncio.run(speak(translated, tgt_code))
            st.audio(audio)
            os.remove(audio)
        else:
            st.warning("No readable text detected.")
