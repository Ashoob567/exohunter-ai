# app.py - MAXIMUM READABILITY + USER FRIENDLY UI (English + Roman Urdu)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ========================== BEAUTIFUL & CLEAN PAGE ==========================
st.set_page_config(
    page_title="ExoHunter AI - Habitable Planet Checker",
    page_icon="planet",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for MAXIMUM readability
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .main {background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); font-family: 'Poppins', sans-serif;}
    h1 {font-size: 3.2rem !important; text-align: center; color: #00d4ff; text-shadow: 0 0 20px #00d4ff60; margin-bottom: 10px;}
    h2 {font-size: 2rem !important; color: #00ff9d;}
    h3 {font-size: 1.6rem !important; color: #7f7fff;}
    
    .big-text {font-size: 1.4rem; line-height: 2; color: #e0e0ff;}
    .result-box {padding: 25px; border-radius: 20px; text-align: center; font-size: 1.8rem; margin: 20px 0;}
    .habitable {background: linear-gradient(90deg, #1a4d2e, #2ecc71); color: white; box-shadow: 0 0 30px #2ecc7180;}
    .not-habitable {background: linear-gradient(90deg, #630d1a, #e74c3c); color: white; box-shadow: 0 0 30px #e74c3c80;}
    
    .report-card {background: #1a1a2e; padding: 30px; border-radius: 20px; border: 2px solid #00d4ff; box-shadow: 0 10px 30px rgba(0,212,255,0.2);}
    .roman-card {background: #2d1b3a; padding: 30px; border-radius: 20px; border: 2px solid gold; box-shadow: 0 10px 30px rgba(255,215,0,0.2);}
    
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff, #7f7fff);
        color: white; font-size: 1.4rem; padding: 15px 40px; border-radius: 50px;
        border: none; font-weight: bold; box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    .stButton>button:hover {transform: translateY(-5px); box-shadow: 0 15px 30px rgba(0,212,255,0.5);}
    
    .metric-card {background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; text-align: center;}
</style>
""", unsafe_allow_html=True)

# ========================== HEADER ==========================
st.title("ExoHunter AI")
st.markdown("<h2 style='text-align:center; color:#00ff9d;'>Check if a Planet Can Support Life</h2>", unsafe_allow_html=True)
st.markdown("<p class='big-text' style='text-align:center;'>Enter planet details → Get instant scientific report in English + Roman Urdu</p>", unsafe_allow_html=True)
st.markdown("---")

# ========================== CONFIG ==========================
MODEL_FILE = "model_pipeline.pkl"
LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    st.error("Please add your Hugging Face token in .env file")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)
model = load_model()

@st.cache_resource(show_spinner="AI is waking up...")
def get_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID, task="conversational", max_new_tokens=1200,
        temperature=0.6, top_p=0.9, repetition_penalty=1.2, streaming=True,
        huggingfacehub_api_token=HF_TOKEN
    )
    return ChatHuggingFace(llm=endpoint)

chat_model = get_llm()

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are ExoHunter AI. Give a LONG, detailed, scientific report in two parts:

1. ENGLISH (Clear & Detailed):
   - Start with: FINAL RESULT: HABITABLE or NOT HABITABLE
   - Confidence percentage
   - Explain EVERY parameter with reasoning
   - Say which parameter is helping and which is harming
   - End with one powerful line

2. ROMAN URDU (Simple but complete):
   - Same report in easy Roman Urdu
   - Explain everything step by step"""),
    MessagesPlaceholder("messages")
])

chain = prompt | chat_model | StrOutputParser() if chat_model else None

def calculate_esi(r, m, d):
    return max(0, (1 - np.sqrt((0.57*((r-1)/(r+1))**2 + 1.07*((m-1)/(m+1))**2 + 0.70*((d-1)/(d+1))**2)/2.34)) * 100)

# ========================== MAIN LAYOUT ==========================
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("### Enter Planet Information")
    
    presets = {
        "Earth (Perfect Example)": {"r":1.0,"m":1.0,"o":1.0,"t":5778},
        "Kepler-452b (Earth's Cousin)": {"r":1.6,"m":5.0,"o":1.05,"t":5757},
        "TRAPPIST-1e (Cold but Possible)": {"r":0.92,"m":0.77,"o":0.029,"t":2510},
        "Hot Jupiter (Gas Giant)": {"r":10.0,"m":300.0,"o":0.05,"t":5000},
    }
    
    choice = st.selectbox("Choose a preset or make your own", ["Custom"] + list(presets.keys()))
    default = presets.get(choice, {"r":1.0,"m":1.0,"o":1.0,"t":5778}) if choice != "Custom" else {"r":1.0,"m":1.0,"o":1.0,"t":5778}

    with st.form("planet_form", clear_on_submit=False):
        st.markdown("**Planet Name**")
        name = st.text_input("", placeholder="e.g., Proxima b", value="My Planet")

        st.markdown("**Size & Weight**")
        c1, c2 = st.columns(2)
        radius = c1.number_input("Radius (Earth = 1)", value=float(default["r"]), step=0.1, help="How big is the planet?")
        mass = c2.number_input("Mass (Earth = 1)", value=float(default["m"]), step=0.1, help="How heavy is it?")

        st.markdown("**Location & Star**")
        c3, c4 = st.columns(2)
        orbit = c3.number_input("Distance from Star (AU)", value=float(default["o"]), step=0.01, help="1 AU = Earth-Sun distance")
        temp = c4.number_input("Star Temperature (K)", value=int(default["t"]), step=50, help="Our Sun = 5778 K")

        submitted = st.form_submit_button("CHECK IF WE CAN LIVE HERE", use_container_width=True)

with col2:
    if submitted:
        with st.spinner("AI is analyzing the planet..."):
            time.sleep(1.5)
            esi = calculate_esi(radius, mass, orbit)
            df = pd.DataFrame([[radius, mass, orbit, temp, esi]],
                            columns=['radius','mass','orbit_distance','star_temperature','earth_similarity_score'])
            pred = int(model.predict(df)[0])
            prob = model.predict_proba(df)[0]
            confidence = prob[1] if pred else prob[0]

        st.markdown(f"### Planet: **{name}**")

        if pred:
            st.markdown(f"<div class='result-box habitable'>LIFE POSSIBLE HERE!<br><h2>FINAL RESULT: HABITABLE</h2>Confidence: {prob[1]:.1%}</div>", unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"<div class='result-box not-habitable'>NOT SAFE FOR HUMANS<br><h2>FINAL RESULT: NOT HABITABLE</h2>Confidence: {prob[0]:.1%}</div>", unsafe_allow_html=True)

        # Metrics
        m1, m2, m3 = st.columns(3)
        with m1: st.markdown(f"<div class='metric-card'><h3>Earth Similarity</h3><h2>{esi:.1f}/100</h2></div>", unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-card'><h3>Gravity</h3><h2>{mass/(radius**2):.2f}g</h2></div>", unsafe_allow_html=True)
        with m3: st.markdown(f"<div class='metric-card'><h3>Star Temp</h3><h2>{temp:,} K</h2></div>", unsafe_allow_html=True)

        st.markdown("### Full Scientific Report")

        if chain:
            placeholder = st.empty()
            full_response = ""

            user_msg = f"""Planet: {name}
            Radius: {radius:.2f} Earths
            Mass: {mass:.2f} Earths
            Orbit Distance: {orbit:.3f} AU
            Star Temperature: {temp:,} K
            ESI Score: {esi:.1f}/100
            Model Prediction: {'HABITABLE' if pred else 'NOT HABITABLE'}
            Confidence: {confidence:.1%}
Give a long, detailed, scientific report in English and Roman Urdu. 
IMPORTANT: NEVER split words in the middle. Do not use line breaks inside words. Always finish a word before going to the next line."""

            try:
                for chunk in chain.stream({"messages": [HumanMessage(content=user_msg)]}):
                    text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    full_response += text

                    # THIS IS THE MAGIC — FORCES PERFECT WORD WRAPPING
                    safe_text = full_response.replace("\n", " ")  # Remove any forced breaks
                    safe_text = safe_text.replace("\\", "")      # Remove backslashes

                    placeholder.markdown(f"""
                    <div style="
                        background: white;
                        color: black;
                        padding: 40px;
                        border-radius: 22px;
                        border: 5px solid #00d4ff;
                        box-shadow: 0 20px 50px rgba(0,0,0,0.2);
                        font-family: 'Georgia', serif;
                        font-size: 19px;
                        line-height: 2.1;
                        text-align: justify;
                        word-break: keep-all;
                        overflow-wrap: anywhere;
                        hyphens: none;
                        white-space: normal;
                    ">
                        {safe_text} <span style="color:#00d4ff; animation: blink 1s infinite;">▍</span>
                    </div>

                    <style>
                    @keyframes blink {{
                        0%, 100% {{ opacity: 1; }}
                        50% {{ opacity: 0; }}
                    }}
                    </style>
                    """, unsafe_allow_html=True)

                # Final version — clean and perfect
                final_text = full_response.replace("\n", " ").replace("\\", "")
                placeholder.markdown(f"""
                <div style="
                    background: white;
                    color: black;
                    padding: 40px;
                    border-radius: 22px;
                    border: 5px solid #00d4ff;
                    box-shadow: 0 20px 50px rgba(0,0,0,0.2);
                    font-family: 'Georgia', serif;
                    font-size: 19px;
                    line-height: 2.1;
                    text-align: justify;
                    word-break: keep-all;
                    overflow-wrap: anywhere;
                ">
                    {final_text}
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("AI not connected")

    else:
        st.markdown("### Welcome! Please enter planet details on the left")
        st.image("https://images.unsplash.com/photo-1614730321143-d1c6c06f4d75?q=80&w=2070&auto=format&fit=crop", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#888; font-size:1.1rem;'>Made with ❤️ for Pakistan | Science + AI for Everyone</p>", unsafe_allow_html=True)