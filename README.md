# ExoHunter AI - Find Habitable Planets!

An intelligent **Streamlit + Machine Learning + LLM** web app that tells you whether a planet can support human life — with **real-time streaming AI explanations** in **English + Roman Urdu**.

**Live Demo**: https://exohunter-ai.streamlit.app  
**GitHub**: https://github.com/Ashoob567/exohunter-ai.git

---

### Features

- Trained **Random Forest** model predicts habitability with confidence score
- **Live streaming** AI report (like ChatGPT) using **Mistral-7B-Instruct**
- Detailed scientific reasoning for every parameter
- Bilingual output: **English + Roman Urdu** (perfect for Pakistan, India, students & teachers)
- Beautiful, user-friendly UI with dark space theme
- Real-time typing animation
- Professional white-background report section (perfect for presentations)

---

### Screenshots

![ExoHunter AI Demo](https://raw.githubusercontent.com/Ashoob567/exohunter-ai/main/screenshot.png)

---

### How It Works

1. Enter planet details (Radius, Mass, Orbit, Star Temperature)
2. Choose from presets (Earth, Kepler-452b, TRAPPIST-1e, etc.)
3. Click **"CHECK IF WE CAN LIVE HERE"**
4. Get instant result + full AI-generated scientific report

---

### Tech Stack

| Technology              | Purpose                              |
|-------------------------|---------------------------------------|
| Python                  | Core language                         |
| Streamlit               | Web app framework                     |
| Scikit-learn            | Random Forest model training          |
| Joblib                  | Model saving/loading                  |
| Pandas & NumPy          | Data handling                         |
| LangChain               | LLM integration & streaming           |
| Hugging Face Inference  | Mistral-7B-Instruct (via API)         |
| python-dotenv           | Secure API key management             |

---

### Installation & Setup

#### 1. Clone the repository
```bash
git clone https://github.com/Ashoob567/exohunter-ai.git
cd exohunter-ai


2. Install dependencies
pip install -r requirements.txt
streamlit>=1.30.0
pandas
numpy
scikit-learn
joblib
python-dotenv
langchain-huggingface>=0.0.3
langchain-core
huggingface-hub


Create a .env file in the root folder:
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here


Get free token → https://huggingface.co/settings/tokens




streamlit run app.py


exohunter-ai/
├── app.py                  Main Streamlit app
├── model_pipeline.pkl      Trained ML model
├── metadata.json           Model info & accuracy
├── requirements.txt        Python dependencies
├── .env                    Your HF token (DO NOT commit!)
├── README.md               This file
├── screenshot.png          Demo image
└── assets/                 (Optional) Planet images