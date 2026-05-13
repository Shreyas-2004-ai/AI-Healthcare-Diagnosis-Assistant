# AI Healthcare Diagnosis Assistant

A full-stack web application that predicts diseases from symptoms using a machine learning model validated by an AI language model. Enter your symptoms, get a diagnosis with description, precautions, medications, diet, and workout recommendations.

---

## How It Works

1. User enters symptoms in the frontend
2. Backend runs the symptoms through the SVC ML model
3. If a feedback-trained Random Forest model exists, it runs first and takes priority
4. The ML result is sent to the Groq LLM (llama-3.1-8b-instant) for validation and correction
5. If Groq is unavailable, a smart rule-based fallback handles the response
6. The final result is returned with disease name, confidence, description, precautions, medications, diet, and workouts

---

## Project Structure

```
mini-project/
├── backend/
│   ├── app.py                        # Main Flask app — all routes and ML logic
│   ├── email_config.py               # Email settings loaded from env vars
│   ├── requirements.txt              # Python dependencies (pinned versions)
│   ├── runtime.txt                   # Python version for deployment (3.12.0)
│   ├── Procfile                      # Gunicorn start command for Heroku
│   ├── Dockerfile                    # Multi-stage Docker image for backend
│   ├── .dockerignore                 # Files excluded from Docker build
│   ├── .env.example                  # Template for required environment variables
│   ├── vercel.json                   # Vercel serverless config (not used in current deploy)
│   └── src/
│       ├── models/
│       │   ├── svc.pkl               # Pre-trained Support Vector Classifier
│       │   └── feedback_rf.pkl       # Random Forest retrained from user feedback
│       └── datasets/
│           ├── Training.csv          # 4920 rows × 133 cols (132 symptoms + prognosis)
│           ├── symtoms_df.csv        # Symptom descriptions
│           ├── Symptom-severity.csv  # Severity weights per symptom
│           ├── description.csv       # Disease descriptions
│           ├── precautions_df.csv    # 4 precautions per disease
│           ├── medications.csv       # Medications per disease
│           ├── diets.csv             # Diet recommendations per disease
│           ├── workout_df.csv        # Workout suggestions per disease
│           └── user_feedback.csv     # Stores user-confirmed diagnoses for retraining
├── frontend/
│   ├── build/                        # Production build (served directly)
│   ├── package.json                  # Node dependencies and scripts
│   ├── vercel.json                   # Vercel static deployment config
│   └── .env.example                  # Frontend env variable template
├── .gitignore
├── README.md
└── PROJECT_REPORT.md
```

---

## Backend

### Tech Stack

| Package | Version | What it does |
|---|---|---|
| Python | 3.12.0 | Runtime |
| Flask | 3.1.2 | REST API framework |
| flask-cors | 6.0.1 | Allows frontend to call the API |
| gunicorn | 21.2.0 | Production WSGI server |
| numpy | 2.3.3 | Builds symptom input vectors |
| pandas | 2.3.2 | Loads and queries all CSV datasets |
| scikit-learn | (installed) | SVC prediction + Random Forest retraining |
| joblib | 1.3.2 | Model serialization support |
| Werkzeug | 3.1.3 | Flask dependency |
| Jinja2 | 3.1.6 | Flask dependency |

### ML Models

**Primary — Support Vector Classifier (svc.pkl)**
- Trained on Training.csv: 4920 samples, 132 binary symptom features, 41 disease classes
- Uses `decision_function` output converted to pseudo-probabilities via softmax
- Returns top-3 disease predictions with confidence scores

**Secondary — Random Forest (feedback_rf.pkl)**
- Trained on Training.csv combined with accumulated user feedback
- 400 estimators, balanced class weights
- Loaded at startup if the file exists; takes priority over SVC when available
- Retrained on demand via `/api/retrain`

**Fuzzy Symptom Matching**
- Input symptoms are normalized (lowercase, spaces → underscores)
- Manual replacements for common typos: `diarrhea` → `diarrhoea`, `shortness_of_breath` → `breathlessness`
- `difflib.get_close_matches` with 0.78 cutoff handles remaining variations
- Unknown symptoms are passed to the LLM instead of being dropped

**Rule-Based Overrides**
- 11 classic symptom clusters trigger direct disease assignment (e.g. burning urination + bladder discomfort → UTI)
- Severe diseases (AIDS, Heart attack, Tuberculosis, etc.) are suppressed unless hallmark symptoms are present
- Activates as final fallback when both LLM APIs are unavailable

### AI Validation Layer

The ML prediction is sent to an LLM for validation and correction. The LLM receives the symptom list and ML prediction, then either confirms or replaces the diagnosis.

- Primary: Groq API — `llama-3.1-8b-instant`, temperature 0.2, 512 max tokens
- Fallback: Smart rule-based system (no second LLM — Gemini integration exists in `/api/enhance` only)
- Both are configured via environment variables — no keys are hardcoded anywhere

### API Endpoints

**POST /api/predict**

Accepts symptoms, runs ML + LLM pipeline, returns full diagnosis.

Request:
```json
{ "symptoms": ["itching", "skin_rash", "fatigue"] }
```

Response:
```json
{
  "disease": "Fungal infection",
  "confidence": "High",
  "description": "...",
  "precautions": ["Keep area dry", "Use antifungal cream", ...],
  "medications": ["Clotrimazole", ...],
  "diets": ["Avoid sugar", ...],
  "workouts": ["Light walking", ...],
  "unknown_symptoms": []
}
```

**POST /api/enhance**

Second-pass AI validation using Groq → Gemini → rule-based fallback chain.

Request:
```json
{ "symptoms": ["headache", "nausea"], "disease": "Migraine" }
```

**POST /api/feedback**

Submits user feedback form. Sends an email via Gmail SMTP if email env vars are configured. Falls back to console logging if email is not set up.

Request:
```json
{
  "name": "John",
  "email": "john@example.com",
  "subject": "Great tool",
  "message": "Very accurate prediction",
  "rating": 5,
  "category": "general"
}
```

**POST /api/learn**

Stores a user-confirmed diagnosis into `user_feedback.csv` for future retraining.

Request:
```json
{ "symptoms": ["cough", "fever"], "confirmed_disease": "Common Cold" }
```

**POST /api/retrain**

Retrains the Random Forest model on Training.csv + all accumulated user feedback. Updates the in-memory model immediately.

### Datasets

| File | Rows | Purpose |
|---|---|---|
| Training.csv | 4920 | ML training data — 132 symptom columns + prognosis |
| description.csv | 41 | One description per disease |
| precautions_df.csv | 41 | 4 precautions per disease |
| medications.csv | 41 | Medications per disease |
| diets.csv | 41 | Diet recommendations per disease |
| workout_df.csv | 41 | Workout suggestions per disease |
| symtoms_df.csv | 132 | Symptom descriptions |
| Symptom-severity.csv | 132 | Severity weight per symptom |
| user_feedback.csv | grows | User-confirmed diagnoses for retraining |

### Diseases Covered (41 total)

Fungal infection, Allergy, GERD, Chronic cholestasis, Drug Reaction, Peptic ulcer disease, AIDS, Diabetes, Gastroenteritis, Bronchial Asthma, Hypertension, Migraine, Cervical spondylosis, Paralysis (brain hemorrhage), Jaundice, Malaria, Chicken pox, Dengue, Typhoid, Hepatitis A/B/C/D/E, Alcoholic hepatitis, Tuberculosis, Common Cold, Pneumonia, Dimorphic haemorrhoids (piles), Heart attack, Varicose veins, Hypothyroidism, Hyperthyroidism, Hypoglycemia, Osteoarthritis, Arthritis, Vertigo, Acne, Urinary tract infection, Psoriasis, Impetigo

### Email System

Configured entirely through environment variables. Three SMTP connection methods are tried in order:
1. SMTP + STARTTLS on port 587
2. SMTP_SSL on port 465
3. Plain SMTP on port 587

If all three fail, the feedback is logged to the console and a success response is still returned to the user.

---

## Frontend

### Tech Stack

| Package | Version | What it does |
|---|---|---|
| React | 18.3.1 | UI framework |
| react-dom | 18.3.1 | DOM rendering |
| react-router-dom | 7.0.2 | Client-side routing |
| axios | 1.7.8 | HTTP calls to backend API |
| bootstrap | 5.3.3 | CSS framework |
| react-bootstrap | 2.10.6 | Bootstrap components for React |
| firebase | 11.1.0 | User authentication |
| react-scripts | 5.0.1 | Build tooling (CRA) |

### Deployment

The frontend is deployed as a static site. The `build/` folder is committed to the repository and served directly by Vercel — no build step runs on the server. `vercel.json` configures SPA routing so all paths fall back to `index.html`.

---

## Running Locally

### Backend

```bash
cd mini-project/backend
pip install -r requirements.txt
```

Create a `.env` file (copy from `.env.example`):
```
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your@gmail.com
EMAIL_PASSWORD=your_app_password
RECIPIENT_EMAIL=your@gmail.com
```

```bash
python app.py
# Backend runs on http://localhost:5000
```

### Frontend

The `build/` folder is already compiled. Serve it directly:

```bash
cd mini-project/frontend/build
python -m http.server 3001
# Frontend runs on http://localhost:3001
```

---

## Docker (Backend)

```bash
cd mini-project/backend
docker build -t health-ai-backend .

docker run -d -p 5000:5000 \
  -e GROQ_API_KEY=your_key \
  -e GEMINI_API_KEY=your_key \
  -e EMAIL_USER=your@gmail.com \
  -e EMAIL_PASSWORD=your_app_password \
  -e RECIPIENT_EMAIL=your@gmail.com \
  health-ai-backend
```

The Dockerfile uses a multi-stage build:
- Stage 1 (builder): installs gcc, g++, libgomp1, and all Python packages
- Stage 2 (runtime): copies only installed packages and app code — no build tools in final image
- Runs gunicorn with 2 workers and 120s timeout

---

## Deploying to Vercel

### Frontend
1. Connect the GitHub repo to Vercel
2. Set root directory to `mini-project/frontend`
3. Framework preset: Other
4. Build command: `echo done` (build is pre-compiled)
5. Output directory: `build`

### Backend
1. Deploy `mini-project/backend` as a separate Vercel project
2. Add all environment variables in the Vercel dashboard
3. Note: the existing `backend/vercel.json` targets `api/predict.py` and `api/feedback.py` as serverless functions — these are separate from `app.py`. For a full Flask deployment use a platform that supports long-running processes (Railway, Render, Heroku).

### Heroku (recommended for backend)
```bash
cd mini-project/backend
heroku create your-app-name
heroku config:set GROQ_API_KEY=your_key
heroku config:set GEMINI_API_KEY=your_key
heroku config:set EMAIL_USER=your@gmail.com
heroku config:set EMAIL_PASSWORD=your_app_password
heroku config:set RECIPIENT_EMAIL=your@gmail.com
git push heroku main
```

---

## Environment Variables

All secrets are loaded from environment variables. Nothing is hardcoded.

| Variable | Required | Purpose |
|---|---|---|
| GROQ_API_KEY | Yes (for LLM) | Groq API for AI prediction validation |
| GEMINI_API_KEY | Yes (for /api/enhance) | Google Gemini fallback in enhance endpoint |
| EMAIL_HOST | No | SMTP host, defaults to smtp.gmail.com |
| EMAIL_PORT | No | SMTP port, defaults to 587 |
| EMAIL_USER | No | Gmail address for sending feedback emails |
| EMAIL_PASSWORD | No | Gmail App Password |
| RECIPIENT_EMAIL | No | Where feedback emails are delivered |

The app works without email vars — feedback just logs to console. The app works without LLM keys — falls back to pure ML + rule-based predictions.

---

## Medical Disclaimer

This tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.
