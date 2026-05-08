# AI Healthcare Diagnosis Assistant

A comprehensive AI-powered health prediction system that uses machine learning and rule-based algorithms to predict diseases based on symptoms and provides detailed health insights including descriptions, precautions, medications, dietary suggestions, and wellness recommendations.

## 🚀 Features

- **AI-Powered Disease Prediction**: Multi-model ensemble approach with SVC, Random Forest, Gradient Boosting, and Logistic Regression
- **Groq & Gemini AI Integration**: Advanced LLM-based validation and correction of predictions
- **Comprehensive Health Insights**: Detailed information about predicted conditions
- **User Feedback Learning**: Continuous model improvement through user feedback
- **Real-time Symptom Suggestions**: Fuzzy matching and autocomplete for better UX
- **Email Feedback System**: SMTP-based feedback collection
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Docker Support**: Containerized backend deployment

---

## 📋 Table of Contents

- [Technology Stack](#technology-stack)
- [Machine Learning Algorithms](#machine-learning-algorithms)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Deployment](#deployment)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

---

## 🛠️ Technology Stack

### **Backend Technologies**

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.12.0 | Core backend language |
| **Flask** | 3.1.2 | Web framework for REST API |
| **Flask-CORS** | 6.0.1 | Cross-origin resource sharing |
| **Gunicorn** | 21.2.0 | Production WSGI server |
| **NumPy** | 2.3.3 | Numerical computing |
| **Pandas** | 2.3.2 | Data manipulation and analysis |
| **Scikit-learn** | 1.7.2 | Machine learning models |
| **Joblib** | 1.3.2 | Model serialization |
| **Requests** | Latest | HTTP library for API calls |

### **Frontend Technologies**

| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 18.3.1 | UI framework |
| **React DOM** | 18.3.1 | DOM rendering |
| **React Router DOM** | 7.0.2 | Client-side routing |
| **Axios** | 1.7.8 | HTTP client |
| **Bootstrap** | 5.3.3 | CSS framework |
| **React Bootstrap** | 2.10.6 | Bootstrap components for React |
| **Firebase** | 11.1.0 | Authentication & database |
| **React Scripts** | 5.0.1 | Build tooling |

### **AI/ML Libraries**

- **Scikit-learn**: SVC, Random Forest, Gradient Boosting, Logistic Regression
- **Groq API**: LLM-based prediction validation (llama-3.1-8b-instant)
- **Gemini API**: Fallback LLM for enhanced predictions (gemini-2.0-flash)

### **DevOps & Deployment**

- **Docker**: Containerization
- **Vercel**: Frontend & serverless backend hosting
- **Heroku**: Alternative backend deployment
- **Git/GitHub**: Version control

---

## 🤖 Machine Learning Algorithms

### **1. Support Vector Classifier (SVC)**
- **Kernel**: RBF (Radial Basis Function)
- **C Parameter**: 10.0
- **Gamma**: Scale
- **Purpose**: Primary classification model with high accuracy
- **Features**: Probability estimates, class weighting

### **2. Random Forest Classifier**
- **Estimators**: 200 trees
- **Max Depth**: None (full depth)
- **Max Features**: sqrt
- **Purpose**: Ensemble learning with feature importance
- **Features**: Bootstrap aggregating, balanced class weights

### **3. Gradient Boosting Classifier**
- **Estimators**: 150
- **Learning Rate**: 0.1
- **Max Depth**: 5
- **Subsample**: 0.8
- **Purpose**: Sequential ensemble for improved accuracy

### **4. Logistic Regression**
- **Solver**: liblinear
- **Penalty**: L2 regularization
- **C Parameter**: 1.0
- **Max Iterations**: 2000
- **Purpose**: Baseline linear model

### **5. Ensemble Voting System**
- **Method**: Weighted soft voting
- **Weights**: Dynamic based on cross-validation F1 scores
  - Random Forest: ~35%
  - Gradient Boosting: ~30%
  - SVM: ~25%
  - Logistic Regression: ~10%

### **6. Rule-Based System**
- **Fuzzy Symptom Matching**: Difflib-based similarity scoring
- **Pattern Recognition**: 13+ disease patterns with symptom combinations
- **Confidence Boosting**: Agreement-based and symptom-match boosting

### **7. LLM Integration (Groq/Gemini)**
- **Primary**: Groq API with llama-3.1-8b-instant
- **Fallback**: Gemini 2.0 Flash
- **Purpose**: Validation, correction, and handling unknown symptoms
- **Temperature**: 0.2 (low for consistency)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Symptom Input│  │  Prediction  │  │   Feedback   │     │
│  │   Component  │  │   Display    │  │     Form     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/REST API
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (Flask API)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              API Routes (app.py)                      │  │
│  │  /api/predict  │  /api/enhance  │  /api/feedback    │  │
│  │  /api/learn    │  /api/retrain                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                 │
│  ┌─────────────────────────┴──────────────────────────┐   │
│  │         ML Prediction Engine                        │   │
│  │  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │  SVC Model   │  │ Random Forest│               │   │
│  │  └──────────────┘  └──────────────┘               │   │
│  │  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │   Gradient   │  │   Logistic   │               │   │
│  │  │   Boosting   │  │  Regression  │               │   │
│  │  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                 │
│  ┌─────────────────────────┴──────────────────────────┐   │
│  │         LLM Integration Layer                       │   │
│  │  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │  Groq API    │  │  Gemini API  │               │   │
│  │  │  (Primary)   │  │  (Fallback)  │               │   │
│  │  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                 │
│  ┌─────────────────────────┴──────────────────────────┐   │
│  │         Data Layer                                  │   │
│  │  • Training.csv (132 symptoms × 41 diseases)       │   │
│  │  • Symptom severity, descriptions, precautions     │   │
│  │  • Medications, diets, workouts                    │   │
│  │  • User feedback storage                           │   │
│  │  • Model persistence (pickle)                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### **Prerequisites**
- Python 3.12+
- Node.js 14+
- npm or yarn
- Git

### **Backend Setup**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Shreyas-2004-ai/AI-Healthcare-Diagnosis-Assistant.git
   cd AI-Healthcare-Diagnosis-Assistant/mini-project/backend
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the `backend/` directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   EMAIL_HOST=smtp.gmail.com
   EMAIL_PORT=587
   EMAIL_USER=your-email@gmail.com
   EMAIL_PASSWORD=your-gmail-app-password
   RECIPIENT_EMAIL=your-email@gmail.com
   ```

5. **Run the backend**:
   ```bash
   python app.py
   ```
   Backend will start on `http://localhost:5000`

### **Frontend Setup**

1. **Navigate to frontend**:
   ```bash
   cd ../frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Set up environment variables**:
   Create a `.env` file in the `frontend/` directory:
   ```env
   REACT_APP_API_BASE=http://localhost:5000
   REACT_APP_GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Run the frontend**:
   ```bash
   npm start
   ```
   Frontend will start on `http://localhost:3001`

---

## 🐳 Docker Deployment

### **Build Docker Image**

```bash
cd backend
docker build -t health-ai-backend .
```

### **Run Docker Container**

```bash
docker run -d -p 5000:5000 \
  -e GROQ_API_KEY=your_key \
  -e GEMINI_API_KEY=your_key \
  -e EMAIL_USER=your_email \
  -e EMAIL_PASSWORD=your_password \
  -e RECIPIENT_EMAIL=your_email \
  health-ai-backend
```

---

## 🌐 Deployment

### **Vercel Deployment**

**Frontend:**
1. Connect GitHub repo to Vercel
2. Set root directory: `mini-project/frontend`
3. Framework: Other/Create React App
4. Build command: `echo 'Using pre-built files'`
5. Output directory: `build`

**Backend (Serverless):**
1. Deploy `mini-project/backend` separately
2. Add environment variables in Vercel dashboard
3. Vercel will auto-detect Python and use `vercel.json`

### **Heroku Deployment**

```bash
cd backend
heroku create your-app-name
heroku config:set GROQ_API_KEY=your_key
heroku config:set GEMINI_API_KEY=your_key
git push heroku main
```

---

## 🔌 API Endpoints

### **POST /api/predict**
Predict disease from symptoms.

**Request:**
```json
{
  "symptoms": ["fever", "cough", "headache"]
}
```

**Response:**
```json
{
  "disease": "Common Cold",
  "confidence": "High",
  "description": "A viral infection...",
  "precautions": ["Rest", "Hydrate"],
  "medications": ["Decongestants"],
  "diets": ["Warm liquids"],
  "workouts": ["Light walking"],
  "unknown_symptoms": []
}
```

### **POST /api/enhance**
Enhance prediction using LLM.

### **POST /api/feedback**
Submit user feedback.

### **POST /api/learn**
Store user-confirmed diagnosis for model improvement.

### **POST /api/retrain**
Retrain model with accumulated feedback.

---

## 📁 Project Structure

```
mini-project/
├── backend/
│   ├── app.py                      # Main Flask application
│   ├── improved_predictor.py       # Enhanced ML predictor
│   ├── alternative_predictor.py    # Rule-based fallback
│   ├── email_config.py             # Email configuration
│   ├── requirements.txt            # Python dependencies
│   ├── runtime.txt                 # Python version
│   ├── Dockerfile                  # Docker configuration
│   ├── .dockerignore               # Docker ignore rules
│   ├── Procfile                    # Heroku configuration
│   ├── vercel.json                 # Vercel configuration
│   └── src/
│       ├── models/
│       │   ├── svc.pkl             # Pre-trained SVC model
│       │   └── feedback_rf.pkl     # Feedback-trained model
│       └── datasets/
│           ├── Training.csv        # Training data
│           ├── Symptom-severity.csv
│           ├── description.csv
│           ├── precautions_df.csv
│           ├── medications.csv
│           ├── diets.csv
│           ├── workout_df.csv
│           └── user_feedback.csv
├── frontend/
│   ├── build/                      # Production build
│   ├── package.json                # Node dependencies
│   ├── vercel.json                 # Vercel config
│   └── .env.example                # Environment template
├── .gitignore
└── README.md
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⚠️ Medical Disclaimer

This system provides AI-powered predictions for **educational purposes only**. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical concerns.

---

## 📄 License

This project is for educational purposes. Ensure compliance with local regulations regarding medical software.

---

## 👨‍💻 Author

**Shreyas**
- GitHub: [@Shreyas-2004-ai](https://github.com/Shreyas-2004-ai)
- Repository: [AI-Healthcare-Diagnosis-Assistant](https://github.com/Shreyas-2004-ai/AI-Healthcare-Diagnosis-Assistant)

---

## 🙏 Acknowledgments

- Scikit-learn for ML algorithms
- Groq & Google for LLM APIs
- React & Flask communities
- Open-source contributors
