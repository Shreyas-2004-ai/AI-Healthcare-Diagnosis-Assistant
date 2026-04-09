from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
from flask_cors import CORS
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from typing import List
import difflib

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Flask app
app = Flask(__name__)
CORS(app)

# Import email configuration
try:
    from email_config import EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASSWORD, RECIPIENT_EMAIL
except ImportError:
    # Fallback configuration (you need to set up email_config.py)
    EMAIL_HOST = 'smtp.gmail.com'
    EMAIL_PORT = 587
    EMAIL_USER = 'your-email@gmail.com'  # Replace with your Gmail
    EMAIL_PASSWORD = 'your-app-password'  # Replace with your Gmail app password
    RECIPIENT_EMAIL = 'shreyasssanil62@gmail.com'

# Load datasets
sym_des = pd.read_csv("src/datasets/symtoms_df.csv")
precautions = pd.read_csv("src/datasets/precautions_df.csv")
workout = pd.read_csv("src/datasets/workout_df.csv")
description = pd.read_csv("src/datasets/description.csv")
medications = pd.read_csv('src/datasets/medications.csv')
diets = pd.read_csv("src/datasets/diets.csv")

# Load model
svc = pickle.load(open('src/models/svc.pkl', 'rb'))

# Optional: load a retrained feedback-driven model if available
rf_model_path = 'src/models/feedback_rf.pkl'
rf_model = None
rf_feature_order: List[str] = []
if os.path.exists(rf_model_path):
    try:
        with open(rf_model_path, 'rb') as f:
            saved = pickle.load(f)
            rf_model = saved.get('model')
            rf_feature_order = saved.get('feature_order', [])
    except Exception:
        rf_model = None
        rf_feature_order = []

# Helper functions
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([str(w) for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [str(col) for row in pre.values for col in row]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [str(med) for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [str(die) for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = [str(wrk) for wrk in wrkout.values]

    return desc, pre, med, die, wrkout

def send_email(subject, body):
    """Send email using Gmail SMTP with multiple fallback methods"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = subject

        # Add body to email
        msg.attach(MIMEText(body, 'plain'))

        # Method 1: Try with regular SMTP (port 587)
        try:
            print("Trying SMTP method 1 (port 587)...")
            server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            text = msg.as_string()
            server.sendmail(EMAIL_USER, RECIPIENT_EMAIL, text)
            server.quit()
            print(f"Email sent successfully to {RECIPIENT_EMAIL}")
            return True
        except Exception as e1:
            print(f"Method 1 failed: {e1}")
            
            # Method 2: Try with SSL (port 465)
            try:
                print("Trying SMTP method 2 (port 465)...")
                server = smtplib.SMTP_SSL(EMAIL_HOST, 465)
                server.login(EMAIL_USER, EMAIL_PASSWORD)
                text = msg.as_string()
                server.sendmail(EMAIL_USER, RECIPIENT_EMAIL, text)
                server.quit()
                print(f"Email sent successfully to {RECIPIENT_EMAIL}")
                return True
            except Exception as e2:
                print(f"Method 2 failed: {e2}")
                
                # Method 3: Try without TLS
                try:
                    print("Trying SMTP method 3 (no TLS)...")
                    server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
                    server.login(EMAIL_USER, EMAIL_PASSWORD)
                    text = msg.as_string()
                    server.sendmail(EMAIL_USER, RECIPIENT_EMAIL, text)
                    server.quit()
                    print(f"Email sent successfully to {RECIPIENT_EMAIL}")
                    return True
                except Exception as e3:
                    print(f"Method 3 failed: {e3}")
                    return False
        
    except Exception as e:
        print(f"Error in send_email: {e}")
        return False

# Symptom dictionary
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


# Model prediction function
def _map_symptoms_with_fuzzy(symptom_list: List[str]):
    """Normalize and fuzzy-match symptoms to known keys in symptoms_dict."""
    known_keys = list(symptoms_dict.keys())
    mapped = []
    unknown = []
    for s in symptom_list:
        if not s:
            continue
        s_norm = str(s).strip().lower().replace(' ', '_')
        # quick fixes for common typos
        replacements = {
            'congestio': 'congestion',
            'eadache': 'headache',
            'diarrhea': 'diarrhoea',
            'shortness_of_breath': 'breathlessness'
        }
        if s_norm in replacements:
            s_norm = replacements[s_norm]
        if s_norm in symptoms_dict:
            mapped.append(s_norm)
            continue
        # fuzzy match
        match = difflib.get_close_matches(s_norm, known_keys, n=1, cutoff=0.78)
        if match:
            mapped.append(match[0])
        else:
            unknown.append(s)
    return mapped, unknown


def get_predicted_value(patient_symptoms, top_k=3):
    # Filter to known symptoms and build input vector
    known_symptoms, fuzzy_unknown = _map_symptoms_with_fuzzy(patient_symptoms)
    input_vector = np.zeros(len(symptoms_dict))
    for item in known_symptoms:
        input_vector[symptoms_dict[item]] = 1

    # Guard: if no known symptoms, return low-confidence response
    if input_vector.sum() == 0:
        return {
            'primary_disease': None,
            'confidence': 0.0,
            'top_predictions': [],
            'unknown_symptoms': fuzzy_unknown
        }

    # Try to get class scores
    scores = None
    confidences = None
    try:
        classes = getattr(svc, 'classes_', None)
        # Normalize classes to string disease names when possible
        label_names = None
        if classes is not None:
            label_names = []
            for c in classes:
                if isinstance(c, str):
                    label_names.append(c)
                else:
                    try:
                        label_names.append(diseases_list.get(int(c), str(c)))
                    except Exception:
                        label_names.append(str(c))

        if hasattr(svc, 'predict_proba'):
            proba = svc.predict_proba([input_vector])[0]
            confidences = proba
            scores = proba
            class_names_for_scores = label_names if label_names is not None else [diseases_list.get(i) for i in range(len(proba))]
        else:
            # Use decision_function and convert to pseudo-probabilities via softmax
            decision = svc.decision_function([input_vector])
            decision = np.array(decision).ravel()
            # Handle binary SVM where decision_function returns single score
            if decision.shape[0] == 1 and classes is not None and len(classes) == 2:
                decision = np.array([-decision[0], decision[0]])
            # numerical-stable softmax
            exps = np.exp(decision - np.max(decision))
            softmax = exps / np.sum(exps)
            confidences = softmax
            scores = softmax
            class_names_for_scores = label_names if label_names is not None else [diseases_list.get(i) for i in range(len(softmax))]
    except Exception:
        # Fallback to hard prediction only
        raw = svc.predict([input_vector])[0]
        if isinstance(raw, str):
            disease = raw
        else:
            try:
                disease = diseases_list.get(int(raw))
            except Exception:
                disease = str(raw)
        return {
            'primary_disease': disease,
            'confidence': None,
            'top_predictions': [{'disease': disease, 'confidence': None}],
            'unknown_symptoms': fuzzy_unknown
        }

    # Map class indices to disease names consistently
    # Build an ordered list of (disease, confidence)
    pred_pairs = []
    for idx in range(len(confidences)):
        name = None
        if 'class_names_for_scores' in locals() and class_names_for_scores is not None:
            name = class_names_for_scores[idx]
        if name is None:
            # fallback to diseases_list by index
            name = diseases_list.get(idx, str(idx))
        pred_pairs.append((name, float(confidences[idx])))

    # Rule-based overrides for classic symptom clusters (allow partial matches)
    classic_rules = [
        ({'cough', 'runny_nose', 'sore_throat', 'mild_fever', 'congestion'}, 'Common Cold', 3),
        ({'nausea', 'vomiting', 'diarrhoea', 'abdominal_pain'}, 'Gastroenteritis', 3),
        ({'burning_micturition', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine'}, 'Urinary tract infection', 3),
        ({'itching', 'skin_rash'}, 'Fungal infection', 2),
        ({'headache', 'nausea', 'vomiting'}, 'Migraine', 2),
        ({'cough', 'breathlessness', 'chest_pain'}, 'Bronchial Asthma', 2),
        ({'yellowish_skin', 'yellow_urine', 'dark_urine', 'yellowing_of_eyes'}, 'Jaundice', 3),
        ({'high_fever', 'headache', 'pain_behind_the_eyes', 'nausea'}, 'Dengue', 3),
        ({'high_fever', 'abdominal_pain', 'diarrhoea', 'vomiting'}, 'Typhoid', 3),
        ({'acidity', 'stomach_pain', 'vomiting'}, 'GERD', 2),
        ({'weight_loss', 'polyuria', 'increased_appetite', 'fatigue'}, 'Diabetes ', 3)
    ]
    known_set = set(known_symptoms)
    for req_set, disease_name, min_count in classic_rules:
        if len(req_set.intersection(known_set)) >= min_count:
            return {
                'primary_disease': disease_name,
                'confidence': 0.6,
                'top_predictions': [{'disease': disease_name, 'confidence': 0.6}],
                'unknown_symptoms': fuzzy_unknown
            }

    # Post-filter severe diagnoses unless hallmark symptoms present
    severe_hallmarks = {
        'AIDS': {'weight_loss', 'extra_marital_contacts', 'receiving_blood_transfusion', 'receiving_unsterile_injections'},
        'Heart attack': {'chest_pain', 'fast_heart_rate', 'sweating'},
        'Paralysis (brain hemorrhage)': {'weakness_of_one_body_side', 'slurred_speech', 'loss_of_balance'},
        'Tuberculosis': {'cough', 'blood_in_sputum', 'chest_pain', 'malaise'},
        'Hepatitis B': {'yellowing_of_eyes', 'dark_urine', 'yellowish_skin'},
        'Hepatitis C': {'yellowing_of_eyes', 'dark_urine', 'yellowish_skin'}
    }

    filtered_pairs = []
    for disease_name, conf in pred_pairs:
        if disease_name in severe_hallmarks:
            if len(severe_hallmarks[disease_name].intersection(known_set)) == 0:
                # Skip severe disease if no hallmark symptoms present
                continue
        filtered_pairs.append((disease_name, conf))

    # Fall back to original pairs if filtering removes all
    if not filtered_pairs:
        filtered_pairs = pred_pairs

    # Sort by confidence desc and take top_k after filtering
    filtered_pairs.sort(key=lambda x: x[1], reverse=True)
    top = filtered_pairs[:max(1, top_k)]

    primary_disease, primary_conf = top[0]

    # Heuristic: if confidence is low and there are few symptoms, avoid alarming diagnoses
    severe_set = {
        'AIDS', 'Heart attack', 'Paralysis (brain hemorrhage)', 'Tuberculosis',
        'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis'
    }
    min_symptoms_for_severe = 3
    low_conf_threshold = 0.35

    if (primary_disease in severe_set and
        (len(known_symptoms) < min_symptoms_for_severe or primary_conf < low_conf_threshold) and
        len(top) > 1):
        # Prefer the next best non-severe if available
        for cand_disease, cand_conf in top[1:]:
            if cand_disease not in severe_set:
                primary_disease, primary_conf = cand_disease, cand_conf
                break

    return {
        'primary_disease': primary_disease,
        'confidence': float(primary_conf),
        'top_predictions': [{'disease': d, 'confidence': c} for d, c in top],
        'unknown_symptoms': [s for s in patient_symptoms if s not in symptoms_dict]
    }


def ensure_feedback_store():
    """Ensure feedback CSV exists with the same schema as Training.csv."""
    feedback_path = 'src/datasets/user_feedback.csv'
    if not os.path.exists(feedback_path):
        try:
            train = pd.read_csv('src/datasets/Training.csv')
            # Create empty file with same columns
            empty = train.iloc[0:0]
            empty.to_csv(feedback_path, index=False)
        except Exception:
            # Minimal fallback: create columns from symptoms_dict + prognosis
            cols = list(symptoms_dict.keys()) + ['prognosis']
            pd.DataFrame(columns=cols).to_csv(feedback_path, index=False)
    return feedback_path


def append_feedback(symptoms: List[str], confirmed_disease: str):
    """Append a labeled example to the feedback CSV in one-hot format."""
    feedback_path = ensure_feedback_store()
    # Load header to get column order
    header = pd.read_csv(feedback_path, nrows=0)
    cols = list(header.columns)
    feature_cols = [c for c in cols if c != 'prognosis']
    row = {c: 0 for c in feature_cols}
    # Map known symptoms to 1
    for s in symptoms:
        if s in row:
            row[s] = 1
    row['prognosis'] = confirmed_disease
    # Append
    df_row = pd.DataFrame([row])[cols]
    df_row.to_csv(feedback_path, mode='a', header=False, index=False)


def retrain_feedback_model():
    """Train a RandomForest on Training.csv + user feedback; persist with feature order."""
    try:
        base = pd.read_csv('src/datasets/Training.csv')
        feedback_path = ensure_feedback_store()
        fb = pd.read_csv(feedback_path)
        combined = pd.concat([base, fb], ignore_index=True)
        # Separate X, y
        feature_cols = [c for c in combined.columns if c != 'prognosis']
        X = combined[feature_cols].values
        y = combined['prognosis'].values
        # Use class_weight balanced
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=400, class_weight='balanced', random_state=42)
        rf.fit(X, y)
        # Persist
        os.makedirs('src/models', exist_ok=True)
        with open(rf_model_path, 'wb') as f:
            pickle.dump({'model': rf, 'feature_order': feature_cols}, f)
        # Update in-memory
        global rf_model, rf_feature_order
        rf_model = rf
        rf_feature_order = feature_cols
        return True, {'classes': sorted(list(set(y)))}
    except Exception as e:
        return False, {'error': str(e)}


def predict_with_feedback_model(patient_symptoms: List[str], top_k: int = 3):
    """Predict using the feedback-trained RF model if loaded."""
    if rf_model is None or not rf_feature_order:
        return None
    # Build feature vector aligned to feature order
    vec = np.zeros(len(rf_feature_order))
    for s in patient_symptoms:
        try:
            idx = rf_feature_order.index(s)
            vec[idx] = 1
        except ValueError:
            continue
    if vec.sum() == 0:
        return {
            'primary_disease': None,
            'confidence': 0.0,
            'top_predictions': [],
            'unknown_symptoms': [s for s in patient_symptoms if s not in rf_feature_order]
        }
    # Predict probabilities if available
    proba = None
    if hasattr(rf_model, 'predict_proba'):
        proba = rf_model.predict_proba([vec])[0]
        classes = rf_model.classes_
        pairs = [(str(classes[i]), float(proba[i])) for i in range(len(classes))]
        pairs.sort(key=lambda x: x[1], reverse=True)
        top = pairs[:max(1, top_k)]
        primary_disease, primary_conf = top[0]
        return {
            'primary_disease': primary_disease,
            'confidence': float(primary_conf),
            'top_predictions': [{'disease': d, 'confidence': c} for d, c in top],
            'unknown_symptoms': [s for s in patient_symptoms if s not in rf_feature_order]
        }
    # Fallback hard prediction
    pred = str(rf_model.predict([vec])[0])
    return {
        'primary_disease': pred,
        'confidence': None,
        'top_predictions': [{'disease': pred, 'confidence': None}],
        'unknown_symptoms': [s for s in patient_symptoms if s not in rf_feature_order]
    }

# Routes
@app.route('/api/predict', methods=['POST'])
def predict():
    import requests as http_requests
    import json, re

    data = request.json
    symptoms = data.get('symptoms', [])
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400

    try:
        # Check which symptoms are known vs unknown
        known = [s for s in symptoms if s in symptoms_dict]
        unknown = [s for s in symptoms if s not in symptoms_dict]
        all_unknown = len(known) == 0

        groq_key = os.environ.get('GROQ_API_KEY', '')

        # --- CASE 1: All symptoms are unknown → go straight to Groq ---
        if all_unknown and groq_key:
            groq_result = _call_groq(groq_key, symptoms, "Unknown")
            if groq_result:
                return jsonify(groq_result), 200

        # --- CASE 2: Known symptoms → run ML model first ---
        pred = predict_with_feedback_model(symptoms) or get_predicted_value(symptoms)
        disease = pred.get('primary_disease')
        ml_confidence = pred.get('confidence') or 0.0

        desc, pre, med, die, wrkout = ("", [], [], [], []) if not disease else helper(disease)

        # Decide if ML result is good enough (confidence >= 0.4 and no unknown symptoms)
        ml_is_good = disease and ml_confidence >= 0.4 and len(unknown) == 0

        if ml_is_good:
            # ML result is confident — still run Groq to validate/correct
            groq_result = _call_groq(groq_key, symptoms, disease) if groq_key else None
            if groq_result:
                return jsonify(groq_result), 200
            # Groq unavailable — return ML result as-is
            return jsonify({
                'disease': disease,
                'confidence': ml_confidence,
                'description': desc,
                'precautions': pre,
                'medications': med,
                'diets': die,
                'workouts': wrkout,
                'unknown_symptoms': unknown
            }), 200
        else:
            # Low confidence or has unknown symptoms → let Groq decide
            groq_result = _call_groq(groq_key, symptoms, disease or "Unknown") if groq_key else None
            if groq_result:
                return jsonify(groq_result), 200
            # Groq unavailable — return whatever ML gave us
            return jsonify({
                'disease': disease,
                'confidence': ml_confidence,
                'description': desc,
                'precautions': pre,
                'medications': med,
                'diets': die,
                'workouts': wrkout,
                'unknown_symptoms': unknown
            }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _call_groq(groq_key, symptoms, ml_prediction):
    """Call Groq and return formatted result dict, or None on failure."""
    import requests as http_requests
    import json, re

    symptom_set = set(s.lower().strip() for s in symptoms)

    prompt = f"""You are an intelligent medical assistant integrated into a disease prediction system.

IMPORTANT RULES:
- Do NOT mention any AI, API, or external service.
- Be confident and direct.

INPUT:
Symptoms: {', '.join(symptoms)}
Initial Prediction: {ml_prediction}

TASK:
1. Analyze the symptoms carefully.
2. Validate whether the initial prediction is correct.
3. If correct keep it. If incorrect replace with a more accurate disease.
4. Prefer common and realistic diseases.

IMPORTANT DECISION RULES:
- General symptoms only (fever, headache, fatigue, body pain, chills) → ALWAYS return "Viral Fever"
- Dengue → ONLY if: pain_behind_the_eyes + skin_rash or red_spots_over_body
- Chicken pox → ONLY if: blister + itching + red_spots_over_body
- Malaria → ONLY if: high_fever + chills + shivering + sweating (all present)
- Typhoid → ONLY if: high_fever + abdominal_pain + constipation
- Jaundice → ONLY if: yellowish_skin or yellowing_of_eyes + dark_urine
- Acne → if: pus_filled_pimples or blackheads
- Allergy → if: continuous_sneezing + runny_nose + watering_from_eyes
- UTI → if: burning_micturition + bladder_discomfort
- Migraine → if: headache + nausea + visual_disturbances
- Asthma → if: breathlessness + cough + chest_pain

OUTPUT FORMAT — STRICT JSON ONLY, no markdown, no extra text:
{{"disease":"Final Disease Name","confidence":"High / Medium / Low","reason":"1-2 line explanation","precautions":["p1","p2","p3"],"medications":["m1","m2","m3"],"diet":["d1","d2"],"workout":["w1","w2"]}}"""

    try:
        resp = http_requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={'Authorization': f'Bearer {groq_key}', 'Content-Type': 'application/json'},
            json={
                'model': 'llama-3.1-8b-instant',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.2,
                'max_tokens': 512
            },
            timeout=15
        )
        if resp.status_code != 200:
            return None

        text = resp.json()['choices'][0]['message']['content'].strip()
        text = text.replace('```json', '').replace('```', '').strip()
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            text = match.group(0)
        parsed = json.loads(text)

        # Normalize keys to match frontend expectations
        disease_name = parsed.get('disease', '')
        desc, pre, med, die, wrkout = helper(disease_name) if disease_name and disease_name != 'Consult a doctor' else ('', [], [], [], [])

        return {
            'disease': disease_name,
            'confidence': parsed.get('confidence', 'Medium'),
            'description': parsed.get('reason', desc),
            'precautions': parsed.get('precautions') or pre or ['Consult a healthcare professional'],
            'medications': parsed.get('medications') or med or ['Consult a doctor'],
            'diets': parsed.get('diet') or die or ['Light meals', 'Stay hydrated'],
            'workouts': parsed.get('workout') or wrkout or ['Rest until symptoms improve'],
            'unknown_symptoms': [s for s in symptoms if s not in symptoms_dict]
        }
    except Exception as e:
        print(f'Groq call failed: {e}')
        return None

@app.route('/api/learn', methods=['POST'])
def learn_from_feedback():
    try:
        data = request.json
        symptoms = data.get('symptoms', [])
        confirmed = data.get('confirmed_disease')
        if not symptoms or not confirmed:
            return jsonify({'error': 'Both symptoms and confirmed_disease are required'}), 400
        append_feedback(symptoms, confirmed)
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def retrain_endpoint():
    ok, info = retrain_feedback_model()
    if ok:
        return jsonify({'success': True, **info}), 200
    return jsonify({'success': False, **info}), 500

@app.route('/api/feedback', methods=['POST'])
def send_feedback():
    try:
        data = request.get_json()
        
        # Extract feedback data
        name = data.get('name', 'Anonymous')
        email = data.get('email', 'No email provided')
        subject = data.get('subject', 'No subject')
        message = data.get('message', 'No message')
        rating = data.get('rating', 5)
        category = data.get('category', 'general')
        
        # Create email content
        email_subject = f"Health AI Feedback: {subject}"
        email_body = f"""
New Feedback from Health AI Website

Name: {name}
Email: {email}
Category: {category}
Rating: {rating}/5
Subject: {subject}

Message:
{message}

---
This feedback was submitted through the Health AI website.
        """
        
        # Try to send email
        email_sent = send_email(email_subject, email_body)
        
        if email_sent:
            return jsonify({
                'success': True,
                'message': 'Feedback sent successfully!'
            }), 200
        else:
            # Fallback: just log to console if email fails
            print("=" * 50)
            print("FEEDBACK EMAIL RECEIVED (Console Log)")
            print("=" * 50)
            print(f"To: {RECIPIENT_EMAIL}")
            print(f"Subject: {email_subject}")
            print(f"Body: {email_body}")
            print("=" * 50)
            
            return jsonify({
                'success': True,
                'message': 'Feedback received! (Email service not configured)'
            }), 200
        
    except Exception as e:
        print(f"Error sending feedback: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to send feedback. Please try again.'
        }), 500

@app.route('/api/enhance', methods=['POST'])
def enhance_with_ai():
    """Enhance/validate ML prediction using Groq → Gemini → rule-based fallback."""
    import requests as http_requests
    import json, re

    data = request.json
    symptoms = data.get('symptoms', [])
    disease = data.get('disease', '')
    symptom_set = set(s.lower().strip() for s in symptoms)

    prompt = f"""You are an intelligent medical assistant integrated into a disease prediction system.

IMPORTANT RULES:
- Do NOT mention any AI, API, or external service.
- Be confident and direct.

INPUT:
Symptoms: {', '.join(symptoms)}
Initial Prediction: {disease}

TASK:
1. Analyze the symptoms carefully.
2. Validate whether the initial prediction is correct.
3. If correct keep it. If incorrect replace with a more accurate disease.
4. Prefer common and realistic diseases.

IMPORTANT DECISION RULES:
- General symptoms only (fever, headache, fatigue, body pain, chills) → ALWAYS return "Viral Fever"
- Dengue → ONLY if: pain_behind_the_eyes + skin_rash or red_spots_over_body
- Chicken pox → ONLY if: blister + itching + red_spots_over_body
- Malaria → ONLY if: high_fever + chills + shivering + sweating (all present)
- Typhoid → ONLY if: high_fever + abdominal_pain + constipation
- Jaundice → ONLY if: yellowish_skin or yellowing_of_eyes + dark_urine
- Acne → if: pus_filled_pimples or blackheads
- Allergy → if: continuous_sneezing + runny_nose + watering_from_eyes
- UTI → if: burning_micturition + bladder_discomfort
- Migraine → if: headache + nausea + visual_disturbances
- Asthma → if: breathlessness + cough + chest_pain

OUTPUT FORMAT — STRICT JSON ONLY, no markdown, no extra text:
{{"disease":"Final Disease Name","confidence":"High / Medium / Low","reason":"1-2 line explanation","precautions":["p1","p2","p3"],"medications":["m1","m2","m3"],"diet":["d1","d2"],"workout":["w1","w2"]}}"""

    def parse_ai_response(text):
        text = text.strip().replace('```json', '').replace('```', '').strip()
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)

    # --- Try Groq first ---
    groq_key = os.environ.get('GROQ_API_KEY', '')
    if groq_key:
        try:
            resp = http_requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers={'Authorization': f'Bearer {groq_key}', 'Content-Type': 'application/json'},
                json={
                    'model': 'llama-3.1-8b-instant',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.2,
                    'max_tokens': 512
                },
                timeout=15
            )
            if resp.status_code == 200:
                text = resp.json()['choices'][0]['message']['content']
                parsed = parse_ai_response(text)
                return jsonify(parsed), 200
        except Exception as e:
            print(f'Groq failed: {e}')

    # --- Try Gemini as backup ---
    gemini_key = os.environ.get('GEMINI_API_KEY', '')
    if gemini_key:
        try:
            resp = http_requests.post(
                f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}',
                json={'contents': [{'parts': [{'text': prompt}]}], 'generationConfig': {'temperature': 0.2, 'maxOutputTokens': 512}},
                timeout=15
            )
            if resp.status_code == 200:
                text = resp.json()['candidates'][0]['content']['parts'][0]['text']
                parsed = parse_ai_response(text)
                return jsonify(parsed), 200
        except Exception as e:
            print(f'Gemini failed: {e}')

    # --- Smart rule-based fallback ---
    general_symptoms = {'fever', 'high_fever', 'mild_fever', 'headache', 'fatigue', 'body_pain',
                        'muscle_pain', 'lethargy', 'malaise', 'sweating', 'chills', 'shivering',
                        'nausea', 'loss_of_appetite', 'weakness_in_limbs'}

    def match(specific, min_count=2):
        return len(symptom_set & specific) >= min_count

    if match({'yellowish_skin', 'yellowing_of_eyes', 'dark_urine', 'yellow_urine'}, 2):
        final_disease, reason, confidence = 'Jaundice', 'Yellowing symptoms and dark urine indicate liver condition.', 'High'
    elif match({'burning_micturition', 'continuous_feel_of_urine', 'bladder_discomfort'}, 2):
        final_disease, reason, confidence = 'Urinary tract infection', 'Burning urination and bladder discomfort are classic UTI signs.', 'High'
    elif len(symptom_set & {'pus_filled_pimples', 'blackheads', 'skin_peeling'}) >= 1:
        final_disease, reason, confidence = 'Acne', 'Skin symptoms like pimples and blackheads indicate acne.', 'High'
    elif match({'continuous_sneezing', 'runny_nose', 'watering_from_eyes'}, 2):
        final_disease, reason, confidence = 'Allergy', 'Sneezing, runny nose and watery eyes are classic allergy symptoms.', 'High'
    elif match({'breathlessness', 'cough', 'chest_pain'}, 2):
        final_disease, reason, confidence = 'Bronchial Asthma', 'Breathlessness and chest symptoms indicate asthma.', 'High'
    elif match({'headache', 'nausea', 'visual_disturbances', 'blurred_and_distorted_vision'}, 2) and 'headache' in symptom_set:
        final_disease, reason, confidence = 'Migraine', 'Severe headache with nausea and visual disturbances suggest migraine.', 'High'
    elif match({'blister', 'skin_rash', 'red_spots_over_body', 'itching'}, 3):
        final_disease, reason, confidence = 'Chicken pox', 'Blister rash with itching are defining chickenpox symptoms.', 'High'
    elif match({'pain_behind_the_eyes', 'red_spots_over_body', 'skin_rash'}, 2):
        final_disease, reason, confidence = 'Dengue', 'Pain behind eyes and rash alongside fever indicate dengue.', 'High'
    elif match({'abdominal_pain', 'constipation', 'high_fever'}, 2):
        final_disease, reason, confidence = 'Typhoid', 'Sustained fever with abdominal pain suggest typhoid.', 'Medium'
    elif match({'high_fever', 'chills', 'shivering', 'sweating'}, 3):
        final_disease, reason, confidence = 'Malaria', 'Cyclic fever with chills and sweating are classic malaria signs.', 'Medium'
    elif symptom_set & {'fever', 'high_fever', 'mild_fever'} and len(symptom_set - general_symptoms) == 0:
        final_disease, reason, confidence = 'Viral Fever', 'General symptoms without specific indicators suggest viral fever.', 'High'
    else:
        final_disease, reason, confidence = disease, f'Symptoms are consistent with {disease}.', 'Medium'

    desc, pre, med, die, wrkout = helper(final_disease) if final_disease and final_disease != 'Consult a doctor' else ('', [], [], [], [])

    return jsonify({
        'disease': final_disease,
        'confidence': confidence,
        'reason': reason,
        'precautions': pre or ['Consult a healthcare professional', 'Rest adequately', 'Stay hydrated'],
        'medications': med or ['Consult a doctor for proper medication'],
        'diet': die or ['Light meals', 'Stay hydrated'],
        'workout': wrkout or ['Rest until symptoms improve']
    }), 200


if __name__ == '__main__':
    app.run(debug=True)

# Vercel serverless function handler
def handler(request):
    return app(request.environ, lambda *args: None)