from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
from flask_cors import CORS
import os
from typing import List
import difflib

# Flask app
app = Flask(__name__)
CORS(app)

# Resolve paths relative to this file so it works both locally and on Vercel
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
DATA_DIR = os.path.join(BASE_DIR, "src", "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "src", "models")

# Load datasets
sym_des = pd.read_csv(os.path.join(DATA_DIR, "symtoms_df.csv"))
precautions = pd.read_csv(os.path.join(DATA_DIR, "precautions_df.csv"))
workout = pd.read_csv(os.path.join(DATA_DIR, "workout_df.csv"))
description = pd.read_csv(os.path.join(DATA_DIR, "description.csv"))
medications = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
diets = pd.read_csv(os.path.join(DATA_DIR, "diets.csv"))

# Load model
svc = pickle.load(open(os.path.join(MODEL_DIR, "svc.pkl"), 'rb'))

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

# Symptom dictionary
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

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

# Routes
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms', [])
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400
    try:
        pred = get_predicted_value(symptoms)
        disease = pred.get('primary_disease')
        # In low-confidence or unknown cases, we can still respond with top suggestions
        desc, pre, med, die, wrkout = ("", [], [], [], []) if not disease else helper(disease)
        response_data = {
            'disease': disease,
            'confidence': pred.get('confidence'),
            'top_predictions': pred.get('top_predictions'),
            'unknown_symptoms': pred.get('unknown_symptoms'),
            'description': desc,
            'precautions': pre,
            'medications': med,
            'diets': die,
            'workouts': wrkout
        }
        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Vercel serverless function handler
def handler(request):
    return app(request.environ, lambda *args: None)
