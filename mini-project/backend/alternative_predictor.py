"""
Alternative Disease Prediction System
This module provides a rule-based prediction system that can work without the pre-trained model.
"""

import re
from typing import List, Dict, Any

class AlternativePredictor:
    def __init__(self):
        # Disease patterns based on symptom combinations
        self.disease_patterns = {
            'Common Cold': {
                'symptoms': ['cough', 'runny_nose', 'sneezing', 'sore_throat', 'mild_fever'],
                'description': 'A viral infection of the upper respiratory tract.',
                'precautions': ['Rest adequately', 'Stay hydrated', 'Use over-the-counter medications', 'Avoid close contact with others'],
                'medications': ['Decongestants', 'Cough suppressants', 'Pain relievers', 'Consult doctor if symptoms persist'],
                'diets': ['Warm liquids like tea or soup', 'Stay hydrated', 'Avoid dairy if it worsens symptoms'],
                'workouts': ['Light walking', 'Rest is important', 'Avoid strenuous exercise']
            },
            'Influenza': {
                'symptoms': ['high_fever', 'body_aches', 'fatigue', 'cough', 'headache'],
                'description': 'A viral infection that attacks your respiratory system.',
                'precautions': ['Rest in bed', 'Stay hydrated', 'Take fever-reducing medications', 'Seek medical attention if severe'],
                'medications': ['Antiviral medications', 'Pain relievers', 'Fever reducers', 'Consult doctor immediately'],
                'diets': ['Light, easily digestible foods', 'Plenty of fluids', 'Avoid heavy meals'],
                'workouts': ['Complete rest recommended', 'No exercise until fever subsides']
            },
            'Gastroenteritis': {
                'symptoms': ['nausea', 'vomiting', 'diarrhoea', 'abdominal_pain', 'dehydration'],
                'description': 'Inflammation of the stomach and intestines, usually caused by infection.',
                'precautions': ['Stay hydrated', 'Rest', 'Avoid solid foods initially', 'Monitor for severe dehydration'],
                'medications': ['Anti-nausea medications', 'Anti-diarrheal medications', 'Oral rehydration solutions'],
                'diets': ['BRAT diet (Bananas, Rice, Applesauce, Toast)', 'Clear liquids', 'Avoid dairy and fatty foods'],
                'workouts': ['Rest is essential', 'No exercise until symptoms improve']
            },
            'Migraine': {
                'symptoms': ['severe_headache', 'nausea', 'sensitivity_to_light', 'sensitivity_to_sound'],
                'description': 'A neurological condition characterized by severe, recurring headaches.',
                'precautions': ['Rest in a dark, quiet room', 'Avoid triggers', 'Stay hydrated', 'Manage stress'],
                'medications': ['Pain relievers', 'Triptans', 'Anti-nausea medications', 'Consult neurologist'],
                'diets': ['Avoid trigger foods', 'Stay hydrated', 'Regular meals', 'Limit caffeine'],
                'workouts': ['Gentle stretching', 'Yoga', 'Avoid strenuous exercise during attacks']
            },
            'Hypertension': {
                'symptoms': ['headache', 'dizziness', 'chest_pain', 'shortness_of_breath', 'vision_problems'],
                'description': 'High blood pressure that can lead to serious health complications.',
                'precautions': ['Monitor blood pressure regularly', 'Reduce salt intake', 'Exercise regularly', 'Manage stress'],
                'medications': ['Blood pressure medications', 'Consult cardiologist', 'Regular check-ups'],
                'diets': ['Low-sodium diet', 'DASH diet', 'Limit alcohol', 'Increase potassium'],
                'workouts': ['Regular aerobic exercise', 'Walking', 'Swimming', 'Avoid heavy lifting']
            },
            'Diabetes': {
                'symptoms': ['excessive_thirst', 'frequent_urination', 'fatigue', 'blurred_vision', 'slow_healing'],
                'description': 'A metabolic disorder affecting blood sugar regulation.',
                'precautions': ['Monitor blood sugar', 'Regular exercise', 'Healthy diet', 'Regular check-ups'],
                'medications': ['Insulin or oral medications', 'Blood sugar monitoring', 'Consult endocrinologist'],
                'diets': ['Low-carb diet', 'Regular meal timing', 'Monitor carbohydrate intake', 'Avoid sugary foods'],
                'workouts': ['Regular aerobic exercise', 'Strength training', 'Monitor blood sugar during exercise']
            },
            'Anxiety Disorder': {
                'symptoms': ['anxiety', 'restlessness', 'rapid_heartbeat', 'sweating', 'difficulty_concentrating'],
                'description': 'A mental health condition characterized by excessive worry and fear.',
                'precautions': ['Practice relaxation techniques', 'Regular exercise', 'Adequate sleep', 'Limit caffeine'],
                'medications': ['Anti-anxiety medications', 'Antidepressants', 'Consult psychiatrist'],
                'diets': ['Balanced diet', 'Limit caffeine and alcohol', 'Regular meals', 'Omega-3 rich foods'],
                'workouts': ['Regular exercise', 'Yoga', 'Meditation', 'Deep breathing exercises']
            },
            'Skin Infection': {
                'symptoms': ['itching', 'skin_rash', 'redness', 'swelling', 'pain'],
                'description': 'An infection affecting the skin, often caused by bacteria, viruses, or fungi.',
                'precautions': ['Keep area clean and dry', 'Avoid scratching', 'Use prescribed medications', 'Monitor for spreading'],
                'medications': ['Topical antibiotics', 'Antifungal creams', 'Oral antibiotics if needed', 'Consult dermatologist'],
                'diets': ['Healthy diet', 'Stay hydrated', 'Avoid trigger foods if allergic'],
                'workouts': ['Light exercise', 'Avoid sweating in affected area', 'Keep area dry']
            },
            'Urinary Tract Infection': {
                'symptoms': ['burning_micturition', 'frequent_urination', 'bladder_discomfort', 'cloudy_urine'],
                'description': 'An infection in any part of the urinary system.',
                'precautions': ['Stay hydrated', 'Urinate frequently', 'Wipe properly', 'Avoid irritating products'],
                'medications': ['Antibiotics', 'Pain relievers', 'Consult urologist', 'Complete full course'],
                'diets': ['Cranberry juice', 'Stay hydrated', 'Avoid caffeine and alcohol', 'Probiotic foods'],
                'workouts': ['Light exercise', 'Avoid holding urine', 'Stay hydrated during exercise']
            },
            'General Illness': {
                'symptoms': ['fatigue', 'mild_fever', 'body_aches', 'loss_of_appetite'],
                'description': 'A general feeling of being unwell, often due to viral infection or stress.',
                'precautions': ['Rest adequately', 'Stay hydrated', 'Monitor symptoms', 'Avoid strenuous activities'],
                'medications': ['Over-the-counter pain relievers', 'Consult doctor if symptoms persist'],
                'diets': ['Light, easily digestible foods', 'Plenty of fluids', 'Avoid heavy meals'],
                'workouts': ['Gentle stretching', 'Light walking if tolerated', 'Rest when needed']
            }
        }
        
        # Symptom synonyms for better matching
        self.symptom_synonyms = {
            'cough': ['cough', 'hacking', 'dry_cough', 'wet_cough'],
            'fever': ['fever', 'high_fever', 'mild_fever', 'temperature'],
            'headache': ['headache', 'migraine', 'head_pain', 'tension_headache'],
            'nausea': ['nausea', 'queasiness', 'upset_stomach', 'sick_feeling'],
            'fatigue': ['fatigue', 'tiredness', 'exhaustion', 'lethargy'],
            'abdominal_pain': ['abdominal_pain', 'stomach_pain', 'belly_pain', 'gut_pain'],
            'diarrhoea': ['diarrhoea', 'diarrhea', 'loose_motions', 'runny_stools'],
            'vomiting': ['vomiting', 'throwing_up', 'emesis'],
            'itching': ['itching', 'pruritus', 'skin_irritation'],
            'skin_rash': ['skin_rash', 'dermatitis', 'eruption'],
            'chest_pain': ['chest_pain', 'thoracic_pain', 'angina'],
            'shortness_of_breath': ['shortness_of_breath', 'breathlessness', 'dyspnea'],
            'dizziness': ['dizziness', 'vertigo', 'lightheadedness'],
            'anxiety': ['anxiety', 'nervousness', 'stress', 'worry'],
            'burning_micturition': ['burning_micturition', 'painful_urination', 'dysuria'],
            'frequent_urination': ['frequent_urination', 'polyuria', 'urinary_frequency']
        }

    def normalize_symptom(self, symptom: str) -> str:
        """Normalize symptom text for better matching"""
        symptom = symptom.lower().strip()
        symptom = re.sub(r'[^a-zA-Z0-9_]', '_', symptom)
        return symptom

    def find_matching_symptoms(self, user_symptoms: List[str]) -> List[str]:
        """Find matching symptoms from user input"""
        matched_symptoms = []
        normalized_user_symptoms = [self.normalize_symptom(s) for s in user_symptoms]
        
        for user_symptom in normalized_user_symptoms:
            # Direct match
            if user_symptom in self.symptom_synonyms:
                matched_symptoms.append(user_symptom)
                continue
            
            # Synonym match
            for key, synonyms in self.symptom_synonyms.items():
                if user_symptom in synonyms or any(syn in user_symptom for syn in synonyms):
                    matched_symptoms.append(key)
                    break
            
            # Partial match
            for key in self.symptom_synonyms:
                if user_symptom in key or key in user_symptom:
                    matched_symptoms.append(key)
                    break
        
        return list(set(matched_symptoms))  # Remove duplicates

    def predict_disease(self, symptoms: List[str]) -> Dict[str, Any]:
        """Predict disease based on symptoms using rule-based logic"""
        matched_symptoms = self.find_matching_symptoms(symptoms)
        
        if not matched_symptoms:
            return self.disease_patterns['General Illness']
        
        # Calculate scores for each disease
        disease_scores = {}
        for disease, pattern in self.disease_patterns.items():
            score = 0
            for symptom in matched_symptoms:
                if symptom in pattern['symptoms']:
                    score += 1
            
            # Normalize score by number of symptoms
            if pattern['symptoms']:
                score = score / len(pattern['symptoms'])
            
            disease_scores[disease] = score
        
        # Find the disease with highest score
        best_disease = max(disease_scores, key=disease_scores.get)
        
        # If no good match, return general illness
        if disease_scores[best_disease] < 0.3:
            return self.disease_patterns['General Illness']
        
        return self.disease_patterns[best_disease]

    def get_prediction(self, symptoms: List[str]) -> Dict[str, Any]:
        """Get complete prediction with all information"""
        try:
            disease_info = self.predict_disease(symptoms)
            
            return {
                'disease': disease_info.get('description', 'Unknown Condition'),
                'description': disease_info.get('description', 'Unable to determine specific condition.'),
                'precautions': disease_info.get('precautions', ['Consult a healthcare professional']),
                'medications': disease_info.get('medications', ['Consult a healthcare professional']),
                'diets': disease_info.get('diets', ['Maintain a balanced diet']),
                'workouts': disease_info.get('workouts', ['Consult a healthcare professional']),
                'confidence': 'medium',
                'method': 'rule_based'
            }
        except Exception as e:
            # Fallback to general illness
            return {
                'disease': 'General Illness',
                'description': 'Based on your symptoms, you may be experiencing a general illness. Please consult a healthcare professional for proper diagnosis.',
                'precautions': ['Rest adequately', 'Stay hydrated', 'Monitor your symptoms', 'Consult a doctor if symptoms persist'],
                'medications': ['Over-the-counter pain relievers if needed', 'Consult a doctor for persistent symptoms'],
                'diets': ['Light, easily digestible foods', 'Plenty of fluids', 'Avoid heavy meals'],
                'workouts': ['Gentle stretching', 'Light walking if tolerated', 'Rest when needed'],
                'confidence': 'low',
                'method': 'fallback'
            }

# Global instance
alternative_predictor = AlternativePredictor()

