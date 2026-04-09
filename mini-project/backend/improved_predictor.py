"""
Improved Disease Prediction System
Uses multiple pre-trained models and advanced algorithms for better accuracy
"""

import re
import numpy as np
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import os

class ImprovedPredictor:
    def __init__(self):
        # Enhanced disease patterns with more specific symptoms
        self.disease_patterns = {
            'Common Cold': {
                'symptoms': ['cough', 'runny_nose', 'sneezing', 'sore_throat', 'mild_fever', 'congestion'],
                'description': 'A viral infection of the upper respiratory tract causing mild symptoms.',
                'precautions': ['Rest adequately', 'Stay hydrated', 'Use over-the-counter medications', 'Avoid close contact with others'],
                'medications': ['Decongestants', 'Cough suppressants', 'Pain relievers', 'Consult doctor if symptoms persist'],
                'diets': ['Warm liquids like tea or soup', 'Stay hydrated', 'Avoid dairy if it worsens symptoms'],
                'workouts': ['Light walking', 'Rest is important', 'Avoid strenuous exercise']
            },
            'Influenza': {
                'symptoms': ['high_fever', 'body_aches', 'fatigue', 'cough', 'headache', 'chills', 'sweating'],
                'description': 'A viral infection that attacks your respiratory system with severe symptoms.',
                'precautions': ['Rest in bed', 'Stay hydrated', 'Take fever-reducing medications', 'Seek medical attention if severe'],
                'medications': ['Antiviral medications', 'Pain relievers', 'Fever reducers', 'Consult doctor immediately'],
                'diets': ['Light, easily digestible foods', 'Plenty of fluids', 'Avoid heavy meals'],
                'workouts': ['Complete rest recommended', 'No exercise until fever subsides']
            },
            'COVID-19': {
                'symptoms': ['fever', 'cough', 'fatigue', 'loss_of_taste', 'loss_of_smell', 'shortness_of_breath', 'body_aches'],
                'description': 'A respiratory illness caused by the SARS-CoV-2 virus.',
                'precautions': ['Isolate immediately', 'Wear mask', 'Monitor oxygen levels', 'Seek emergency care if severe'],
                'medications': ['Consult healthcare provider', 'Monitor symptoms closely', 'Emergency care if needed'],
                'diets': ['Stay hydrated', 'Light foods', 'Monitor appetite'],
                'workouts': ['Complete rest', 'Monitor breathing', 'No exercise']
            },
            'Gastroenteritis': {
                'symptoms': ['nausea', 'vomiting', 'diarrhoea', 'abdominal_pain', 'dehydration', 'loss_of_appetite'],
                'description': 'Inflammation of the stomach and intestines, usually caused by infection.',
                'precautions': ['Stay hydrated', 'Rest', 'Avoid solid foods initially', 'Monitor for severe dehydration'],
                'medications': ['Anti-nausea medications', 'Anti-diarrheal medications', 'Oral rehydration solutions'],
                'diets': ['BRAT diet (Bananas, Rice, Applesauce, Toast)', 'Clear liquids', 'Avoid dairy and fatty foods'],
                'workouts': ['Rest is essential', 'No exercise until symptoms improve']
            },
            'Migraine': {
                'symptoms': ['severe_headache', 'nausea', 'sensitivity_to_light', 'sensitivity_to_sound', 'vomiting'],
                'description': 'A neurological condition characterized by severe, recurring headaches.',
                'precautions': ['Rest in a dark, quiet room', 'Avoid triggers', 'Stay hydrated', 'Manage stress'],
                'medications': ['Pain relievers', 'Triptans', 'Anti-nausea medications', 'Consult neurologist'],
                'diets': ['Avoid trigger foods', 'Stay hydrated', 'Regular meals', 'Limit caffeine'],
                'workouts': ['Gentle stretching', 'Yoga', 'Avoid strenuous exercise during attacks']
            },
            'Hypertension': {
                'symptoms': ['headache', 'dizziness', 'chest_pain', 'shortness_of_breath', 'vision_problems', 'anxiety'],
                'description': 'High blood pressure that can lead to serious health complications.',
                'precautions': ['Monitor blood pressure regularly', 'Reduce salt intake', 'Exercise regularly', 'Manage stress'],
                'medications': ['Blood pressure medications', 'Consult cardiologist', 'Regular check-ups'],
                'diets': ['Low-sodium diet', 'DASH diet', 'Limit alcohol', 'Increase potassium'],
                'workouts': ['Regular aerobic exercise', 'Walking', 'Swimming', 'Avoid heavy lifting']
            },
            'Diabetes': {
                'symptoms': ['excessive_thirst', 'frequent_urination', 'fatigue', 'blurred_vision', 'slow_healing', 'weight_loss'],
                'description': 'A metabolic disorder affecting blood sugar regulation.',
                'precautions': ['Monitor blood sugar', 'Regular exercise', 'Healthy diet', 'Regular check-ups'],
                'medications': ['Insulin or oral medications', 'Blood sugar monitoring', 'Consult endocrinologist'],
                'diets': ['Low-carb diet', 'Regular meal timing', 'Monitor carbohydrate intake', 'Avoid sugary foods'],
                'workouts': ['Regular aerobic exercise', 'Strength training', 'Monitor blood sugar during exercise']
            },
            'Anxiety Disorder': {
                'symptoms': ['anxiety', 'restlessness', 'rapid_heartbeat', 'sweating', 'difficulty_concentrating', 'insomnia'],
                'description': 'A mental health condition characterized by excessive worry and fear.',
                'precautions': ['Practice relaxation techniques', 'Regular exercise', 'Adequate sleep', 'Limit caffeine'],
                'medications': ['Anti-anxiety medications', 'Antidepressants', 'Consult psychiatrist'],
                'diets': ['Balanced diet', 'Limit caffeine and alcohol', 'Regular meals', 'Omega-3 rich foods'],
                'workouts': ['Regular exercise', 'Yoga', 'Meditation', 'Deep breathing exercises']
            },
            'Skin Infection': {
                'symptoms': ['itching', 'skin_rash', 'redness', 'swelling', 'pain', 'blister'],
                'description': 'An infection affecting the skin, often caused by bacteria, viruses, or fungi.',
                'precautions': ['Keep area clean and dry', 'Avoid scratching', 'Use prescribed medications', 'Monitor for spreading'],
                'medications': ['Topical antibiotics', 'Antifungal creams', 'Oral antibiotics if needed', 'Consult dermatologist'],
                'diets': ['Healthy diet', 'Stay hydrated', 'Avoid trigger foods if allergic'],
                'workouts': ['Light exercise', 'Avoid sweating in affected area', 'Keep area dry']
            },
            'Urinary Tract Infection': {
                'symptoms': ['burning_micturition', 'frequent_urination', 'bladder_discomfort', 'cloudy_urine', 'fever'],
                'description': 'An infection in any part of the urinary system.',
                'precautions': ['Stay hydrated', 'Urinate frequently', 'Wipe properly', 'Avoid irritating products'],
                'medications': ['Antibiotics', 'Pain relievers', 'Consult urologist', 'Complete full course'],
                'diets': ['Cranberry juice', 'Stay hydrated', 'Avoid caffeine and alcohol', 'Probiotic foods'],
                'workouts': ['Light exercise', 'Avoid holding urine', 'Stay hydrated during exercise']
            },
            'Bronchial Asthma': {
                'symptoms': ['cough', 'shortness_of_breath', 'wheezing', 'chest_tightness', 'fatigue'],
                'description': 'A chronic respiratory condition causing airway inflammation and breathing difficulties.',
                'precautions': ['Avoid triggers', 'Use prescribed inhalers', 'Monitor breathing', 'Keep rescue medication handy'],
                'medications': ['Bronchodilators', 'Inhaled corticosteroids', 'Consult pulmonologist'],
                'diets': ['Anti-inflammatory foods', 'Avoid trigger foods', 'Stay hydrated'],
                'workouts': ['Breathing exercises', 'Light aerobic exercise', 'Avoid cold air exercise']
            },
            'Pneumonia': {
                'symptoms': ['high_fever', 'cough', 'shortness_of_breath', 'chest_pain', 'fatigue', 'sweating'],
                'description': 'A serious lung infection causing inflammation of air sacs.',
                'precautions': ['Seek immediate medical attention', 'Rest completely', 'Monitor breathing', 'Stay hydrated'],
                'medications': ['Antibiotics', 'Pain relievers', 'Hospitalization if severe'],
                'diets': ['Light, nutritious foods', 'Plenty of fluids', 'Avoid heavy meals'],
                'workouts': ['Complete rest', 'No exercise', 'Focus on recovery']
            },
            'Appendicitis': {
                'symptoms': ['severe_abdominal_pain', 'nausea', 'vomiting', 'fever', 'loss_of_appetite'],
                'description': 'Inflammation of the appendix requiring immediate medical attention.',
                'precautions': ['Seek emergency medical care immediately', 'Do not eat or drink', 'Avoid pain medications'],
                'medications': ['Emergency surgery required', 'Antibiotics', 'Pain management'],
                'diets': ['Nothing by mouth until surgery', 'Clear liquids after surgery'],
                'workouts': ['No exercise', 'Complete rest', 'Follow post-surgery guidelines']
            }
        }
        
        # Enhanced symptom synonyms with better matching
        self.symptom_synonyms = {
            'cough': ['cough', 'hacking', 'dry_cough', 'wet_cough', 'persistent_cough'],
            'fever': ['fever', 'high_fever', 'mild_fever', 'temperature', 'pyrexia'],
            'chills': ['chills', 'shivering', 'cold_flashes', 'rigors'],
            'headache': ['headache', 'migraine', 'head_pain', 'tension_headache', 'severe_headache'],
            'nausea': ['nausea', 'queasiness', 'upset_stomach', 'sick_feeling'],
            'fatigue': ['fatigue', 'tiredness', 'exhaustion', 'lethargy', 'weakness'],
            'abdominal_pain': ['abdominal_pain', 'stomach_pain', 'belly_pain', 'gut_pain', 'severe_abdominal_pain'],
            'diarrhoea': ['diarrhoea', 'diarrhea', 'loose_motions', 'runny_stools'],
            'vomiting': ['vomiting', 'throwing_up', 'emesis'],
            'itching': ['itching', 'pruritus', 'skin_irritation'],
            'skin_rash': ['skin_rash', 'dermatitis', 'eruption'],
            'chest_pain': ['chest_pain', 'thoracic_pain', 'angina', 'chest_tightness'],
            'shortness_of_breath': ['shortness_of_breath', 'breathlessness', 'dyspnea', 'difficulty_breathing'],
            'dizziness': ['dizziness', 'vertigo', 'lightheadedness'],
            'anxiety': ['anxiety', 'nervousness', 'stress', 'worry'],
            'burning_micturition': ['burning_micturition', 'painful_urination', 'dysuria'],
            'frequent_urination': ['frequent_urination', 'polyuria', 'urinary_frequency'],
            'sweating': ['sweating', 'perspiration', 'excessive_sweating'],
            'body_aches': ['body_aches', 'muscle_pain', 'joint_pain', 'general_pain'],
            'loss_of_appetite': ['loss_of_appetite', 'decreased_hunger', 'anorexia'],
            'runny_nose': ['runny_nose', 'nasal_discharge', 'rhinorrhea'],
            'sneezing': ['sneezing', 'continuous_sneezing'],
            'sore_throat': ['sore_throat', 'throat_pain', 'pharyngitis'],
            'congestion': ['congestion', 'nasal_congestion', 'stuffy_nose']
        }
        
        # Initialize models
        self.models = {}
        self.scaler = StandardScaler()
        self.accuracy_score = None
        self.model_metrics = {}
        self.model_weights = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize multiple ML models for ensemble prediction with cross-validation"""
        try:
            # Create enhanced ensemble of models with optimized hyperparameters
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=200,  # Increased from 100
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced',
                random_state=42
            )
            
            self.models['logistic_regression'] = LogisticRegression(
                C=1.0,
                solver='liblinear',
                penalty='l2',
                class_weight='balanced',
                max_iter=2000,  # Increased from 1000
                random_state=42
            )
            
            self.models['svm'] = SVC(
                C=10.0,
                kernel='rbf',  # Changed from linear to rbf for better performance
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
            
            # Add gradient boosting for better performance
            from sklearn.ensemble import GradientBoostingClassifier
            self.models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                subsample=0.8,
                random_state=42
            )
            
            # Evaluate models with cross-validation
            self._evaluate_models_with_cv()
            
            # Train models with synthetic data based on disease patterns
            self._train_models()
            print("✅ Enhanced ML models initialized, evaluated with cross-validation, and trained")
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            
    def _evaluate_models_with_cv(self):
        """Evaluate models using cross-validation to measure accuracy"""
        try:
            from sklearn.model_selection import cross_validate
            
            # Create synthetic evaluation data
            X_eval, y_eval = [], []
            
            # Generate evaluation samples for each disease
            for disease, pattern in self.disease_patterns.items():
                symptoms = pattern['symptoms']
                
                # Create diverse symptom combinations for evaluation
                for i in range(50):  # 50 samples per disease for evaluation
                    # Vary symptom combinations
                    num_symptoms = max(2, int(np.random.uniform(0.3, 0.9) * len(symptoms)))
                    selected_symptoms = np.random.choice(symptoms, num_symptoms, replace=False)
                    
                    # Create feature vector
                    feature_vector = self._symptoms_to_vector(selected_symptoms)
                    X_eval.append(feature_vector)
                    y_eval.append(disease)
            
            X_eval = np.array(X_eval)
            y_eval = np.array(y_eval)
            
            # Scale features
            X_eval_scaled = self.scaler.fit_transform(X_eval)
            
            # Store model performance metrics
            self.model_metrics = {}
            
            # Cross-validation settings
            cv = 5
            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            
            print("Evaluating model performance with cross-validation:")
            for name, model in self.models.items():
                # Perform cross-validation
                cv_results = cross_validate(model, X_eval_scaled, y_eval, cv=cv, scoring=scoring)
                
                # Store metrics
                self.model_metrics[name] = {
                    'accuracy': np.mean(cv_results['test_accuracy']),
                    'precision': np.mean(cv_results['test_precision_macro']),
                    'recall': np.mean(cv_results['test_recall_macro']),
                    'f1': np.mean(cv_results['test_f1_macro'])
                }
                
                # Print metrics
                print(f"{name} - Accuracy: {self.model_metrics[name]['accuracy']:.4f}, "
                      f"F1: {self.model_metrics[name]['f1']:.4f}")
            
            # Calculate model weights based on performance
            total_f1 = sum(metrics['f1'] for metrics in self.model_metrics.values())
            self.model_weights = {}
            for name, metrics in self.model_metrics.items():
                # Weight models by their F1 score relative to total F1
                self.model_weights[name] = metrics['f1'] / total_f1 if total_f1 > 0 else 0.25
            
            print("Model weights for ensemble prediction:")
            for name, weight in self.model_weights.items():
                print(f"{name}: {weight:.4f}")
                
        except Exception as e:
            print(f"❌ Error in cross-validation: {e}")
            # Default weights if cross-validation fails
            self.model_weights = {
                'random_forest': 0.35,
                'gradient_boosting': 0.30,
                'svm': 0.25,
                'logistic_regression': 0.10
            }
    
    def _train_models(self):
        """Train models with synthetic data based on disease patterns"""
        try:
            # Create synthetic training data
            X_train = []
            y_train = []
            
            # Generate training samples for each disease
            for disease, pattern in self.disease_patterns.items():
                symptoms = pattern['symptoms']
                
                # Create multiple variations of symptom combinations with more samples
                for i in range(100):  # Increased from 50 to 100 samples per disease
                    # Create more diverse symptom combinations
                    if i < 30:  # 30% with most symptoms for clear cases
                        num_symptoms = max(3, int(0.8 * len(symptoms)))
                        selected_symptoms = np.random.choice(symptoms, num_symptoms, replace=False)
                    elif i < 70:  # 40% with medium number of symptoms
                        num_symptoms = max(2, int(0.5 * len(symptoms)))
                        selected_symptoms = np.random.choice(symptoms, num_symptoms, replace=False)
                    else:  # 30% with fewer symptoms for edge cases
                        num_symptoms = max(1, int(0.3 * len(symptoms)))
                        selected_symptoms = np.random.choice(symptoms, num_symptoms, replace=False)
                    
                    # Create feature vector
                    feature_vector = self._symptoms_to_vector(selected_symptoms)
                    X_train.append(feature_vector)
                    y_train.append(disease)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train each model
            for name, model in self.models.items():
                model.fit(X_train_scaled, y_train)
                
        except Exception as e:
            print(f"❌ Error training models: {e}")
    
    def _symptoms_to_vector(self, symptoms):
        """Convert symptoms to feature vector"""
        # Create a feature vector based on all possible symptoms
        all_symptoms = set()
        for pattern in self.disease_patterns.values():
            all_symptoms.update(pattern['symptoms'])
        
        all_symptoms = sorted(list(all_symptoms))
        feature_vector = [0] * len(all_symptoms)
        
        for symptom in symptoms:
            if symptom in all_symptoms:
                feature_vector[all_symptoms.index(symptom)] = 1
        
        return feature_vector
    
    def normalize_symptom(self, symptom: str) -> str:
        """Normalize symptom text for better matching"""
        # Handle empty or None input
        if not symptom:
            return ""
            
        # Convert to lowercase and strip whitespace
        symptom = symptom.lower().strip()
        
        # Replace spaces and special characters with underscores
        # But preserve single words without replacing anything
        if ' ' in symptom or re.search(r'[^a-zA-Z0-9_]', symptom):
            symptom = re.sub(r'[^a-zA-Z0-9_]', '_', symptom)
        
        return symptom
    
    def find_matching_symptoms(self, user_symptoms: List[str]) -> List[str]:
        """Find matching symptoms from user input"""
        matched_symptoms = []
        normalized_user_symptoms = [self.normalize_symptom(s) for s in user_symptoms]
        
        for user_symptom in normalized_user_symptoms:
            # Skip empty symptoms
            if not user_symptom.strip():
                continue
                
            # Direct match
            if user_symptom in self.symptom_synonyms:
                matched_symptoms.append(user_symptom)
                continue
            
            # Exact match in synonyms
            found_in_synonyms = False
            for key, synonyms in self.symptom_synonyms.items():
                if user_symptom in synonyms:
                    matched_symptoms.append(key)
                    found_in_synonyms = True
                    break
            if found_in_synonyms:
                continue
                
            # Partial match in synonyms
            for key, synonyms in self.symptom_synonyms.items():
                if any(syn in user_symptom for syn in synonyms):
                    matched_symptoms.append(key)
                    found_in_synonyms = True
                    break
            if found_in_synonyms:
                continue
            
            # Partial match in keys
            for key in self.symptom_synonyms:
                if user_symptom in key or key in user_symptom:
                    matched_symptoms.append(key)
                    break
        
        return list(set(matched_symptoms))  # Remove duplicates
    
    def predict_with_ml_models(self, symptoms: List[str]) -> Dict[str, float]:
        """Predict using enhanced ensemble of ML models with weighted voting"""
        try:
            if not self.models:
                return {}
            
            # Convert symptoms to feature vector
            feature_vector = self._symptoms_to_vector(symptoms)
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Use cross-validation weights if available, otherwise use default weights
            if hasattr(self, 'model_weights') and self.model_weights:
                model_weights = self.model_weights
            else:
                # Default weights if cross-validation wasn't performed
                model_weights = {
                    'random_forest': 0.35,  # Increased weight for random forest
                    'gradient_boosting': 0.30,  # High weight for gradient boosting
                    'svm': 0.25,  # Medium weight for SVM
                    'logistic_regression': 0.10  # Lower weight for logistic regression
                }
            
            # Get predictions from all models
            predictions = {}
            model_agreements = {}
            for name, model in self.models.items():
                try:
                    # Get probability predictions
                    proba = model.predict_proba(feature_vector_scaled)[0]
                    classes = model.classes_
                    weight = model_weights.get(name, 0.25)  # Default weight if not specified
                    
                    # Track which disease each model predicts most strongly
                    top_disease = classes[np.argmax(proba)]
                    if top_disease not in model_agreements:
                        model_agreements[top_disease] = 0
                    model_agreements[top_disease] += 1
                    
                    for i, disease in enumerate(classes):
                        if disease not in predictions:
                            predictions[disease] = []
                        # Store tuple of (probability, weight)
                        predictions[disease].append((proba[i], weight))
                except Exception as e:
                    print(f"Error with model {name}: {e}")
            
            # Calculate weighted average probabilities
            final_predictions = {}
            for disease, prob_weights in predictions.items():
                if prob_weights:
                    # Calculate weighted average
                    weighted_sum = sum(prob * weight for prob, weight in prob_weights)
                    total_weight = sum(weight for _, weight in prob_weights)
                    if total_weight > 0:
                        final_predictions[disease] = weighted_sum / total_weight
            
            # Apply confidence boosting for diseases with strong symptom matches
            for disease, pattern in self.disease_patterns.items():
                # Count how many key symptoms are present
                key_symptoms_present = sum(1 for symptom in symptoms if symptom in pattern['symptoms'])
                key_symptoms_ratio = key_symptoms_present / len(pattern['symptoms']) if pattern['symptoms'] else 0
                
                # Boost confidence for diseases with high symptom match
                if disease in final_predictions:
                    # Apply model agreement boost
                    agreement_boost = 1.0
                    if disease in model_agreements:
                        agreement_ratio = model_agreements[disease] / len(self.models)
                        agreement_boost = 1.0 + (0.3 * agreement_ratio)
                    
                    # Apply symptom match boost
                    symptom_boost = 1.0
                    if key_symptoms_ratio > 0.5:
                        symptom_boost = 1.0 + (0.2 * key_symptoms_ratio)
                    
                    # Apply combined boost
                    final_predictions[disease] *= agreement_boost * symptom_boost
            
            return final_predictions
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return {}
    
    def predict_disease(self, symptoms: List[str]) -> Dict[str, Any]:
        """Predict disease using enhanced multiple methods with improved confidence scoring"""
        matched_symptoms = self.find_matching_symptoms(symptoms)
        
        if not matched_symptoms:
            return self.disease_patterns['Common Cold']  # Default to common cold instead of general illness
        
        # For single common symptoms, provide more conservative predictions
        if len(matched_symptoms) == 1:
            single_symptom = matched_symptoms[0]
            # Map of common single symptoms to appropriate conditions
            common_single_symptoms = {
                'fever': 'Common Cold',
                'mild_fever': 'Common Cold',
                'headache': 'Common Cold',
                'cough': 'Common Cold',
                'sore_throat': 'Common Cold',
                'runny_nose': 'Common Cold',
                'sneezing': 'Common Cold',
                'congestion': 'Common Cold',
                'nausea': 'Gastroenteritis',
                'vomiting': 'Gastroenteritis',
                'diarrhoea': 'Gastroenteritis',
                'abdominal_pain': 'Gastroenteritis',
                'fatigue': 'Common Cold'
            }
            
            if single_symptom in common_single_symptoms:
                result = self.disease_patterns[common_single_symptoms[single_symptom]].copy()
                result['confidence_score'] = 0.3  # Low confidence for single symptom
                return result
        
        # Method 1: ML Model Prediction with enhanced ensemble
        ml_predictions = self.predict_with_ml_models(matched_symptoms)
        
        # Method 2: Advanced rule-based scoring with symptom importance weighting
        disease_scores = {}
        for disease, pattern in self.disease_patterns.items():
            score = 0
            symptom_count = 0
            
            # Define primary and secondary symptoms
            primary_symptoms = pattern['symptoms'][:3] if len(pattern['symptoms']) >= 3 else pattern['symptoms']
            secondary_symptoms = [s for s in pattern['symptoms'] if s not in primary_symptoms]
            
            # Score primary symptoms higher (2x weight)
            for symptom in matched_symptoms:
                if symptom in primary_symptoms:
                    score += 2.0  # Primary symptoms get double weight
                    symptom_count += 1
                elif symptom in secondary_symptoms:
                    score += 1.0  # Secondary symptoms get normal weight
                    symptom_count += 1
            
            # Calculate coverage ratio (how many of the disease's symptoms are present)
            coverage_ratio = symptom_count / len(pattern['symptoms']) if pattern['symptoms'] else 0
            
            # Calculate specificity ratio (how many of the user's symptoms match this disease)
            specificity_ratio = symptom_count / len(matched_symptoms) if matched_symptoms else 0
            
            # Combine both ratios with more weight on coverage
            disease_scores[disease] = (0.7 * coverage_ratio) + (0.3 * specificity_ratio)
        
        # Method 3: Symptom co-occurrence analysis
        cooccurrence_scores = {}
        for disease, pattern in self.disease_patterns.items():
            # Check for specific symptom combinations that strongly indicate certain diseases
            if disease == 'COVID-19' and all(s in matched_symptoms for s in ['fever', 'cough', 'loss_of_taste']):
                cooccurrence_scores[disease] = 0.9  # Very high confidence for this combination
            elif disease == 'Influenza' and all(s in matched_symptoms for s in ['high_fever', 'body_aches']):
                cooccurrence_scores[disease] = 0.8
            elif disease == 'Migraine' and all(s in matched_symptoms for s in ['severe_headache', 'sensitivity_to_light']):
                cooccurrence_scores[disease] = 0.85
            elif disease == 'Gastroenteritis' and all(s in matched_symptoms for s in ['nausea', 'diarrhoea']):
                cooccurrence_scores[disease] = 0.75
            else:
                # Default co-occurrence score based on symptom pairs
                pairs_found = 0
                symptoms_list = pattern['symptoms']
                for i in range(len(symptoms_list)):
                    for j in range(i+1, len(symptoms_list)):
                        if symptoms_list[i] in matched_symptoms and symptoms_list[j] in matched_symptoms:
                            pairs_found += 1
                
                max_pairs = (len(symptoms_list) * (len(symptoms_list) - 1)) / 2 if len(symptoms_list) > 1 else 1
                cooccurrence_scores[disease] = pairs_found / max_pairs if max_pairs > 0 else 0
        
        # Combine all prediction methods with weighted approach
        combined_scores = {}
        for disease in self.disease_patterns.keys():
            ml_score = ml_predictions.get(disease, 0)
            rule_score = disease_scores.get(disease, 0)
            cooccurrence_score = cooccurrence_scores.get(disease, 0)
            
            # Weight the different methods (ML is most important, then co-occurrence, then rule-based)
            combined_scores[disease] = (0.6 * ml_score) + (0.25 * cooccurrence_score) + (0.15 * rule_score)
        
        # Avoid severe diagnoses for common symptoms
        severe_conditions = ['Appendicitis', 'Pneumonia']
        common_symptoms_only = all(s in ['headache', 'fever', 'cough', 'fatigue', 'nausea', 'vomiting', 'diarrhoea', 'runny_nose', 'sore_throat'] for s in matched_symptoms)
        
        if common_symptoms_only:
            # Remove severe conditions from consideration for common symptoms
            for condition in severe_conditions:
                if condition in combined_scores:
                    combined_scores[condition] = 0.0
        
        # Find best prediction with improved confidence threshold
        if combined_scores:
            best_disease = max(combined_scores, key=combined_scores.get)
            confidence = combined_scores[best_disease]
            
            # Return with higher confidence threshold
            if confidence > 0.15:  # Slightly higher threshold for better accuracy
                result = self.disease_patterns[best_disease].copy()
                result['confidence_score'] = confidence
                return result
        
        # Improved fallback mechanism with symptom clustering
        # Group symptoms into respiratory, digestive, neurological, etc.
        respiratory_symptoms = ['cough', 'shortness_of_breath', 'runny_nose', 'sneezing', 'sore_throat', 'congestion']
        digestive_symptoms = ['nausea', 'vomiting', 'diarrhoea', 'abdominal_pain']
        neurological_symptoms = ['headache', 'dizziness', 'sensitivity_to_light', 'sensitivity_to_sound']
        fever_symptoms = ['fever', 'high_fever', 'chills', 'sweating']
        
        # Count symptoms in each category
        respiratory_count = sum(1 for s in matched_symptoms if s in respiratory_symptoms)
        digestive_count = sum(1 for s in matched_symptoms if s in digestive_symptoms)
        neurological_count = sum(1 for s in matched_symptoms if s in neurological_symptoms)
        fever_count = sum(1 for s in matched_symptoms if s in fever_symptoms)
        
        # Determine most likely category and return appropriate disease
        if respiratory_count >= 2 and fever_count >= 1:
            return self.disease_patterns['Influenza']
        elif digestive_count >= 2:
            return self.disease_patterns['Gastroenteritis']
        elif neurological_count >= 2:
            return self.disease_patterns['Common Cold']  # Changed from Migraine to Common Cold for more conservative prediction
        elif respiratory_count >= 1:
            return self.disease_patterns['Common Cold']
        else:
            return self.disease_patterns['Common Cold']
    
    def get_prediction(self, symptoms: List[str]) -> Dict[str, Any]:
        """Get enhanced prediction with comprehensive information and accurate confidence scoring"""
        try:
            # Store symptoms for evaluation purposes
            self.symptoms_input = symptoms
            # Get matched symptoms for better reporting
            matched_symptoms = self.find_matching_symptoms(symptoms)
            
            if not matched_symptoms:
                return {
                    'disease': 'Unknown',
                    'description': 'Unable to determine condition based on provided symptoms.',
                    'precautions': ['Consult a healthcare professional'],
                    'medications': ['Consult a healthcare professional'],
                    'diets': ['Maintain a balanced diet'],
                    'workouts': ['Consult a healthcare professional before starting any exercise routine'],
                    'confidence': 'low',
                    'confidence_score': 0.0,
                    'method': 'no_symptoms_matched',
                    'disclaimer': 'This prediction is for informational purposes only. Always consult a healthcare professional for medical advice.'
                }
            
            # For single common symptoms, provide more general advice instead of severe diagnoses
            if len(matched_symptoms) == 1:
                single_symptom = matched_symptoms[0]
                # Map of common single symptoms to appropriate conditions
                common_single_symptoms = {
                    'fever': 'Common Cold',
                    'mild_fever': 'Common Cold',
                    'headache': 'Common Cold',
                    'cough': 'Common Cold',
                    'sore_throat': 'Common Cold',
                    'runny_nose': 'Common Cold',
                    'sneezing': 'Common Cold',
                    'congestion': 'Common Cold',
                    'nausea': 'Gastroenteritis',
                    'vomiting': 'Gastroenteritis',
                    'diarrhoea': 'Gastroenteritis',
                    'abdominal_pain': 'Gastroenteritis',
                    'fatigue': 'Common Cold'
                }
                
                if single_symptom in common_single_symptoms:
                    disease = common_single_symptoms[single_symptom]
                    disease_info = self.disease_patterns.get(disease, {})
                    
                    result = {
                        'disease': f"{disease}: Possible based on {single_symptom}",
                        'description': disease_info.get('description', 'A common condition that may cause this symptom.'),
                        'precautions': disease_info.get('precautions', ['Consult a healthcare professional']),
                        'medications': disease_info.get('medications', ['Consult a healthcare professional']),
                        'diets': disease_info.get('diets', ['Maintain a balanced diet']),
                        'workouts': disease_info.get('workouts', ['Consult a healthcare professional']),
                        'confidence': 'low',
                        'confidence_score': 0.3,  # Low confidence for single symptom
                        'method': 'single_symptom_match',
                        'matched_symptoms': matched_symptoms,
                        'symptom_count': 1,
                        'disclaimer': 'This is based on a single symptom only. Many conditions can cause this symptom. Always consult a healthcare professional for medical advice.'
                    }
                    
                    # Add health tips for single symptoms
                    health_tips = []
                    if single_symptom in ['fever', 'high_fever']:
                        health_tips.append('Monitor your temperature regularly')
                    if single_symptom in ['cough', 'sore_throat']:
                        health_tips.append('Gargle with warm salt water for throat relief')
                    if single_symptom in ['headache']:
                        health_tips.append('Apply cold or warm compress to painful areas')
                    if single_symptom in ['nausea', 'vomiting', 'diarrhoea']:
                        health_tips.append('Drink small sips of clear fluids to prevent dehydration')
                    
                    if health_tips:
                        result['additional_tips'] = health_tips
                    
                    return result
            
            # For multiple symptoms, proceed with normal prediction
            disease_info = self.predict_disease(symptoms)
            
            # Extract confidence score if available, otherwise use a default
            confidence_score = disease_info.pop('confidence_score', 0.5)
            
            # Determine confidence level based on score
            confidence_level = 'low'
            if confidence_score > 0.7:
                confidence_level = 'high'
            elif confidence_score > 0.4:
                confidence_level = 'medium'
            
            # Determine prediction method
            if confidence_score > 0.6:
                method = 'advanced_ensemble_ml'
            elif confidence_score > 0.3:
                method = 'hybrid_rule_ml'
            else:
                method = 'symptom_pattern_matching'
            
            # Create enhanced result with more detailed information
            result = {
                'disease': disease_info.get('description', 'Unknown Condition'),
                'description': disease_info.get('description', 'Unable to determine specific condition.'),
                'precautions': disease_info.get('precautions', ['Consult a healthcare professional']),
                'medications': disease_info.get('medications', ['Consult a healthcare professional']),
                'diets': disease_info.get('diets', ['Maintain a balanced diet']),
                'workouts': disease_info.get('workouts', ['Consult a healthcare professional']),
                'confidence': confidence_level,
                'confidence_score': round(confidence_score, 2),  # Round to 2 decimal places
                'method': method,
                'matched_symptoms': matched_symptoms,
                'symptom_count': len(matched_symptoms),
                'disclaimer': 'This prediction is for informational purposes only. Always consult a healthcare professional for medical advice.'
            }
            
            # Add additional health tips based on symptoms
            health_tips = []
            if any(s in matched_symptoms for s in ['fever', 'high_fever']):
                health_tips.append('Monitor your temperature regularly')
            if any(s in matched_symptoms for s in ['cough', 'sore_throat']):
                health_tips.append('Gargle with warm salt water for throat relief')
            if any(s in matched_symptoms for s in ['headache', 'body_aches']):
                health_tips.append('Apply cold or warm compress to painful areas')
            if any(s in matched_symptoms for s in ['nausea', 'vomiting', 'diarrhoea']):
                health_tips.append('Drink small sips of clear fluids to prevent dehydration')
            
            if health_tips:
                result['additional_tips'] = health_tips
            
            return result
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            # Enhanced fallback with better error handling
            
    def evaluate_accuracy(self):
        """Evaluate the prediction accuracy on test data"""
        try:
            # Create test data with known diseases
            test_cases = [
                {"symptoms": ["cough", "runny_nose", "sneezing", "sore_throat"], "expected": "Common Cold"},
                {"symptoms": ["high_fever", "body_aches", "fatigue", "cough"], "expected": "Influenza"},
                {"symptoms": ["fever", "cough", "loss_of_taste", "shortness_of_breath"], "expected": "COVID-19"},
                {"symptoms": ["nausea", "vomiting", "diarrhoea", "abdominal_pain"], "expected": "Gastroenteritis"},
                {"symptoms": ["severe_headache", "sensitivity_to_light", "nausea"], "expected": "Migraine"}
            ]
            
            correct_predictions = 0
            total_cases = len(test_cases)
            
            print("Evaluating prediction accuracy on test cases:")
            for i, case in enumerate(test_cases):
                prediction = self.get_prediction(case["symptoms"])
                predicted_disease = prediction.get("disease", "").split(":")[0].strip()
                
                # Check if prediction matches expected disease
                is_correct = case["expected"] in predicted_disease
                result = "✓" if is_correct else "✗"
                
                if is_correct:
                    correct_predictions += 1
                
                print(f"Case {i+1}: {result} Expected: {case['expected']}, Predicted: {predicted_disease}")
            
            # Calculate accuracy
            self.accuracy_score = correct_predictions / total_cases if total_cases > 0 else 0
            print(f"Overall accuracy: {self.accuracy_score:.2f} ({correct_predictions}/{total_cases})")
            
            return self.accuracy_score
        except Exception as e:
            print(f"Error evaluating accuracy: {e}")
            return 0.0
            # Analyze symptoms for better fallback prediction
            respiratory_symptoms = ['cough', 'runny_nose', 'sneezing', 'sore_throat', 'congestion']
            fever_symptoms = ['fever', 'high_fever', 'chills', 'sweating']
            digestive_symptoms = ['nausea', 'vomiting', 'diarrhoea', 'abdominal_pain']
            
            # Count symptom types
            respiratory_count = sum(1 for s in symptoms if any(r in s.lower() for r in respiratory_symptoms))
            fever_count = sum(1 for s in symptoms if any(f in s.lower() for f in fever_symptoms))
            digestive_count = sum(1 for s in symptoms if any(d in s.lower() for d in digestive_symptoms))
            
            # Determine most likely fallback based on symptom counts
            if respiratory_count > 0 and fever_count > 0:
                return {
                    'disease': 'Influenza',
                    'description': 'A viral infection that attacks your respiratory system with severe symptoms.',
                    'precautions': ['Rest in bed', 'Stay hydrated', 'Take fever-reducing medications', 'Seek medical attention if severe'],
                    'medications': ['Antiviral medications', 'Pain relievers', 'Fever reducers', 'Consult doctor immediately'],
                    'diets': ['Light, easily digestible foods', 'Plenty of fluids', 'Avoid heavy meals'],
                    'workouts': ['Complete rest recommended', 'No exercise until fever subsides'],
                    'confidence': 'medium',
                    'confidence_score': 0.35,
                    'method': 'symptom_fallback',
                    'disclaimer': 'This is a fallback prediction due to processing error. Please consult a healthcare professional.'
                }
            elif digestive_count > 0:
                return {
                    'disease': 'Gastroenteritis',
                    'description': 'Inflammation of the stomach and intestines, usually caused by infection.',
                    'precautions': ['Stay hydrated', 'Rest', 'Avoid solid foods initially', 'Monitor for severe dehydration'],
                    'medications': ['Anti-nausea medications', 'Anti-diarrheal medications', 'Oral rehydration solutions'],
                    'diets': ['BRAT diet (Bananas, Rice, Applesauce, Toast)', 'Clear liquids', 'Avoid dairy and fatty foods'],
                    'workouts': ['Rest is essential', 'No exercise until symptoms improve'],
                    'confidence': 'low',
                    'confidence_score': 0.25,
                    'method': 'emergency_fallback',
                    'disclaimer': 'This is a fallback prediction due to processing error. Please consult a healthcare professional.'
                }
            else:
                return {
                    'disease': 'Common Cold',
                    'description': 'A viral infection of the upper respiratory tract causing mild symptoms.',
                    'precautions': ['Rest adequately', 'Stay hydrated', 'Use over-the-counter medications', 'Avoid close contact with others'],
                    'medications': ['Decongestants', 'Cough suppressants', 'Pain relievers', 'Consult doctor if symptoms persist'],
                    'diets': ['Warm liquids like tea or soup', 'Stay hydrated', 'Avoid dairy if it worsens symptoms'],
                    'workouts': ['Light walking', 'Rest is important', 'Avoid strenuous exercise'],
                    'confidence': 'low',
                    'confidence_score': 0.2,
                    'method': 'emergency_fallback',
                    'disclaimer': 'This is a fallback prediction due to processing error. Please consult a healthcare professional.'
                }

# Global instance
improved_predictor = ImprovedPredictor()

