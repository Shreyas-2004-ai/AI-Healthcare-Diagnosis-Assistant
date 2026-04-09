# Health AI - Symptom Prediction System

A comprehensive health prediction system that uses machine learning to predict diseases based on symptoms and provides detailed health insights including descriptions, precautions, medications, dietary suggestions, and wellness tips.

## Features

- **AI-Powered Disease Prediction**: Uses a pre-trained Support Vector Classifier (SVC) model
- **Comprehensive Health Insights**: Provides detailed information about predicted conditions
- **User-Friendly Interface**: Intuitive symptom input with suggestions
- **Error-Free Operation**: Robust error handling and validation
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Suggestions**: Symptom autocomplete for better user experience

## Technology Stack

### Backend
- **Python 3.8+**
- **Flask**: Web framework
- **Flask-CORS**: Cross-origin resource sharing
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning (SVC model)
- **NumPy**: Numerical computing

### Frontend
- **React.js**: User interface
- **Axios**: HTTP client
- **CSS3**: Styling with modern design

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Node.js 14 or higher
- npm or yarn

### Backend Setup

1. **Navigate to the project directory**:
   ```bash
   cd mini-project
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Flask backend server**:
   ```bash
   python main.py
   ```
   
   The backend will start on `http://127.0.0.1:5000`

### Frontend Setup

1. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

2. **Start the React development server**:
   ```bash
   npm start
   ```
   
   The frontend will start on `http://localhost:3000`

## How to Use

1. **Open the application** in your browser at `http://localhost:3000`

2. **Enter Symptoms**:
   - Type symptoms in the input field
   - Use common symptom names like "headache", "fever", "cough"
   - Separate multiple symptoms with commas
   - Use the autocomplete suggestions for better accuracy

3. **Get Predictions**:
   - Click "Get Health Insights" to analyze your symptoms
   - View the predicted condition and confidence level

4. **Explore Health Information**:
   - **Description**: Detailed explanation of the condition
   - **Precautions**: Preventive measures and safety tips
   - **Medications**: Common treatment options
   - **Dietary Suggestions**: Nutritional recommendations
   - **Wellness Tips**: Lifestyle and exercise advice

## Supported Symptoms

The system recognizes a wide range of symptoms including:

- **General**: fever, fatigue, headache, dizziness
- **Respiratory**: cough, breathlessness, chest pain
- **Digestive**: nausea, vomiting, abdominal pain, diarrhea
- **Skin**: itching, skin rash, blister, acne
- **Neurological**: confusion, muscle weakness, joint pain
- **And many more...**

## Model Information

- **Algorithm**: Support Vector Classifier (SVC)
- **Training Data**: Comprehensive dataset with 132 symptoms and 41 diseases
- **Accuracy**: High accuracy for common conditions
- **Diseases Covered**: 41 different medical conditions

## Important Notes

⚠️ **Medical Disclaimer**: This system provides AI-powered predictions for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## Troubleshooting

### Common Issues

1. **Backend Connection Error**:
   - Ensure the Flask server is running on port 5000
   - Check if the port is not occupied by another application

2. **Model Loading Error**:
   - Verify that `src/models/svc.pkl` exists
   - Ensure all dataset files are present in `src/datasets/`

3. **Symptom Not Recognized**:
   - Use the exact symptom names from the suggestions
   - Check the symptoms list for proper formatting

### Error Messages

- **"Unable to connect to prediction service"**: Backend server is not running
- **"No symptoms provided"**: Please enter at least one symptom
- **"Prediction failed"**: Check backend logs for detailed error information

## File Structure

```
mini-project/
├── main.py                 # Flask backend server
├── requirements.txt        # Python dependencies
├── package.json           # Node.js dependencies
├── src/
│   ├── models/
│   │   └── svc.pkl        # Pre-trained SVC model
│   ├── datasets/          # Training and reference data
│   └── components/        # React components
└── README.md             # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational purposes. Please ensure compliance with local regulations regarding medical software.

## Support

For technical support or questions, please check the troubleshooting section above or create an issue in the repository.
