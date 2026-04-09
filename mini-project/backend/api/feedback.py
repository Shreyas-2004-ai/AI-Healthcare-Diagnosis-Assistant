from flask import Flask, request, jsonify
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# Flask app
app = Flask(__name__)

# Email configuration from environment variables
EMAIL_HOST = os.environ.get('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', '587'))
EMAIL_USER = os.environ.get('EMAIL_USER', 'your-email@gmail.com')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'your-app-password')
RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL', 'shreyasssanil62@gmail.com')

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

# Vercel serverless function handler
def handler(request):
    return app(request.environ, lambda *args: None)
