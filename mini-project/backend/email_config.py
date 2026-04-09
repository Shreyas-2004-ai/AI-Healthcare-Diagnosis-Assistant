# Email Configuration for Health AI Feedback System
# Replace these values with your actual Gmail credentials

# Gmail SMTP Settings
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
# Alternative settings for regular password
EMAIL_HOST_ALT = 'smtp.gmail.com'
EMAIL_PORT_ALT = 465

# Your Gmail credentials (replace with your actual email)
EMAIL_USER = 'shreyasssanil62@gmail.com'  # Your Gmail address
EMAIL_PASSWORD = 'Shreyas@2004'  # Your Gmail app password

# Recipient email (where feedback will be sent)
RECIPIENT_EMAIL = 'shreyasssanil62@gmail.com'

# Instructions for setting up Gmail App Password:
# 1. Go to your Google Account settings
# 2. Enable 2-Step Verification if not already enabled
# 3. Go to Security > App passwords
# 4. Generate a new app password for "Mail"
# 5. Use that 16-character password in EMAIL_PASSWORD above
# 6. Make sure to use your actual Gmail address in EMAIL_USER

# Note: Never commit your actual email credentials to version control!
# For production, use environment variables instead.
