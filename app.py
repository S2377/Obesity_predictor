from flask import Flask, request, jsonify, render_template, session, url_for, redirect, make_response
import joblib
import datetime
import numpy as np
from joblib import load
import plotly.graph_objects as go
from authlib.integrations.requests_client import OAuth2Session
from authlib.integrations.flask_client import OAuth
import os
from flask_mail import Mail, Message
from email.mime.image import MIMEImage
import warnings

# -------------------------------------------------------------Global Variable--------------------------------------------------------
# current date
x = datetime.datetime.now()
app = Flask(__name__, static_folder='static')
# model = joblib.load('Model/RandomForestClassifier.pkl')
user_email_lst = []
user_name_lst = []
NObeyesdad_labels = {
    0: 'Underweight',
    1: 'Normal Weight',
    2: 'Obesity Type I',
    3: 'Obesity Type II',
    4: 'Obesity Type III',
    5: 'Overweight Level I',
    6: 'Overweight Level II'
}

def user_bmi(h, w):
    """Calculate BMI based on height and weight."""
    
    return w / (h ** 2)

def insert_data(email, name):
    """Insert user data into the global lists."""
    
    if email not in user_email_lst:
        user_email_lst.append(email)
        user_name_lst.append(name)
    print(user_email_lst, user_name_lst)


#----------------------------------------------------------Authentication---------------------------------------------------------------
# OAuth configuration
app.secret_key = os.environ.get('SECRET_KEY')
app.config['PERMANENT_SESSION_LIFETIME'] = 3600*24*7

oauth = OAuth(app)
client_id = '630606159234-p9u62ohjjdoqsngd5pgd7ob2salaivio.apps.googleusercontent.com'
client_secret = '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'
redirect_uri = 'http://localhost:5000/loginnew'

google = oauth.register(
    name='google',
    client_id=client_id,
    client_secret=client_secret,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    refresh_token_url=None,
    refresh_token_params=None,
    redirect_uri='http://127.0.0.1:5000/authorize',
    client_kwargs={'scope': 'openid email profile'},
    jwks_uri='https://www.googleapis.com/oauth2/v3/certs',
)

@app.route('/login')
def login():
    """Redirect to Google authorization."""
    
    redirect_uri = url_for('authorize', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/authorize')
def authorize():
    """Authorize and store user information."""
    
    token = google.authorize_access_token()
    session['token'] = token
    return redirect(url_for('profile'))

@app.route('/profile')
def profile():
    """Fetch user profile information."""
    
    token = session.get('token')
    if token is None:
        return redirect(url_for('login'))

    client_id = '630606159234-p9u62ohjjdoqsngd5pgd7ob2salaivio.apps.googleusercontent.com',
    oauth = OAuth2Session(client_id, token=token)
    user_info = oauth.get('https://www.googleapis.com/oauth2/v3/userinfo').json()

    insert_data(user_info['email'], user_info['name'])
    session['user_info'] = user_info
    return render_template('index.html', user_info=user_info)

@app.route('/logout')
def logout():
    """Logout user by removing session data."""
    
    session.pop('user_info', None)
    return render_template("index.html")




# -----------------------------------------------------------------------Mail------------------------------------------------------------
# Mail configuration
app.config["MAIL_SERVER"] = 'smtp.gmail.com'
app.config["MAIL_PORT"] = 465
app.config["MAIL_USERNAME"] = 'malviyaji6281@gmail.com'  # Update with your Gmail email
app.config['MAIL_PASSWORD'] = 'uswo thvi ogxi kkbi'  # Update with your Gmail password
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail = Mail(app)

def send_otp(email, h, w, predicted_classes):
    """Send health report email to the user."""
    
    msg = Message(subject='Your Health Report', sender=('Rajat Malviya', app.config["MAIL_USERNAME"]), recipients=[email])
    msg.body = f"""
    Dear {user_name_lst[-1]} ,
    
    We wanted to provide you with an update on your health status based on the data you have provided to OPridictor. Your current BMI is {round(user_bmi(h, w), 2)}, indicating {predicted_classes}.
    It is important to continue monitoring your health and making positive lifestyle changes to improve your overall well-being.
    
    Based on your health status, we have curated a personalized food suggestion template to help you make healthier choices. Please find the below:
    
    Remember, small changes in your diet and lifestyle can have a big impact on your health. If you have any questions or need further assistance, feel free to reach out to us.
    
    Stay healthy!
    
    Best regards,
    The OPridictor Team"""
    
    try:
        mail.send(msg)
        return True
    except Exception as e:
        print("Error sending email:", e)
        return False
    
    

app.secret_key = 'your_secret_key'
@app.route('/send_otp', methods=["POST"])
def send_otp_route():
    """Route for sending OTP email."""
    email = user_email_lst[-1]
    data = session.get('data', {})
    predicted_classes = session.get('predicted_classes', '')
    if send_otp(email, data['height'], data['weight'], predicted_classes):
        session['email'] = email
        return render_template('index.html')
    else:
        return "Failed to send email. Please try again later."

#--------------------------------------------------------------------------graph---------------------------------------------------------
def predict_graph(X_test):
    """Predict the class and probabilities of a given instance."""
    X_test = np.array(X_test)
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_test = X_test.reshape(1, -1)
        logistic_regression, scaler = load('logistic_regression.joblib')

        X_test_scaled = scaler.transform(X_test)
        predicted_probabilities = logistic_regression.predict_proba(X_test_scaled)
        predicted_classes = logistic_regression.predict(X_test_scaled)

        label_names = logistic_regression.classes_
        for i, (label, probabilities) in enumerate(zip(predicted_classes, predicted_probabilities)):
            label_name = label_names[label]
            print(f"Instance {i + 1}: Predicted Class = {label_name}, Probabilities = {probabilities}")

        fig = go.Figure(data=[go.Pie(labels=list(NObeyesdad_labels.values()), values=probabilities, hole=0.4)])
        fig.update_layout(title='Predicted Probabilities for Each Class')
        plot_div = fig.to_html(full_html=False)
        
        nutrient_requirements = {
            "Protein": 15,
            "Carbohydrates": 50,
            "Fats": 30,
            "Vitamins": 2,
            "Minerals": 3,
            "Water": 70,
           "Fiber": 25}
        labels = list(nutrient_requirements.keys())
        values = list(nutrient_requirements.values())

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
        fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=11)
        fig.update_layout(title_text="Nutrient Requirements for Healthy Body")
        helth = fig.to_html(full_html=False)
        
    return plot_div, NObeyesdad_labels[label_name], helth

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/portal', methods=['GET', 'POST'])
def portal():
    """Handle the user portal."""
    data = {}
    if request.method == "POST":
        form_name = request.form['user_email']
        
        if form_name == 'user_data':
            age = int(request.form['age'].split('-')[0])
            current_date = int('20' + x.strftime("%x").split('/')[-1])
            data = {
                'gender': int(request.form['gender']),
                'age': (current_date - age),
                'height': float(request.form['height']) / 100,
                'weight': float(request.form['weight']),
                'family_history': int(request.form['family_history']),
                'favc': int(request.form['favc']),
                'fcvc': int(request.form['fcvc']),
                'ncp': int(request.form['ncp']),
                'caec': int(request.form['caec']),
                'smoke': int(request.form['smoke']),
                'ch2o': int(request.form['ch2o']),
                'scc': int(request.form['scc']),
                'faf': int(request.form['faf']),
                'tue': int(request.form['tue']),
                'calc': int(request.form['calc']),
                'mtrans': int(request.form['mtrans']),
            }
            
            plot_div, predicted_classes, helth = predict_graph([data[key] for key in data.keys()])
            session['data'] = data
            session['predicted_classes'] = predicted_classes
            
        return render_template('user.html', data=data, plot_div=plot_div, predicted_classes=predicted_classes, helth=helth)
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
