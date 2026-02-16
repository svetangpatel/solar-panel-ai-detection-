from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from flask import send_from_directory
import yagmail

app = Flask(__name__)
DATABASE = 'database.db'

model = tf.keras.models.load_model("model.keras", compile=False)

classes = ["Bird Drop",
    "Clean Panel",
    "Crack",
    "Dusty Panel",
    "Electrical Damage",
    "Snow Covered"]

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def send_email_alert(result, confidence):
    try:
        yag = yagmail.SMTP("youremailid@gmail.com", "abcd efgh ijkl mopq")

        subject = "⚠ Solar Panel Fault Alert"
        body = f"""
Solar Panel Fault Detected!

Fault Type: {result}
Confidence: {round(confidence*100,2)}%

Action Required: Please check panel immediately.
        """

        yag.send("youremailid@gmail.com", subject, body)
        print("Email sent successfully")

    except Exception as e:
        print("Email error:", e)

@app.route('/dashboard')
def dashboard():

    cursor.execute("SELECT COUNT(*) FROM records")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM records WHERE result='Dusty Panel'")
    dusty = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM records WHERE result IN ('Electrical Damage', 'Crack')")
    damaged = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM records WHERE result='Clean Panel'")
    clean = cursor.fetchone()[0]
    
    conn.close()

    return render_template("dashboard.html",
                           total=total,
                           dusty=dusty,
                           damaged=damaged,
                           clean=clean)

# HOME PAGE
@app.route('/')
def home():
    return render_template("index.html")


# PREDICTION
@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img_path = "uploads/" + filename

        img = cv2.imread(filepath)
        img = cv2.resize(img,(224,224))
        img = img/255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        index = np.argmax(pred)
        confidence = np.max(pred)*100

        result = classes[index]

        if result in ["Bird Drop",
           "Clean Panel",
           "Crack",
           "Dusty Panel",
           "Electrical Damage",
           "Snow Covered"]:
           send_email_alert(result, confidence)
    

        if result == "Dusty Panel":
            loss_percent = 20
            suggestion = "Clean panel to improve efficiency"
        elif result == "Electrical Damage":
            loss_percent = 45
            suggestion = "Panel needs repair"
        elif result == "Shaded":
            loss_percent = 30
            suggestion = "Remove obstacle"
        elif result == "Snow Covered":
            loss_percent = 50
            suggestion = "Remove snow"
        elif result == "Bird Drop":
            loss_percent = 15
            suggestion = "Clean panel to improve efficiency"
        elif result == "Crack":
            loss_percent = 40
            suggestion = "Panel needs repair"
        else:
            loss_percent = 0
            suggestion = "Panel working perfectly"

        return render_template("index.html",
                               prediction=result,
                               confidence=round(confidence,2),
                               suggestion=suggestion,
                               loss_percent=loss_percent,   
                               img_path=img_path)

    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)