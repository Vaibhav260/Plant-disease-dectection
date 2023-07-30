from flask import Flask, render_template, request
import joblib
import numpy as np
import cv2

app = Flask(__name__, template_folder='/Users/vaibhav/Nmims/sem 6/Machine learning/Machine learning code/templates')

# Load the trained model
knn = joblib.load('/Users/vaibhav/Nmims/sem 6/Machine learning/Machine learning code/model.joblib')

# Define the route for the prediction
def predict(image):
    # Load the image and preprocess it
    img = cv2.imread(image)
    img = cv2.resize(img, (128, 128))
    img = img.flatten().reshape(1, -1)

    # Make the prediction
    prediction = knn.predict(img)
    return prediction[0]

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        image = request.files['image']
        image.save('/Users/vaibhav/Nmims/sem 6/Machine learning/Machine learning code/templates/static/image.png')
        prediction = predict('/Users/vaibhav/Nmims/sem 6/Machine learning/Machine learning code/templates/static/image.png')
        return render_template('result.html', prediction=prediction)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
