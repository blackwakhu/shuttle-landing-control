from flask import Flask, render_template, request
import numpy as np
import pickle 

app = Flask(__name__)

model = pickle.load(open('GaussianNB.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    initial = [float(x) for x in request.form.values()]
    final = np.array(initial).reshape(1,-1)
    pred = model.predict(final)
    if pred == '1':
        prediction = 'manual'
    else:
        prediction = 'auto'
    
    return render_template('index.html', prediction=f'the system to be used should be {prediction}')

if __name__ == '__main__':
    app()