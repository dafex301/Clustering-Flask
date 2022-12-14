import os
import numpy as np
import flask
import pickle
from flask import Flask, redirect, url_for, request, render_template


# creating instance of the class
app = Flask(__name__, template_folder='templates')

# to tell flask what url should trigger the function index()


@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


# prediction function
# Memprediksi input dari form user
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 3)
    loaded_model = pickle.load(
        open("./model/model.pkl", "rb"))  # load the model
    # predict the values using loded model
    result = loaded_model.predict(to_predict)
    return result


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        name = request.form['name']
        gender = request.form['gender']
        age = request.form['age']
        annual_income = request.form['annual_income']
        spending_score = request.form['spending_score']

        to_predict_list = list(map(int, [age, annual_income, spending_score]))
        result = ValuePredictor(to_predict_list)

        if int(result) == 0:
            prediction = 'Kamu harus lebih rajin bekerja'
        elif int(result) == 1:
            prediction = 'Kamu seorang pekerja keras tapi konsumtif'
        elif int(result) == 2:
            prediction = 'Kamu mampu mengatur keuangan dengan baik'
        elif int(result) == 3:
            prediction = 'Kamu biasa-biasa saja'
        elif int(result) == 4:
            prediction = 'Kamu kemungkinan adalah pemuda/mahasiswa'

        return render_template("result.html", prediction=prediction, name=name)


if __name__ == "__main__":
    app.run()  # use debug = False for jupyter notebook
