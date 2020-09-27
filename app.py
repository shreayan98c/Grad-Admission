from flask import Flask, render_template, request
from sklearn.externals import joblib
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/universities', methods=['POST'])
def universities():
	if request.method == 'POST':
		gre = request.form['gre']
		toefl = request.form['toefl']
		rating = request.form['gridRadios']
		sop = request.form['sop']
		lor = request.form['lor']
		cgpa = request.form['cgpa']
		research = request.form.getlist('research')
		if(not research):
			research=0
		else:
			research=1
		model_input = pd.DataFrame([[gre, toefl, rating, sop, lor, cgpa, research]], columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research'])
		prediction = model.predict(model_input)[0]

		if(prediction>0.9):
			univ_list = ['Harvard', 'Stanford', 'Gatech']
			
		else:
			univ_list = ['UCLA', 'UCSD', 'Columbia']

		return render_template('univ.html', univ_list=univ_list, prediction=prediction)

# @app.route('/results')

if __name__ == '__main__':
	app.run(debug=True)