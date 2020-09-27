from flask import Flask, render_template, request
from sklearn.externals import joblib
import pickle

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
		
		# return render_template('univ.html', result=result)

# @app.route('/results')

if __name__ == '__main__':
	app.run(debug=True)