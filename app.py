from flask import Flask, render_template, request, url_for
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
		if(prediction>0.9999):
			prediction=0.9999
		if(prediction<0):
			prediction=0

		if(prediction>0.9):
			univ_list = [
				'Massachusetts Institute of Technology (MIT) United States',
				'Stanford University, United States',
				'Carnegie Mellon University, United States',
				'Harvard University, United States',
				'California Institute of Technology (Caltech), United States',
				'University of Oxford, United Kingdom',
				'ETH Zurich (Swiss Federal Institute of Technology), Switzerland',
				'University of Cambridge, United Kingdom',
				'Imperial College London, United Kingdom',
				'University of Chicago, United States',
			]
		
		elif(prediction>0.8):
			univ_list = [
				'National University of Singapore (NUS), Singapore',
				'Princeton University, United States',
				'Nanyang Technological University, Singapore (NTU), Singapore',
				'Ecole Polytechnique Fédérale de Lausanne (EPFL), Switzerland',
				'Tsinghua University, China',
				'University of Pennsylvania, United States',
				'Yale University, United States',
				'Cornell University, United States',
				'Columbia University, United States',
				'Georgia Institute of Technology, United States',
			]
		
		elif(prediction>0.7):
			univ_list = [
				'University of Michigan-Ann Arbor, United States',
				'University of Hong Kong (UKU), Hong Kong',
				'University of Edinburgh, United Kingdom',
				'Peking University, China',
				'University of Tokyo, Japan',
				'Johns Hopkins University, United States',
				'University of Toronto, Canada',
				'University of Manchester, United Kingdom',
				'Northwestern University, United States',
				'University of California, Berkeley (UCB), United States',
			]
		
		elif(prediction>0.6):
			univ_list = [
				'Australian National University, Australia',
				'King\'s College London, United Kingdom',
				'McGill University, Canada',
				'Hong Kong University of Science and Technology (HKUST), Hong Kong',
				'New York University (NYU), United States',
				'University of California, Los Angeles (UCLA), United States',
				'Seoul National University, South Korea',
				'Kyoto University, Japan',
				'KAIST - Korea Advanced Institute of Science & Technology, South Korea',
				'University of Sydney, Australia',
			]

		elif(prediction>0.5):
			univ_list = [
				'University of Melbourne, Australia',
				'Duke University, United States',
				'Chinese University of Hong Kong (CUHK), Hong Kong',
				'University of New South Wales (UNSW Sydney), Australia',
				'University of British Columbia, Canada',
				'University of Queensland, Australia',
				'Shanghai Jiao Tong University, China',
				'City University of Hong Kong, Hong Kong',
				'London School of Economics and Political Science (LSE), United Kingdom',
				'Technical University of Munich, Germany',
			]

		elif(prediction>0.4):
			univ_list = [
				'UCL (University College London), United Kingdom',
				'Universite PSL, France',
				'Zhejiang University, China',
				'University of California, San Diego (UCSD), United States',
				'Monash University, Australia',
				'Tokyo Institute of Technology, Japan',
				'Delft University of Technology, Netherlands',
				'University of Bristol, United Kingdom',
				'Universiti Malaya (UM), Malaysia',
				'Brown University, United States',
			]

		elif(prediction>0.3):
			univ_list = [
				'University of Amsterdam, Netherlands',
				'University of Warwick, United Kingdom',
				'Ludwig-Maximilians-Universität München, Germany',
				'Ruprecht-Karls-Universitat Heidelberg, Germany',
				'University of Wisconsin-Madison, United States',
				'National Taiwan University (NTU), Taiwan',
				'Universidad de Buenos Aires (UBA), Argentina',
				'Ecole Polytechnique, France',
				'Korea University, South Korea',
				'University of Zurich, Switzerland',
			]
		
		elif(prediction>0.2):
			univ_list = [
				'University of Texas at Austin, United States',
				'Osaka University, Japan',
				'University of Washington, United States',
				'Lomonosov Moscow State University, Russia',
				'Hong Kong Polytechnic University, Hong Kong',
				'University of Copenhagen, Denmark',
				'Pohang University of Science and Technology (POSTECH), South Korea',
				'University of Glasgow, United Kingdom',
				'Tohoku University, Japan',
				'Fudan University, China',
			]

		elif(prediction>0.1):
			univ_list = [
				'University of Auckland, New Zealand',
				'University of Illinois at Urbana-Champaign, United States',
				'Sorbonne University, France',
				'KU Leuven, Belgium',
				'Durham University, United Kingdom',
				'Yonsei University, South Korea',
				'University of Birmingham, United Kingdom',
				'Sungkyunkwan University (SKKU), South Korea',
				'Rice University, United States',
				'University of Southampton, United Kingdom',
			]
		
		else:
			univ_list = [
				'University of Leeds, United Kingdom',
				'University of Western Australia, Australia',
				'University of Sheffield, United Kingdom',
				'University of Science and Technology of China, China',
				'University of North Carolina, Chapel Hill, United States',
				'University of St Andrews, United Kingdom',
				'Lund University, Sweden',
				'KTH Royal Institute of Technology, Sweden',
				'University of Nottingham, United Kingdom',
				'Universidad Nacional Autónoma de México (UNAM), Mexico',
			]

		return render_template('univ.html', univ_list=univ_list, prediction=prediction)

# @app.route('/results')

if __name__ == '__main__':
	app.run(debug=True)