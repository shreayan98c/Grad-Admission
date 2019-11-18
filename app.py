from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/universities')
def univ():
	return render_template('univ.html')

# @app.route('/results')

if __name__ == '__main__':
	app.run(debug=True)