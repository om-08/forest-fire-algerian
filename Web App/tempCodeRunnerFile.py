import pickle
from flask import Flask,request,jsonify,render_template
import  pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge



application = Flask(__name__)
app = application




ridge_model = pickle.load(open('ridge.pkl','rb'))
standard_scaler  = pickle.load(open('scaler.pkl','rb'))




@app.route("/")
def index():    
    return render_template('index.html')










if __name__=='__main__':
    app.run(host = "0.0.0.0")