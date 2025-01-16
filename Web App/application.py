import pickle
from flask import Flask,request,jsonify,render_template
import  pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge



application = Flask(__name__)
app = application

ridge_model = pickle.load(open(r"Machine Learning\Regression Algorithm\Ridge Lasso& Elastic Net Regression\Web App\models\ridge.pkl", "rb"))
standard_scaler = pickle.load(open(r"Machine Learning\Regression Algorithm\Ridge Lasso& Elastic Net Regression\Web App\models\scaler.pkl", "rb"))

@app.route("/")
def index():    
    return render_template('index.html')

@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        
        
        # Performing Standardization change after user put the numbers : 
        new_data_scaled = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])  # needs to be 2 day array as it is independet feature
        
        # Now Predecting The new Data using ridge regression
        result = ridge_model.predict(new_data_scaled)
        
        # This result will be a list of one single output hence we need to index it out and then print the result: 
         
        
        # as we need to show the result in the home.html we shall again use render html
        
        return render_template('home.html',results = result[0])
        
    else:
        return render_template('home.html')  # This will return the html File when triggerd with get method like the homepage 


if __name__=='__main__':
    app.run(host = "0.0.0.0")