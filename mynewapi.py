#imports of libraries
from flask import Flask,request,jsonify 
#loads model
import joblib
#creates dataframe 
import pandas as pd

#create flask app
app = Flask(__name__)

#LISTENING FOR A URI OF /PREDICT ITS A POST REQUEST(SHOWN IN THE BODY)
@app.route('/predict', methods=['POST'])
def predict():
#loading model and column names
    colnames = joblib.load('/Users/jsingh/Documents/Draft_Tech/ML_project/jade_columns.pkl')
    model = joblib.load('/Users/jsingh/Documents/Draft_Tech/ML_project/jade_model.pkl')
    
    #get JSON REQUEST
    feature_data = request.json

    #convert json to pandas dataframe (col names)
    
    df = pd.DataFrame(feature_data)
    df = df.reindex(columns=colnames)
    
    #predicts the dataframe
    
    prediction = list(model.predict(df))
    
    return jsonify({'prediction':str(prediction)})
#runs the application
    app.run(debug=True)
    