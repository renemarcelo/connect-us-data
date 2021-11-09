from flask import Flask, Response, json, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# initialize the flask app
app = Flask(__name__)
CORS(app)

# route 1: county dashboard
# @app.route('/model')

@app.route('/')
def hello_world():
    return 'Connect US - A Census Bureau Product Club App'

@app.route('/model', methods= ['POST'])
def input_values():
      print(request.json, "request")
      values = request.json
      if values:
          income = values["state"]["income"]
          temp = values["state"]["climate"]
          pop = values["state"]["pop"]
          elevation = values['state']["elevation"]
          print(income,temp,pop,elevation, "values")
          result = county(income, temp, pop, elevation)
          print(result, "results printed")
      return jsonify(result)

# {'incomeRangeVal': '563042', 'climateRangeVal': '51', 'popRangeVal': '606995'}

def county(income, temp, pop, elevation):
    print('county is running')
    np.random.seed(123)
    
    # data import
    data = pd.read_csv('./data/joined_data5.csv').drop(columns=['Unnamed: 0'])
    # dataframe for prediction
    model_df = data[['county','median_income','temp','total_population','elev_in_ft']]
    # dataset for additional county data
    additional_data = data[['county', 'employed_population_over16', '18_24_college_grad_enroll',
                            'median_gross_rent_dollars', 'employed', 'unemployed',
                            'percent_insured', 'mean_hours_worked', 'median_age_workers_16_to_64',
                            'mean_travel_time_to_work_min', 'perc_pop_25plus_bach_or_higher',
                            'perc_k_12_enrollment', 'label']]
    #import scaler
    scaler = pickle.load(open('./model/scaler.pkl', 'rb'))
    #import knn_model
    knn_model = pickle.load(open('./model/knn_model_new.pkl', 'rb'))
    
    #user preferences
    pref={'median_income': [income], 
        'temp': [temp], 
        'total_population': [pop],
        'elevation': [elevation]
        }    
    
    #create user preferences dataframe & predict
    prefdf = pd.DataFrame.from_dict(pref, orient='index').T
    pref_scaled = scaler.transform(prefdf)
    pred = knn_model.predict(pref_scaled)[0]
    
    #output from user preferences
    user_df = data[data.label == pred][['county', 'median_income', 'temp', 'total_population']]
    user_show = pd.DataFrame(user_df)
    
    #return function with additional data
    return {'model_results' : user_show.merge(additional_data, on='county').to_dict()}

if __name__ == "__main__":
    app.run(debug=True)
