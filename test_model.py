from flask import Flask, Response, json, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import time

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
    start = time.time()
    print('County is running...')
    np.random.seed(123)
    
    data = pd.read_csv('./data/joined_data_2.csv').drop(columns=['Unnamed: 0'])
    #additional dataframe
    additional_data= pd.read_csv('./data/joined_data_3.csv')
    
    cols = data.columns[1:]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[cols])
    end = time.time()
    print(end - start, 'Time after fit_transform')
    kmeans = KMeans(n_clusters=200)
    kmeans.fit(X_scaled)
    end = time.time()
    print(end - start, 'Time after kmeans.fit')
    data['label'] = kmeans.labels_
    y = data.label

    knn_model = pickle.load(open('./model/knn_updated2.pkl', 'rb'))
    end = time.time()
    print(end - start, 'Time after pickle load')
#   knn_model.fit(X_scaled, y)
    
    pref={'median_income': [income], 
        'temp': [temp], 
        'total_population': [pop],
        'elevation': [elevation]
        }
    
    prefdf = pd.DataFrame.from_dict(pref, orient='index').T
    pref_scaled = scaler.transform(prefdf)
    pred = knn_model.predict(pref_scaled)[0]
    user_df = data[data.label == pred].county
    user_show = pd.DataFrame(user_df)
    
    end = time.time()
    print(end - start, 'Completed Time')
    return {'model_results' : user_show.merge(additional_data, on='county').to_dict()}



if __name__ == "__main__":
    app.run(debug=True)
