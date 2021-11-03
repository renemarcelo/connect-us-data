from flask import Flask, Response, json, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
# from flask_cors import CORS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# initialize the flask app
app = Flask(__name__)
# CORS(app)


# route 1: county dashboard
# @app.route('/model')

@app.route('/model', methods= ['POST'])
def input_values():
      print(request.json)
      values = request.json
      if values:
          income = values["incomeRangeVal"]
          temp = values["climateRangeVal"]
          pop = values["popRangeVal"]
          print(income,temp,pop, "values")
          result = county(income, temp, pop)
          print(result, "results printed")
      return jsonify(result)
          

# {'incomeRangeVal': '563042', 'climateRangeVal': '51', 'popRangeVal': '606995'}

def county(income, temp, pop):
    print('county is running')
    np.random.seed(123)
    
    data = pd.read_csv('./data/joined_data_2.csv').drop(columns=['Unnamed: 0'])
    cols = data.columns[1:]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[cols])
    
    kmeans = KMeans(n_clusters=200)
    kmeans.fit(X_scaled)

    data['label'] = kmeans.labels_
    y = data.label

    knn_model = pickle.load(open('./model/knn_updated2.pkl', 'rb'))
    print('pickle load')
#     knn_model.fit(X_scaled, y)
    
    pref={'median_income': [income], 
        'temp': [temp], 
        'total_population': [pop],
        'elevation': [80]
        }
    print('line43')
    
    prefdf = pd.DataFrame.from_dict(pref, orient='index').T
    print(prefdf)
    pref_scaled = scaler.transform(prefdf)
    print('line48')
    pred = knn_model.predict(pref_scaled)[0]
    print('line50')
    
    user_df = data[data.label == pred].county
    user_show = pd.DataFrame(user_df)
    print(user_show)
    #return(user_show)
    return {'model_results' : user_show.to_dict()}



if __name__ == "__main__":
    app.run(debug=True)
