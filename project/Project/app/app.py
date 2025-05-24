# from flask import Flask, render_template, request
# import pandas as pd
# import joblib
# import os

# app = Flask(__name__)

# # --- Path Setup ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, 'model')

# # Define paths for all four models
# csv_path = os.path.join(MODEL_DIR, 'PG_Dataset_Cleaned_For_Model.csv')
# similarity_path = os.path.join(MODEL_DIR, 'similarity.pkl')
# gb_model_path = os.path.join(MODEL_DIR, 'model_gbr.pkl')  # Gradient Boosting Regressor
# rf_model_path = os.path.join(MODEL_DIR, 'model_rfr.pkl')  # Random Forest Regressor
# predictor_model_path = os.path.join(MODEL_DIR, 'predictor.pkl')  # Predictor model

# # --- Load Data and Models ---
# try:
#     df = pd.read_csv(csv_path)
#     df['name_of_pg_clean'] = df['Name_of_PG'].str.strip().str.lower()
# except FileNotFoundError:
#     raise FileNotFoundError(f"CSV file not found! Make sure it exists at: {csv_path}")

# try:
#     similarity = joblib.load(similarity_path)
#     gb_model = joblib.load(gb_model_path)
#     rf_model = joblib.load(rf_model_path)
#     predictor_model = joblib.load(predictor_model_path)
# except FileNotFoundError as e:
#     raise FileNotFoundError(f"Model file not found! Check files in: {MODEL_DIR}\n{e}")

# # --- Routes ---
# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     pg_name = request.form['pg_name'].strip().lower()

#     if pg_name not in df['name_of_pg_clean'].values:
#         return render_template('result.html', recommendations=["PG not found. Please check the name."])

#     idx = df[df['name_of_pg_clean'] == pg_name].index[0]
#     scores = list(enumerate(similarity[idx]))
#     scores = sorted(scores, key=lambda x: x[1], reverse=True)
#     top_pgs = [df.iloc[i[0]]['Name_of_PG'] for i in scores[1:6]]

#     return render_template('result.html', recommendations=top_pgs)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         features = [
#             float(request.form['bedrooms']),
#             float(request.form['distance']),
#             int(request.form['year']),
#         ]
#     except (ValueError, KeyError):
#         return render_template('result.html', prediction="Invalid input. Please enter valid values.")

#     # Predict using both models and average the results
#     gb_pred = gb_model.predict([features])[0]
#     rf_pred = rf_model.predict([features])[0]
#     avg_pred = (gb_pred + rf_pred) / 2

#     return render_template('result.html', prediction=round(avg_pred, 2))

# @app.route('/predictor', methods=['POST'])
# def predictor():
#     try:
#         features = [
#             float(request.form['bedrooms']),
#             float(request.form['distance']),
#             int(request.form['year']),
#         ]
#     except (ValueError, KeyError):
#         return render_template('result.html', prediction="Invalid input. Please enter valid values.")

#     # Use the predictor model to make a prediction
#     prediction = predictor_model.predict([features])[0]

#     return render_template('result.html', prediction=round(prediction, 2))

# # --- Run the App ---
# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, render_template, request
# import pandas as pd
# import joblib
# import os

# app = Flask(__name__)

# # --- Path Setup ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, 'model')

# # Define paths for all four models
# csv_path = os.path.join(MODEL_DIR, 'PG_Dataset_Cleaned_For_Model.csv')
# similarity_path = os.path.join(MODEL_DIR, 'similarity.pkl')
# gb_model_path = os.path.join(MODEL_DIR, 'model_gbr.pkl')
# rf_model_path = os.path.join(MODEL_DIR, 'model_rfr.pkl')
# predictor_model_path = os.path.join(MODEL_DIR, 'predictor.pkl')

# # --- Load Data and Models ---
# try:
#     df = pd.read_csv(csv_path)
#     df['name_of_pg_clean'] = df['Name_of_PG'].str.strip().str.lower()
# except FileNotFoundError:
#     raise FileNotFoundError(f"CSV file not found! Make sure it exists at: {csv_path}")

# try:
#     similarity = joblib.load(similarity_path)
#     gb_model = joblib.load(gb_model_path)
#     rf_model = joblib.load(rf_model_path)
#     predictor_model = joblib.load(predictor_model_path)
# except FileNotFoundError as e:
#     raise FileNotFoundError(f"Model file not found! Check files in: {MODEL_DIR}\n{e}")

# # --- Routes ---
# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     pg_name = request.form['pg_name'].strip().lower()

#     if pg_name not in df['name_of_pg_clean'].values:
#         return render_template('result.html', recommendations=["PG not found. Please check the name."])

#     idx = df[df['name_of_pg_clean'] == pg_name].index[0]
#     scores = list(enumerate(similarity[idx]))
#     scores = sorted(scores, key=lambda x: x[1], reverse=True)
#     top_pgs = [df.iloc[i[0]]['Name_of_PG'] for i in scores[1:6]]

#     return render_template('result.html', recommendations=top_pgs)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         bedrooms = float(request.form['bedrooms'])
#         distance = float(request.form['distance'])
#         year = int(request.form['year'])
#     except (ValueError, KeyError):
#         return render_template('result.html', prediction="Invalid input. Please enter valid values.")

#     features = [bedrooms, distance, year]

#     # Predict using both models and average the results
#     gb_pred = gb_model.predict([features])[0]
#     rf_pred = rf_model.predict([features])[0]
#     avg_pred = (gb_pred + rf_pred) / 2

#     return render_template(
#         'result.html',
#         prediction=round(avg_pred, 2),
#         bedrooms=bedrooms,
#         distance=distance,
#         year=year
#     )

# @app.route('/predictor', methods=['POST'])
# def predictor():
#     try:
#         bedrooms = float(request.form['bedrooms'])
#         distance = float(request.form['distance'])
#         year = int(request.form['year'])
#     except (ValueError, KeyError):
#         return render_template('result.html', prediction="Invalid input. Please enter valid values.")

#     features = [bedrooms, distance, year]
#     prediction = predictor_model.predict([features])[0]

#     return render_template(
#         'result.html',
#         prediction=round(prediction, 2),
#         bedrooms=bedrooms,
#         distance=distance,
#         year=year
#     )

# # --- Run the App ---
# if __name__ == '__main__':
#     app.run(debug=True)
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         bedrooms = float(request.form['bedrooms'])
#         distance = float(request.form['distance'])
#         year = int(request.form['year'])
#     except (ValueError, KeyError):
#         return render_template('result.html', prediction="Invalid input. Please enter valid values.")

#     features = [bedrooms, distance, year]

#     # Predict using both models and average the results
#     gb_pred = gb_model.predict([features])[0]
#     rf_pred = rf_model.predict([features])[0]
#     avg_pred = (gb_pred + rf_pred) / 2

#     return render_template(
#         'result.html',
#         prediction=round(avg_pred, 2),
#         bedrooms=bedrooms,
#         distance=distance,
#         year=year
#     )

# @app.route('/predictor', methods=['POST'])
# def predictor():
#     try:
#         bedrooms = float(request.form['bedrooms'])
#         distance = float(request.form['distance'])
#         year = int(request.form['year'])
#     except (ValueError, KeyError):
#         return render_template('result.html', prediction="Invalid input. Please enter valid values.")

#     features = [bedrooms, distance, year]
#     prediction = predictor_model.predict([features])[0]

#     return render_template(
#         'result.html',
#         prediction=round(prediction, 2),
#         bedrooms=bedrooms,
#         distance=distance,
#         year=year
#     )
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# --- Path Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Define paths for all four models
csv_path = os.path.join(MODEL_DIR, 'PG_Dataset_Cleaned_For_Model.csv')
similarity_path = os.path.join(MODEL_DIR, 'similarity.pkl')
gb_model_path = os.path.join(MODEL_DIR, 'model_gbr.pkl')
rf_model_path = os.path.join(MODEL_DIR, 'model_rfr.pkl')
predictor_model_path = os.path.join(MODEL_DIR, 'predictor.pkl')

# --- Load Data and Models ---
try:
    df = pd.read_csv(csv_path)
    df['name_of_pg_clean'] = df['Name_of_PG'].str.strip().str.lower()
except FileNotFoundError:
    raise FileNotFoundError(f"CSV file not found! Make sure it exists at: {csv_path}")

try:
    similarity = joblib.load(similarity_path)
    gb_model = joblib.load(gb_model_path)
    rf_model = joblib.load(rf_model_path)
    predictor_model = joblib.load(predictor_model_path)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Model file not found! Check files in: {MODEL_DIR}\n{e}")

# --- Routes ---
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    pg_name = request.form['pg_name'].strip().lower()

    if pg_name not in df['name_of_pg_clean'].values:
        return render_template('result.html', recommendations=[{"Name_of_PG": "PG not found. Please check the name."}])

    idx = df[df['name_of_pg_clean'] == pg_name].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in scores[1:6]]
    recommended_pgs = df.iloc[top_indices][['Name_of_PG', 'Monthly_rent', 'Year_of_build', 'Distance_from_college','Location_City','Furnishing_services']]

    recommendations = recommended_pgs.to_dict(orient='records')
    return render_template('result.html', recommendations=recommendations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        bedrooms = float(request.form['bedrooms'])
        distance = float(request.form['distance'])
        year = int(request.form['year'])
    except (ValueError, KeyError):
        return render_template('result.html', prediction="Invalid input. Please enter valid values.")

    features = [bedrooms, distance, year]

    gb_pred = gb_model.predict([features])[0]
    rf_pred = rf_model.predict([features])[0]
    avg_pred = (gb_pred + rf_pred) / 2

    return render_template(
        'result.html',
        prediction=round(avg_pred, 2),
        bedrooms=bedrooms,
        distance=distance,
        year=year
    )


# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)

