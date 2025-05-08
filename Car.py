import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Load and preprocess the data
df = pd.read_csv("CARS.csv")
df.MSRP = df.MSRP.replace("[$,]", "", regex=True).astype('int64')
df.Invoice = df.Invoice.replace("[$,]", "", regex=True).astype('int64')
df.Cylinders = df.Cylinders.fillna(0)

# Encode categorical columns
label_encoders = {}
categorical_cols = ['Make', 'Model', 'Type', 'Origin', 'DriveTrain']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split features and target
X = df.drop(['MSRP'], axis=1)
y = df['MSRP']

# Train model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("Car MSRP Prediction App")

# Input fields
make = st.selectbox("Make", label_encoders['Make'].classes_)
model_input = st.selectbox("Model", label_encoders['Model'].classes_)
car_type = st.selectbox("Type", label_encoders['Type'].classes_)
origin = st.selectbox("Origin", label_encoders['Origin'].classes_)
drive_train = st.selectbox("Drive Train", label_encoders['DriveTrain'].classes_)
engine_size = st.slider("Engine Size", float(df.EngineSize.min()), float(df.EngineSize.max()), 3.0)
cylinders = st.slider("Cylinders", int(df.Cylinders.min()), int(df.Cylinders.max()), 4)
horsepower = st.slider("Horsepower", int(df.Horsepower.min()), int(df.Horsepower.max()), 200)
mpg_city = st.slider("MPG (City)", int(df.MPG_City.min()), int(df.MPG_City.max()), 20)
mpg_highway = st.slider("MPG (Highway)", int(df.MPG_Highway.min()), int(df.MPG_Highway.max()), 25)
weight = st.slider("Weight", int(df.Weight.min()), int(df.Weight.max()), 3500)
wheelbase = st.slider("Wheelbase", float(df.Wheelbase.min()), float(df.Wheelbase.max()), 107.0)
length = st.slider("Length", float(df.Length.min()), float(df.Length.max()), 190.0)
invoice = st.number_input("Invoice Price ($)", min_value=0, value=25000)

# Encode user inputs
base_input = pd.DataFrame([[ 
    label_encoders['Make'].transform([make])[0],
    label_encoders['Model'].transform([model_input])[0],
    engine_size,
    cylinders,
    horsepower,
    mpg_city,
    mpg_highway,
    weight,
    wheelbase,
    length,
    invoice,
    label_encoders['Type'].transform([car_type])[0],
    label_encoders['Origin'].transform([origin])[0],
    label_encoders['DriveTrain'].transform([drive_train])[0]
]], columns=X.columns)

# Prediction and display
if st.button("Predict MSRP"):
    prediction = model.predict(base_input)
    st.success(f"Estimated MSRP: ${int(prediction[0]):,}")

    # Generate line plot data for 3 features
    hp_vals = np.linspace(df.Horsepower.min(), df.Horsepower.max(), 50).astype(int)
    eng_vals = np.linspace(df.EngineSize.min(), df.EngineSize.max(), 50)
    wt_vals = np.linspace(df.Weight.min(), df.Weight.max(), 50).astype(int)

    records = []

    for hp in hp_vals:
        temp_input = base_input.copy()
        temp_input['Horsepower'] = hp
        pred = model.predict(temp_input)[0]
        records.append(('Horsepower', hp, pred))

    for es in eng_vals:
        temp_input = base_input.copy()
        temp_input['EngineSize'] = es
        pred = model.predict(temp_input)[0]
        records.append(('Engine Size', es, pred))

    for wt in wt_vals:
        temp_input = base_input.copy()
        temp_input['Weight'] = wt
        pred = model.predict(temp_input)[0]
        records.append(('Weight', wt, pred))

    plot_df = pd.DataFrame(records, columns=['Feature', 'Value', 'Predicted MSRP'])
    fig = px.line(plot_df, x='Value', y='Predicted MSRP', color='Feature',
                  title='Predicted MSRP vs Feature Value')
    st.plotly_chart(fig)
