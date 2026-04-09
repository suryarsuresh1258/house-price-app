from flask import Flask, render_template, request
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objs as go
import random

app = Flask(__name__)

# Load trained models
model = joblib.load("house_price_stacking_model.pk1")
rf_model = joblib.load("house_price_rf_model.pk1")

print("✅ Models loaded successfully")


# ---------------- HOME PAGE ----------------
@app.route('/')
def home():
    return render_template("index.html")


# ---------------- PREDICTION ----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from form
        input_data = {
            'calculatedfinishedsquarefeet': float(request.form.get('calculatedfinishedsquarefeet', 0)),
            'bedroomcnt': float(request.form.get('bedroomcnt', 0)),
            'bathroomcnt': float(request.form.get('bathroomcnt', 0)),
            'yearbuilt': float(request.form.get('yearbuilt', 0)),
            'garagecarcnt': float(request.form.get('garagecarcnt', 0)),

            # Keep your remaining features (important)
            'garagetotalsqft': 400,
            'latitude': 34.05,
            'longitude': -118.25,
            'lotsizesquarefeet': 6000,
            'unitcnt': 1,
            'taxamount': 5000,
            'regionidcity': 12447,
            'propertylandusetypeid': 261,
            'month': 6
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # 🔥 IMPORTANT: Keep exact feature order
        input_df = input_df[[
            'calculatedfinishedsquarefeet',
            'bedroomcnt',
            'bathroomcnt',
            'yearbuilt',
            'garagecarcnt',
            'garagetotalsqft',
            'latitude',
            'longitude',
            'lotsizesquarefeet',
            'unitcnt',
            'taxamount',
            'regionidcity',
            'propertylandusetypeid',
            'month'
        ]]

        # Prediction
        prediction = model.predict(input_df)[0]

        if prediction < 0:
            prediction = abs(prediction)

        lower = prediction * 0.9
        upper = prediction * 1.1

        confidence_text = f"Estimated Range: ${round(lower,2)} - ${round(upper,2)}"

        # ---------------- FEATURE IMPORTANCE ----------------
        importance_df = pd.DataFrame({
            "Feature": input_df.columns,
            "Importance": rf_model.feature_importances_
        })

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation='h',
            title="Feature Importance"
        )

        plot_html = fig.to_html(full_html=False)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Price: ${round(prediction,2)}",
            confidence_text=confidence_text,
            plot_html=plot_html
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


# ---------------- DASHBOARD ----------------
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    bedrooms = 5

    if request.method == 'POST':
        bedrooms = int(request.form.get('bedrooms', 5))

    # Simulated data for visualization
    bedroom_list = list(range(1, bedrooms + 1))
    price_list = [i * random.randint(90000, 140000) for i in bedroom_list]

    # Bar Chart
    bar = go.Bar(x=bedroom_list, y=price_list, name="Bedrooms vs Price")

    # Scatter / Trend
    scatter = go.Scatter(
        x=bedroom_list,
        y=price_list,
        mode='lines+markers',
        name="Trend"
    )

    # Histogram
    hist = go.Histogram(x=price_list, name="Price Distribution")

    graph_data = {
        "bar": bar,
        "scatter": scatter,
        "hist": hist
    }

    return render_template(
        "dashboard.html",
        graph_data=graph_data,
        bedrooms=bedrooms
    )


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
