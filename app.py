from flask import Flask, render_template, request
import pandas as pd
import joblib
import plotly.express as px

app = Flask(__name__)

# Load models
model = joblib.load("house_price_stacking_model.pk1")
rf_model = joblib.load("house_price_rf_model.pk1")

print("MODEL LOADED")


# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template("index.html")


# ---------------- PREDICT ----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'calculatedfinishedsquarefeet': float(request.form.get('calculatedfinishedsquarefeet', 0)),
            'bedroomcnt': float(request.form.get('bedroomcnt', 0)),
            'bathroomcnt': float(request.form.get('bathroomcnt', 0)),
            'yearbuilt': float(request.form.get('yearbuilt', 0)),
            'garagecarcnt': float(request.form.get('garagecarcnt', 0)),

            # Default values
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

        input_df = pd.DataFrame([data])

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

        prediction = model.predict(input_df)[0]
        prediction = abs(prediction)

        confidence_text = f"Estimated Range: ${round(prediction*0.9,2)} - ${round(prediction*1.1,2)}"

        # Feature importance (interactive)
        importance_df = pd.DataFrame({
            "Feature": input_df.columns,
            "Importance": rf_model.feature_importances_
        })

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation='h',
            title="Feature Importance",
            color="Importance"
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

    # Sample data
    data = pd.DataFrame({
        "bedrooms": list(range(1, bedrooms + 1)),
        "bathrooms": list(range(1, bedrooms + 1)),
        "area": [i * 500 for i in range(1, bedrooms + 1)],
        "price": [i * 100000 for i in range(1, bedrooms + 1)]
    })

    # -------- Interactive Charts --------

    fig1 = px.bar(
        data,
        x="bedrooms",
        y="price",
        title="Bedrooms vs Price",
        color="price"
    )

    fig2 = px.scatter(
        data,
        x="area",
        y="price",
        title="Area vs Price",
        color="price"
    )

    fig3 = px.scatter(
        data,
        x="bathrooms",
        y="price",
        title="Bathrooms vs Price",
        color="price"
    )

    return render_template(
        "dashboard.html",
        bar=fig1.to_html(full_html=False),
        scatter=fig2.to_html(full_html=False),
        scatter2=fig3.to_html(full_html=False),
        bedrooms=bedrooms
    )


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
