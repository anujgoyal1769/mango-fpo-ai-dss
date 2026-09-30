from pathlib import Path

import joblib
import pandas as pd


MODEL_FILE = Path("models/mango_xgboost_model.joblib")


def load_model():
    """Load the trained XGBoost pipeline."""
    bundle = joblib.load(MODEL_FILE)
    return bundle["pipeline"], bundle["features"]


def prepare_input(
    date,
    state,
    district,
    market,
    variety,
    min_price,
    max_price,
    modal_price,
    lag_1_price,
    lag_3_price,
    lag_7_price,
    price_change_1,
    price_change_3,
    rolling_mean_3,
    rolling_mean_7,
    rolling_std_7,
):
    """Create one model-input row using the project's feature schema."""

    date = pd.to_datetime(date)

    return pd.DataFrame(
        [{
            "min_price": min_price,
            "max_price": max_price,
            "modal_price": modal_price,
            "lag_1_price": lag_1_price,
            "lag_3_price": lag_3_price,
            "lag_7_price": lag_7_price,
            "price_change_1": price_change_1,
            "price_change_3": price_change_3,
            "rolling_mean_3": rolling_mean_3,
            "rolling_mean_7": rolling_mean_7,
            "rolling_std_7": rolling_std_7,
            "year": date.year,
            "month": date.month,
            "day": date.day,
            "day_of_year": date.dayofyear,
            "day_of_week": date.dayofweek,
            "week_of_year": int(date.isocalendar().week),
            "state": state,
            "district": district,
            "market": market,
            "variety": variety,
        }]
    )


def predict_tomorrow(**kwargs):
    """Return the predicted next-day modal price."""

    pipeline, _ = load_model()

    input_df = prepare_input(**kwargs)

    prediction = pipeline.predict(input_df)[0]

    return float(prediction)


if __name__ == "__main__":

    prediction = predict_tomorrow(
        date="2025-08-13",
        state="Punjab",
        district="Mohali",
        market="Banur",
        variety="Other",
        min_price=3000,
        max_price=4000,
        modal_price=3500,
        lag_1_price=3400,
        lag_3_price=3300,
        lag_7_price=3200,
        price_change_1=100,
        price_change_3=200,
        rolling_mean_3=3350,
        rolling_mean_7=3250,
        rolling_std_7=180,
    )

    print(
        f"Predicted next-day modal price: "
        f"₹{prediction:,.2f}/quintal"
    )
    