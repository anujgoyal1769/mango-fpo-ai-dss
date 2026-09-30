import pandas as pd


def calculate_scenarios(
    predicted_price,
    current_price,
    quantity_kg,
    transport_cost,
    storage_cost
):
    """
    Calculate financial scenarios for one selected market.

    Prices are ₹ per quintal.
    Quantity is kilograms.
    Transport and storage are scenario costs in ₹.
    """

    quantity_quintal = quantity_kg / 100

    current_revenue = (
        current_price * quantity_quintal
    )

    predicted_revenue = (
        predicted_price * quantity_quintal
    )

    current_after_transport = (
        current_revenue - transport_cost
    )

    predicted_after_transport = (
        predicted_revenue - transport_cost
    )

    predicted_after_storage = (
        predicted_revenue
        - transport_cost
        - storage_cost
    )

    revenue_difference = (
        predicted_revenue - current_revenue
    )

    return {
        "quantity_quintal": quantity_quintal,
        "current_revenue": current_revenue,
        "predicted_revenue": predicted_revenue,
        "current_after_transport": current_after_transport,
        "predicted_after_transport": predicted_after_transport,
        "predicted_after_storage": predicted_after_storage,
        "revenue_difference": revenue_difference,
    }


def add_market_scenario_metrics(
    market_df,
    quantity_kg,
    transport_cost,
    storage_cost
):
    """
    Add transparent revenue scenario columns
    to a market prediction DataFrame.

    Transport and storage are uniform scenario costs
    supplied by the FPO user.
    """

    result = market_df.copy()

    quantity_quintal = quantity_kg / 100

    result["gross_revenue"] = (
        result["predicted_next_price"]
        * quantity_quintal
    )

    result["after_transport"] = (
        result["gross_revenue"]
        - transport_cost
    )

    result["after_transport_storage"] = (
        result["gross_revenue"]
        - transport_cost
        - storage_cost
    )

    return result