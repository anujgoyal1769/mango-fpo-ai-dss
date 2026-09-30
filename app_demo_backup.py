import os, joblib, pandas as pd, streamlit as st, plotly.express as px

st.set_page_config(page_title="Mango FPO AI DSS",page_icon="🥭",layout="wide")
DATA="data/mango_demo.csv"; MODEL="models/price_model.joblib"

st.title("🥭 Mango FPO AI Decision Support System")
st.caption("Research prototype — replace demo data with verified real-world data before reporting results.")

@st.cache_data
def load(): return pd.read_csv(DATA,parse_dates=["date"]).sort_values("date")
df=load()

if not os.path.exists(MODEL):
    st.error("Model missing. Run: python src/train_model.py")
    st.stop()
bundle=joblib.load(MODEL); model=bundle["model"]; features=bundle["features"]

st.sidebar.header("FPO Inputs")
market=st.sidebar.selectbox("Market",sorted(df.market.unique()))
variety=st.sidebar.selectbox("Mango variety",sorted(df.variety.unique()))
qty=st.sidebar.number_input("Quantity (kg)",100,100000,1000,100)
transport=st.sidebar.number_input("Transport cost (₹)",0.0,1000000.0,2500.0,500.0)
storage=st.sidebar.number_input("Storage cost (₹)",0.0,1000000.0,1000.0,250.0)

m=df[(df.market==market)&(df.variety==variety)].copy()
latest=m.iloc[-1]
row=pd.DataFrame([{
"year":latest.date.year,"month":latest.date.month,"dayofyear":latest.date.dayofyear,
"arrival_kg":latest.arrival_kg,"temperature_c":latest.temperature_c,
"rainfall_mm":latest.rainfall_mm,"humidity_pct":latest.humidity_pct,
"market_code":bundle["market_map"][market],"variety_code":bundle["variety_map"][variety]}])[features]
pred=float(model.predict(row)[0])
revenue=pred*qty/100

a,b,c,d=st.columns(4)
a.metric("Latest modal price",f"₹{latest.modal_price:,.0f}/quintal")
b.metric("Predicted price",f"₹{pred:,.0f}/quintal")
c.metric("Estimated gross revenue",f"₹{revenue:,.0f}")
d.metric("After transport",f"₹{revenue-transport:,.0f}")

st.subheader("📈 Price Trend")
fig=px.line(m,x="date",y="modal_price",title=f"{market} — {variety}")
st.plotly_chart(fig,use_container_width=True)

st.subheader("🏪 Market Comparison")
latest_date=df.date.max()
comp=df[(df.date==latest_date)&(df.variety==variety)].copy()
comp["estimated_revenue_after_transport"]=comp.modal_price*qty/100-transport
st.dataframe(comp[["market","modal_price","arrival_kg","estimated_revenue_after_transport"]].rename(columns={
"market":"Market","modal_price":"Current Price (₹/quintal)","arrival_kg":"Arrival (kg)",
"estimated_revenue_after_transport":"Estimated Revenue After Transport (₹)"}),use_container_width=True,hide_index=True)

st.subheader("🤖 Decision-Support Insight")
if pred>latest.modal_price*1.03:
    st.info("The prototype predicts a higher price than the latest observation. Compare this forecast with verified storage cost, spoilage risk, transport cost and FPO constraints before deciding.")
elif pred<latest.modal_price*0.97:
    st.info("The prototype predicts a lower price than the latest observation. Compare immediate selling with verified demand, logistics and storage conditions.")
else:
    st.info("The prototype predicts a relatively stable price. Compare markets and logistics costs before deciding.")

with st.expander("⚠️ Demo-data notice"):
    st.write("This application currently uses synthetic demonstration data. Its predictions are not real-market evidence and must not be presented as research findings.")

st.subheader("Dataset Preview")
st.dataframe(df.tail(20),use_container_width=True,hide_index=True)
