import os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df=pd.read_csv("data/mango_demo.csv",parse_dates=["date"]).sort_values("date")
market_map={m:i for i,m in enumerate(sorted(df.market.unique()))}
variety_map={v:i for i,v in enumerate(sorted(df.variety.unique()))}
df["year"]=df.date.dt.year
df["month"]=df.date.dt.month
df["dayofyear"]=df.date.dt.dayofyear
df["market_code"]=df.market.map(market_map)
df["variety_code"]=df.variety.map(variety_map)

features=["year","month","dayofyear","arrival_kg","temperature_c","rainfall_mm","humidity_pct","market_code","variety_code"]
split=int(len(df)*0.8)
train,test=df.iloc[:split],df.iloc[split:]

model=RandomForestRegressor(n_estimators=300,random_state=42,min_samples_leaf=2,n_jobs=-1)
model.fit(train[features],train.modal_price)
pred=model.predict(test[features])
mae=mean_absolute_error(test.modal_price,pred)
rmse=np.sqrt(mean_squared_error(test.modal_price,pred))
r2=r2_score(test.modal_price,pred)

os.makedirs("models",exist_ok=True)
joblib.dump({"model":model,"features":features,"market_map":market_map,"variety_map":variety_map,
             "metrics":{"MAE":mae,"RMSE":rmse,"R2":r2}},"models/price_model.joblib")
print(f"MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.4f}")
