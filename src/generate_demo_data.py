import os
import numpy as np
import pandas as pd

np.random.seed(42)
os.makedirs("data", exist_ok=True)

markets={"Indore":0,"Lucknow":1,"Delhi":2,"Mumbai":3}
varieties={"Alphonso":0,"Dashehari":1,"Kesar":2}
dates=pd.date_range("2021-01-01","2025-12-31",freq="7D")
base_market={"Indore":3600,"Lucknow":3900,"Delhi":4300,"Mumbai":4700}
base_variety={"Alphonso":1200,"Dashehari":300,"Kesar":700}

rows=[]
for date in dates:
    season=np.sin(2*np.pi*date.dayofyear/365.25)
    temp=27+7*np.sin(2*np.pi*((date.dayofyear-80)/365.25))+np.random.normal(0,1.5)
    rain=max(0,3+5*np.sin(2*np.pi*((date.dayofyear-150)/365.25))+np.random.normal(0,4))
    humidity=np.clip(65+15*np.sin(2*np.pi*((date.dayofyear-120)/365.25))+np.random.normal(0,5),25,95)
    for market in markets:
        for variety in varieties:
            arrival=max(100,3500+2200*max(0,season)+np.random.normal(0,400))
            price=(base_market[market]+base_variety[variety]-0.18*arrival+
                   20*(temp-27)-7*rain+3*humidity+500*(-season)+np.random.normal(0,180))
            rows.append([date,market,variety,arrival,temp,rain,humidity,max(1200,price)])

df=pd.DataFrame(rows,columns=["date","market","variety","arrival_kg","temperature_c","rainfall_mm","humidity_pct","modal_price"])
df.to_csv("data/mango_demo.csv",index=False)
print(f"Created {len(df):,} demo rows.")
