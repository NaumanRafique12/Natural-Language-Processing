from fastapi import FastAPI, Request
# from pydantic import BaseModel
import uvicorn
import numpy as np
# https://airtable.com/api
from airtable import airtable
at = airtable.Airtable(base_id= 'appng9B3mdMCl2h8m', api_key='patpRe32y2nD6rTI5.249b90585f3ed62b11c985bc1832581c9677b4301770ddf9d66e6a08c176bf10',table_name="Predictions")

app = FastAPI()

import streamlit as st
import pickle
import numpy as np

import sklearn

# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))

df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")


# brand
@app.post("/predict")
async def read_root(request: Request):
    data = await request.json()
    print(data)
    try:
        company = data['Company']
        print(data)

        type = data['TypeName']
        ram = data['Ram']

        weight = data['Weight']
        touchscreen = data['Touchscreen']
        ips = data['IPS']
        screen_size = data['Screen Size']
        resolution = data['Screen Resolution']
       # ,['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
        cpu = data['CPU']
        cpu_size = data['Cpu Size']
        hdd = data['HDD']
       # ,[0,128,256,512,1024,2048])
        ssd = data['SSD']
        gpu = data['GPU']
        gpu_size = data['Gpu Size']
        os = data['OS']
        ppi = None
        if touchscreen == 'Yes':
            touchscreen = 1
        else:
            touchscreen = 0
        if ips == 'Yes':
            ips = 1
        else:
            ips = 0

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
        pred = pipe.predict(
           [[company, type, ram, weight, touchscreen, ips, ppi, cpu, cpu_size, gpu_size, hdd, ssd, gpu, os]])
        text= {"The predicted price of this configuration is ": str(int(np.exp(pred[0])))}
    except:
       text= {"errr":"erro"}
    finally:
        # pass
        data['Output']=int(np.exp(pred[0]))
        at.insert(data)
    return text
if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8080, reload=True)