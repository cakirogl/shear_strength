#import pip
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
#from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
#import warnings
#warnings.filterwarnings("ignore", message="numpy.dtype size changed")
#warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
#pip.main(['install', '--upgrade', "numpy"])

url = "https://raw.githubusercontent.com/cakirogl/shear_strength/main/dataset.csv"
#model_selector = st.selectbox('**Predictive model**', ["XGBoost", "LightGBM", "CatBoost", "Random Forest"])
#Because of an error in streamlit we removed CatBoost
model_selector = st.selectbox('**Predictive model**', ["XGBoost", "LightGBM", "Random Forest"])
df = pd.read_csv(url);
x, y = df.iloc[:, :-1], df.iloc[:, -1]
scaler = MinMaxScaler();
x=scaler.fit_transform(x);
input_container = st.container()
#ic1,ic2,ic3 = input_container.columns(3)
ic1,ic2 = input_container.columns(2)
FRP_types=["CFRP", "BFRP", "GFRP", "AFRP"]
with ic1:
    b = st.number_input("**Beam width [mm]:**",min_value=100.0,max_value=400.0,step=10.0,value=200.0)
    d = st.number_input("**Beam depth [mm]:**",min_value=120.0,max_value=1691.0,step = 10.0,value=400.0)
    a_d = st.number_input("**Span to depth ratio**", min_value=0.5, max_value=2.4, step=0.1, value=1.0)
    fc = st.number_input("**f$_c^{\prime}$ [MPa]:**", min_value=20.0, max_value=70.0, step=1.0, value=36.0)
with ic2:
    rho = st.number_input("**Longitudinal reinf. ratio [\%]:**", min_value=0.26, max_value=2.68, step=0.1, value=1.5)
    Ef = st.number_input("**FRP modulus of elasticity [GPa]:**", min_value=38.0, max_value=150.0, step=1.0, value=50.0)
    FRP=st.selectbox("**FRP type:**", options=FRP_types)
    
if FRP=="CFRP":
    FRP=2
elif FRP=="BFRP":
    FRP=1
elif FRP=="GFRP":
    FRP=3
elif FRP=="AFRP":
    FRP=0
#with ic3:
#    cca = st.number_input("**Ceramic coarse aggregate [kg/m$^3$]:**", min_value=0.0, max_value=32.13, step=1.0, value=12.85)
#    cca_spec_gr = st.number_input("**Ceramic coarse aggregate specific gravity:**", min_value = 0.0, max_value = 2.0, step = 0.1, value = 1.9)
#    cca_abs_cap = st.number_input("**Ceramic coarse aggregate Absorption Capacity [%]:**", min_value=0.0, max_value = 14.23, step = 0.2, value = 14.0)
#    cca_dens = st.number_input("**Ceramic coarse aggregate density [kg/m$^3$]:**", min_value=0.0, max_value = 1114.15, step = 100.0, value = 1114.0)

new_sample=np.array([[b, d, a_d, fc, rho, Ef, FRP]],dtype=object)
new_sample=pd.DataFrame(new_sample, columns=df.columns[:-1])
new_sample=scaler.transform(new_sample);
if model_selector=="LightGBM":
    model=LGBMRegressor(random_state=0, verbose=-1)
    model.fit(x,y)
elif model_selector=="XGBoost":
    model=XGBRegressor(random_state=0)
    model.fit(x, y)
if model_selector=="CatBoost":
    model=CatBoostRegressor(random_state=0, logging_level="Silent")
    model.fit(x,y)
elif model_selector=="Random Forest":
    model=RandomForestRegressor(random_state=0)
    model.fit(x, y)

with ic2:
    #st.write(f":blue[**Compressive strength = **{model_c.predict(new_sample)[0]:.3f}** MPa**]\n")
    st.write(f":blue[**Shear strength = **{model.predict(new_sample)[0]:.3f}** kN**]\n")