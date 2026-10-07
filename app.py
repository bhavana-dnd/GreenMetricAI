import os
import streamlit as st
import pandas as pd
import plotly.express as px
from greenmetricai.features import extract_code_metrics
from greenmetricai.model import GreenMetricModel
from greenmetricai.scoring import SustainabilityScorer
from greenmetricai.explain import global_shap

st.set_page_config(page_title="GreenMetricAI",page_icon="🌱",layout="wide")
st.title("🌱 GreenMetricAI — Real Data")
st.caption("Measured MBPP execution data → ML prediction → SHAP")
MODEL="models/greenmetric_model.joblib"
DATA="data/processed/mbpp_measured.csv"

if not os.path.exists(MODEL):
    st.error("Train the real-data model first: python scripts/train.py --data data/processed/mbpp_measured.csv")
    st.stop()

model=GreenMetricModel.load(MODEL)
a,b=st.tabs(["Analyze","SHAP"])

with a:
    code=st.text_area("Python code",height=300,value="def square_sum(n):\n    return sum(i*i for i in range(n))")
    if st.button("Analyze",type="primary"):
        m=extract_code_metrics(code)
        e=model.predict(m)
        s=SustainabilityScorer(model.energy_values).score(e)
        x,y,z,w=st.columns(4)
        x.metric("Predicted energy",f"{e:.6g} kWh")
        y.metric("Impact index",f"{s['impact_index']:.1f}/100")
        z.metric("Sustainability",f"{s['sustainability_score']:.1f}/100")
        w.metric("Class",s["classification"])
        st.subheader("Static features")
        st.dataframe(pd.DataFrame([m.to_dict()]).T.rename(columns={0:"value"}))
        st.subheader("Real-data validation")
        st.json(model.metrics)

with b:
    if os.path.exists(DATA):
        df=pd.read_csv(DATA)
        imp,*_=global_shap(model,df)
        fig=px.bar(imp.head(12).sort_values("mean_abs_shap"),
                   x="mean_abs_shap",y="feature",orientation="h",
                   title="SHAP importance from real measured data")
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(imp)
    else:
        st.warning("Measured dataset not found.")
