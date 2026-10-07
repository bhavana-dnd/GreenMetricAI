import argparse,pandas as pd
from greenmetricai.model import GreenMetricModel
from greenmetricai.explain import global_shap

p=argparse.ArgumentParser()
p.add_argument("--data",required=True)
p.add_argument("--model",default="models/greenmetric_model.joblib")
a=p.parse_args()
df=pd.read_csv(a.data)
m=GreenMetricModel.load(a.model)
imp,*_=global_shap(m,df)
print(imp.to_string(index=False))
imp.to_csv("reports/shap_importance.csv",index=False)
