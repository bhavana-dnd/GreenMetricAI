import argparse,pandas as pd
from greenmetricai.model import GreenMetricModel

p=argparse.ArgumentParser()
p.add_argument("--data",required=True)
p.add_argument("--target",default="energy_kwh")
p.add_argument("--model-dir",default="models")
a=p.parse_args()

df=pd.read_csv(a.data)
m=GreenMetricModel()
print(m.fit(df,a.target,"task_id"))
m.save(f"{a.model_dir}/greenmetric_model.joblib")
