from pathlib import Path
import joblib,pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor
from .features import FEATURES

class GreenMetricModel:
    def __init__(self):
        self.rf=RandomForestRegressor(n_estimators=300,max_depth=18,min_samples_leaf=2,random_state=42,n_jobs=-1)
        self.xgb=XGBRegressor(n_estimators=400,max_depth=7,learning_rate=.04,subsample=.9,colsample_bytree=.9,
                              objective="reg:squarederror",random_state=42,n_jobs=4)
        self.weights={"rf":.45,"xgb":.55}; self.metrics={}; self.energy_values=None

    def fit(self,df,target="energy_kwh",group_col="task_id"):
        df=df.dropna(subset=FEATURES+[target]).copy()
        X,y=df[FEATURES].astype(float),df[target].astype(float)
        self.energy_values=y.values
        g=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42)
        tr,te=next(g.split(X,y,groups=df[group_col]))
        Xtr,Xte=X.iloc[tr],X.iloc[te]; ytr,yte=y.iloc[tr],y.iloc[te]
        self.rf.fit(Xtr,ytr); self.xgb.fit(Xtr,ytr)
        a,b=self.rf.predict(Xte),self.xgb.predict(Xte); p=.45*a+.55*b
        def m(y,p): return {"r2":float(r2_score(y,p)),"mae":float(mean_absolute_error(y,p)),
                            "rmse":float(mean_squared_error(y,p)**.5)}
        self.metrics={"rf":m(yte,a),"xgb":m(yte,b),"ensemble":m(yte,p),
                      "train_rows":len(ytr),"test_rows":len(yte),"test_tasks":int(df.iloc[te][group_col].nunique())}
        return self.metrics

    def predict(self,metrics):
        row=pd.DataFrame([metrics.to_dict() if hasattr(metrics,"to_dict") else metrics])
        return float(.45*self.rf.predict(row[FEATURES])[0]+.55*self.xgb.predict(row[FEATURES])[0])

    def save(self,path="models/greenmetric_model.joblib"):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        joblib.dump({"rf":self.rf,"xgb":self.xgb,"weights":self.weights,
                     "metrics":self.metrics,"energy_values":self.energy_values},path)

    @classmethod
    def load(cls,path):
        o=cls(); p=joblib.load(path)
        o.rf=p["rf"];o.xgb=p["xgb"];o.weights=p["weights"];o.metrics=p["metrics"];o.energy_values=p["energy_values"]
        return o
