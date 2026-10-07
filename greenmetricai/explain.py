import pandas as pd,shap
from .features import FEATURES
def global_shap(model,df,sample_size=300):
    X=df[FEATURES].dropna().sample(min(sample_size,len(df)),random_state=42)
    explainer=shap.TreeExplainer(model.rf)
    values=explainer.shap_values(X)
    imp=pd.DataFrame({"feature":FEATURES,"mean_abs_shap":abs(values).mean(axis=0)}).sort_values("mean_abs_shap",ascending=False)
    return imp,explainer,X,values
