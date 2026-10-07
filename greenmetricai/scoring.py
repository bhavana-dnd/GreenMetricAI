import numpy as np
class SustainabilityScorer:
    def __init__(self,energy_distribution):
        x=np.asarray(energy_distribution,float)
        self.p05,self.p95=np.percentile(x,[5,95])
    def score(self,energy):
        impact=50 if self.p95<=self.p05 else np.clip((energy-self.p05)/(self.p95-self.p05),0,1)*100
        s=100-impact
        label="Green" if s>=75 else "Moderate" if s>=50 else "High impact" if s>=25 else "Critical"
        return {"energy_kwh":float(energy),"impact_index":float(impact),
                "sustainability_score":float(s),"classification":label}
