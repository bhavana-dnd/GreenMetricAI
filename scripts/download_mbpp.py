import argparse,urllib.request
from pathlib import Path
from greenmetricai.mbpp import MBPP_RAW_URL

p=argparse.ArgumentParser()
p.add_argument("--variant",choices=["sanitized"],default="sanitized")
p.add_argument("--output",default="data/raw/sanitized-mbpp.json")
a=p.parse_args()
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
print("Downloading real MBPP dataset from Google Research...")
urllib.request.urlretrieve(MBPP_RAW_URL,a.output)
print(f"Saved real dataset to {a.output}")
