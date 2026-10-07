# GreenMetricAI — Real-Data Editio

The experimental dataset is built from the real **MBPP (Mostly Basic Python Problems)** benchmark. MBPP contains roughly 1,000 crowd-sourced Python programming problems, each with a reference implementation and automated tests. The Google Research repository also provides a sanitized subset. See:
https://github.com/google-research/google-research/tree/master/mbpp

Recommended dataset:
- `sanitized-mbpp.json`
- real human/crowd-sourced programming tasks
- reference Python solutions
- test cases
- CC BY 4.0 according to the published dataset metadata

## Real-data pipeline

```text
Google Research MBPP
        |
        v
Download sanitized-mbpp.json
        |
        v
Select valid Python reference solutions
        |
        v
Static analysis
        |
        v
Repeat real benchmark tests
        |
        v
CodeCarbon measurement
        |
        v
Measured dataset
        |
        v
RF + XGBoost
        |
        v
SHAP
        |
        v
LLM optimization
        |
        v
Correctness tests + energy re-measurement
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Download real benchmark data

```bash
python scripts/download_mbpp.py --variant sanitized
```

This downloads the official sanitized MBPP JSON from Google Research. No generated/synthetic data is created.

## 2. Build measured dataset

Start small:

```bash
python scripts/collect_mbpp.py --tasks 50 --repeats 20
```

For the full sanitized benchmark:

```bash
python scripts/collect_mbpp.py --tasks all --repeats 50
```

Each task is:
1. parsed,
2. statically analyzed,
3. executed against its real reference tests,
4. repeated to make the measurement window large enough,
5. measured with CodeCarbon,
6. saved with hardware/runtime/energy/carbon metadata.

## 3. Train

```bash
python scripts/train.py --data data/processed/mbpp_measured.csv
```

## 4. SHAP

```bash
python scripts/explain.py --data data/processed/mbpp_measured.csv
```

## 5. Dashboard

```bash
streamlit run app.py
```

## Important measurement design

Very short functions can consume too little energy for stable measurement. The collector therefore repeats the **real benchmark tests** many times inside the measurement window. The source code and test cases remain real; only the number of repetitions is an experimental measurement parameter.

Do not report repeated execution energy as "energy of one function call" without dividing by the number of repetitions and validating that the result is above the measurement noise floor.

For publication-quality results:
- run multiple independent measurement sessions,
- record CPU/RAM/OS/Python versions,
- keep power mode and background load controlled,
- use medians and confidence intervals,
- randomize task order,
- validate across more than one machine if possible.

## Security

MBPP includes executable code and tests. The collector is intended for a controlled research machine. Do not run untrusted code on a normal host. For a public service, execute benchmarks and LLM-generated candidates in a hardened sandbox/container/VM with CPU, memory, filesystem, network and timeout restrictions.

## What is NOT included

There is deliberately:
- no `generate_synthetic.py`
- no synthetic CSV
- no hard-coded energy labels
- no fabricated R²
- no hard-coded "GSSS reduction" values

The ML target is measured `energy_kwh`.
