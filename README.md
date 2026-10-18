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

The **Optimize LLM code** tab accepts Python code and behavior checks, asks
Gemini for up to three alternatives, runs the original and candidates against
those checks, and ranks passing candidates by GreenMetricAI's predicted energy.
The `.env.example` key is a placeholder; the app reads `GEMINI_API_KEY` from
the process environment or a project `.env` file. Copy `.env.example` to `.env`
and replace the placeholder with your Gemini API key, or paste the key into the
masked field on the optimizer page. The page never writes a key entered there
to disk. In PowerShell, you can also set it before launch:

```powershell
$env:GEMINI_API_KEY = "your_key_here"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The optimizer also computes local SHAP values for the original and accepted
alternatives. These feature contributions explain the Random Forest's
log-energy prediction; they do not explain measured energy or establish causal
effects. The separate SHAP tab provides global feature importance across the
available measured examples.

The current workflow reports predictions, not measured savings, and does not
estimate the LLM call's energy or water use. Only execute code you trust; the
check runner uses a timeout but is not a security sandbox.

## Pull request checks

GitHub Actions runs the project test suite for pull requests and pushes to
`main`. Each run publishes a Markdown report with the test result and the
checked-in paired MBPP energy evidence, including its confidence interval.
That experiment is labeled as historical evidence; CI does not claim to
measure the PR's energy on a shared hosted runner. A candidate is eligible for
human review only when its recorded checks pass and its measured confidence
interval clears the configured minimum reduction. No code is applied
automatically. The test job receives no API keys; comments are posted only for
same-repository pull requests. Fork pull requests can read the report in the
Actions run summary and artifact.

The workflow uses `requirements-ci.txt`, a minimal dependency set for running
the unit tests. It does not require Gemini credentials or benchmark downloads.

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

### Sequential paired energy validation

`python -m scripts.compare_optimization` uses paired runs to decide whether a
candidate's measured energy change is larger than ordinary run-to-run noise.
Each pair measures the original and optimized code with the same duration
protocol; execution order alternates between pairs. It starts with five pairs
and can collect up to fifteen.

For each valid pair, the validator computes the log ratio
`log(optimized_energy / original_energy)`. It builds a Student-t interval on
those paired log ratios and applies a Bonferroni adjustment for the planned
sequential checks. It stops when the interval establishes either a measured
reduction above 3% or an effect below that threshold. If neither conclusion is
supported after fifteen valid pairs, it reports the result as inconclusive; if
too few runs return valid energy readings, it reports insufficient
measurements. Results are saved in `measured_validation` in the task's JSON
artifact and shown on the Energy Validation dashboard.

This is evidence for the measured task and machine, not proof that every
program will save energy. The interval assumes the paired log ratios are
approximately independent and normally distributed; repeat the experiment in
independent sessions and on other hardware before making broad claims.

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
