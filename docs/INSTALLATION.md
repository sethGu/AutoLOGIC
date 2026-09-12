# Installation and use

## Minimal offline demonstration

Tested platform: Windows, Python 3.12.14, x86-64 CPU. The demo requires no GPU,
API key, pretrained weights, or external data download. A normal desktop with
4 GB RAM is sufficient for the 240-row example. Linux and macOS commands are
provided below but have not been tested for this release.

Create and activate a virtual environment before installing dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-demo.txt
python demo/run_demo.py
```

If PowerShell activation is disabled, use `.\.venv\Scripts\python.exe` directly
for the install and run commands. Activation is optional when using that path.

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python demo/run_demo.py
```

Pinned numerical packages: NumPy 1.26.4, pandas 2.2.3, SciPy 1.15.2,
scikit-learn 1.6.1, joblib 1.6.0, threadpoolctl 3.6.0. Transitive dependencies
are installed by pip. The first install typically takes 1–5 minutes with a
working package-index connection; download speed and existing caches affect
this estimate. The installation and example were verified in a fresh virtual
environment. See `demo/README.md` for input details and checked outputs.

## Full LLM workflows

Install `requirements.txt` in a separate Python 3.10+ environment. Optional
benchmark libraries are listed in `requirements-optional.txt`. These broader
requirements specify supported ranges; they are not a complete lock file for
every historical experiment. Full installation time and memory use depend on
the selected learners and dataset. The minimal demo's timing does not apply to
the full benchmark. Neural routes may use a CUDA-compatible PyTorch installation;
there is no GPU requirement for the offline demo.

```bash
python -m pip install -r requirements.txt
```

Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` in the process environment. The full
runners use the model alias `gpt-3.5-turbo`. Availability depends on the endpoint.
The endpoint must support the requested model and generation arguments. API
calls may incur charges. No credentials are distributed in this repository.

The four principal entry points are:

```bash
python autologic/ensemble/classification_ensemble/classification_auto_ensemble.py --help
python autologic/ensemble/multiclassification_ensemble/multiclassification_auto_ensemble.py --help
python autologic/ensemble/regression_ensemble/regression_auto_ensemble.py --help
python autologic/ensemble/cluster_ensemble/cluster_auto_ensemble_hidden_evaluator.py --help
```

The principal classification and regression runners look for CSV inputs under
`autologic/data/csv_data/` and serialized legacy inputs under `autologic/data/`.
Place `<dataset>.csv` and its optional `<dataset>-description.txt` there. Pass
the filename stem with `--dataset` and explicitly select `--llm gpt-3.5-turbo`.
Other reproduction runners may support environment-based data directories;
consult the selected loader and `data/README.md` before a benchmark run.

`--task-spec` accepts a JSON task specification; examples are under `configs/`.
For the offline binary sample with full generation enabled, copy
`demo/data/demo_binary.csv` to `autologic/data/csv_data/demo_binary.csv` and use
`demo_binary` as the classification dataset.
This online path requires the full dependencies and a supported remote model;
it was not executed in the offline verification.

## Verification

```bash
python -m unittest discover -s tests -p "test_*.py"
python scripts/check_release.py
```

Run these commands after installing the demo requirements so the numerical
demonstration tests are executed. Without those dependencies they are skipped.
