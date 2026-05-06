# Cognitive Modeling Project

Bayesian models of handwriting response times — a hierarchical regression approach and a Drift Diffusion Model (DDM) approach, both fit with Stan via `cmdstanpy`.

## File Structure

```
Project/
├── README.md
├── CogModel_Template.tex             # LaTeX writeup template
├── CogModel_Template.pdf             # Compiled writeup template
│
├── DDM_Models/
│   ├── diffusion_models_of_writing_time.ipynb   # Main DDM notebook
│   ├── writing_dataset.csv                      # Data (auto-loaded by the notebook)
│   ├── models/                                  # Stan models used in the notebook
│   │   ├── model1_complete_pooling.stan
│   │   ├── model2_narrow_prior.stan
│   │   ├── model2_wide_prior.stan
│   │   ├── model2_partial_pooling_unconstrained.stan
│   │   └── model3_theory_constrained.stan
│   └── drafts/                                  # Earlier drafts / scratch work
│
└── Regression_Models/
    ├── handwriting.ipynb                        # Main regression notebook
    ├── handwriting.stan                         # Stan model
    └── WritingStudyFastSlow_long.csv            # Data (auto-loaded by the notebook)
```

The data CSV lives alongside each notebook and is loaded automatically when the cells run — no manual download or path adjustment is required.

## Setup

Both notebooks require Python 3.10+ and the following packages:

```bash
pip install cmdstanpy pandas numpy matplotlib seaborn arviz jupyter
```

`cmdstanpy` needs CmdStan installed once:

```python
import cmdstanpy
cmdstanpy.install_cmdstan()
```

## Running the notebooks

### `DDM_Models/diffusion_models_of_writing_time.ipynb`

```bash
cd DDM_Models
jupyter notebook diffusion_models_of_writing_time.ipynb
```

Then in Jupyter: **Cell → Run All** (or **Kernel → Restart & Run All**).

The notebook reads `writing_dataset.csv` from the same folder and compiles the `.stan` files in `models/` on first run.

### `Regression_Models/handwriting.ipynb`

```bash
cd Regression_Models
jupyter notebook handwriting.ipynb
```

Then in Jupyter: **Cell → Run All**.

The notebook reads `WritingStudyFastSlow_long.csv` from the same folder and compiles `handwriting.stan` on first run.

> Run each notebook from its own directory so the relative paths to the CSV and `.stan` files resolve correctly. Stan compilation and MCMC sampling can take a few minutes the first time through.
