# quantum\_irops

QUBO-based airline passenger re-accommodation for irregular operations (IROPS).  
Formulates a Quadratic Unconstrained Binary Optimization problem that assigns disrupted passengers to alternate flights while respecting cabin capacity, priority (CVM), and Copa's Phase-2 Rules Set.

## Installation

```bash
# clone and install with uv (recommended)
git clone https://github.com/FaiBarazi/quantum_irops.git
cd quantum_irops
uv sync

# or with pip
pip install .
```

### Optional solver backends

| Extra | Package | Use |
|-------|---------|-----|
| `neal` | `dwave-neal` | D-Wave's SA sampler (faster than the built-in SA for large QUBOs) |
| `dwave` | `dwave-ocean-sdk` | Real QPU via D-Wave Leap |

```bash
uv sync --extra neal        # pip install ".[neal]"
uv sync --extra dwave       # pip install ".[dwave]"
```

## Quick start

```python
from pipeline import run_pipeline

assignments, unbooked = run_pipeline(
    pnr="PRMI_DM_ALL_PNRs.csv",
    cancelled="PRMI-DM_TARGET_FLIGHTS.csv",
    available="PRMI-DM-AVAILABLE_FLIGHTS.csv",
    batch_strategy="by_priority_tier",
    priority_bins=30,
    output_assignments="assignments.csv",
    output_unbooked="unbooked.csv",
)
```

See `run_pipeline_example.py` for a runnable example against the sample data in `notebooks/data/`.

## CLI

```bash
reaccom --pnr pnr.csv --cancelled target.csv --available avail.csv \
        --method sa --num-reads 500 --batch-strategy by_priority_tier
```

Solver choices: `sa` (built-in, no extra deps), `neal`, `dwave`.

```bash
reaccom --help
```

## Dashboard

The Shiny dashboard (`app.py`) visualises affected routes and passenger counts:

```bash
shiny run app.py
```

## Project layout

```
pipeline/           QUBO pipeline package
  candidates.py     candidate flight generation & filtering
  qubo.py           QUBO formulation + SA / neal / D-Wave solvers
  results.py        solution interpretation & capacity tracking
  runner.py         orchestrator + public run_pipeline() API
  config.py         all tunable dataclasses (QUBOWeights, etc.)
  cli.py            argparse CLI (entry point: reaccom)
run_pipeline_example.py   minimal runnable example
notebooks/          exploratory notebooks + sample CSV data
app.py              Shiny IROPS dashboard
```

## License

See [LICENSE](LICENSE).