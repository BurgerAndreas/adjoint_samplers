# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Adjoint-based diffusion samplers for energy-based Boltzmann distributions without ground-truth samples. Implements Adjoint Sampling (AS) and Adjoint Schrödinger Bridge Sampler (ASBS) algorithms, with extensions for transition state discovery using Gentlest Ascent Dynamics (GAD).

## Commands

**Run training:**
```bash
uv run train.py experiment=<config>
```

**Quick demos:**
```bash
uv run train.py experiment=demo_asbs
uv run train.py experiment=dw4_asbs    # 4D double-well
uv run train.py experiment=lj13_asbs   # 13-particle Lennard-Jones
uv run train.py experiment=lj55_asbs   # 55-particle Lennard-Jones
```

**Multi-seed runs:**
```bash
uv run train.py experiment=dw4_asbs seed=0,1,2 -m
```

**Override config parameters:**
```bash
uv run train.py experiment=dw4_asbs energy.gad=True adj_num_epochs_per_stage=300
```

**Download test data:**
```bash
bash scripts/download.sh
```

**Linting (pre-commit):**
```bash
pre-commit run --all-files
```

## Architecture

### Core Training Flow (`train.py` → `train_loop.py`)

1. **Hydra config** loads experiment from `configs/experiment/` which composes problem, model, SDE, source, matcher configs
2. **Components instantiated**: energy function, source distribution, SDE, controller (neural net), optional corrector
3. **Training alternates** between Adjoint Matching (AM) and Corrector Matching (CM) stages
4. **Each epoch**: populate replay buffer with SDE trajectories, train networks via MSE loss on score targets
5. **Evaluation**: generate samples, compute metrics, cluster analysis, log to W&B

### Key Modules

- **`adjoint_samplers/components/`**: Core algorithm pieces
  - `sde.py` - SDE implementations (VE, VP, Graph variants, ControlledSDE)
  - `matcher.py` - AdjointMatcher, CorrectorMatcher (backward adjoint simulation)
  - `model.py` - Neural networks (FourierMLP, EGNN)
  - `buffer.py` - Replay buffer
  - `evaluator.py` - Metrics and visualization

- **`adjoint_samplers/energies/`**: Energy functions
  - `base_energy.py` - BaseEnergy with auto-gradient/Hessian, GAD modifications
  - Implementations: double_well, lennard_jones, scine (quantum chemistry)

- **`adjoint_samplers/utils/`**: Training, evaluation, clustering utilities
  - `clustering.py` - RMSD matrix, HDBSCAN, Density Peaks
  - `frequency_analysis.py` - Vibrational analysis for molecules
  - `eval_utils.py` - Visualization and metric computation

### Configuration System

Hydra-based modular configs in `configs/`:
- `experiment/` - Complete experiment presets (dw4_asbs, lj13_as, c3h4_asbs, etc.)
- `problem/` - Energy function definitions
- `model/` - Architecture configs (egnn, fouriermlp)
- `sde/` - SDE type configs (graph_ve, graph_vp, ve, vp)
- `source/` - Source distributions (harmonic, gaussian, delta)
- `matcher/` - Matcher configs (adjoint_ve, adjoint_vp, corrector)

Key parameters: `nfe` (SDE steps), `sigma_max/min` (noise schedule), `train_batch_size`, `eval_freq`, `adj_num_epochs_per_stage`, `ctr_num_epochs_per_stage`

### Energy Function Pattern

All energies inherit from `BaseEnergy` and implement `_energy()`. The base class provides:
- Automatic gradient via autograd
- Hessian computation (full or Hessian-vector products)
- GAD vector field for transition state sampling
- Terminal cost functions with bias correction
