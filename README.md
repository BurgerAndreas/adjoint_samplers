<h1 align='center'>Adjoint-based Diffusion Samplers</h1>

<br>



Adjoint-based diffusion samplers is a new class of diffusion models for sampling energy-based Boltzmann distributions. 
Sampling from these Boltzmann distributions is defined by unnormalized energy functions, $E(x)$, without requiring ground-truth samples.
The repository contains the following adjoint-based diffusion samplers:

- [Adjoint Sampling](https://arxiv.org/abs/2504.11713) (ICML 2025)
- [Adjoint Schrödinger Bridge Sampler](https://arxiv.org/abs/2506.22565) (NeurIPS 2025 Oral)


Adjoint Schrödinger Bridge Sampler (ASBS) enables the use of arbitrary source distributions (e.g., harmonic priors) by solving a general Schrödinger Bridge problem. This is achieved through an alternating optimization scheme equivalent to Iterative Proportional Fitting, consisting of two matching objectives. 

Adjoint Matching (AM): Learns the drift $u_t$ by regressing against the energy gradient and a bias-correcting term from the previous stage:
$$
u^{(k)} := \arg \min \mathbb{E} \left[ \left\| u_t(X_t) + \sigma_t (\nabla E(X_1) + h^{(k-1)}(X_1)) \right\|^2 \right]
$$
where $h^{(0)} := 0$

Corrector Matching (CM): Learns the corrector $h$ (the score of the bridge potential) to debias the prior by matching the score of the base process:
$$
h^{(k)} := \arg \min \mathbb{E} \left[ \left\| h(X_1) - \nabla_{X_1} \log p^{base}(X_1|X_0) \right\|^2 \right]
$$

Gentlest Ascent Dynamics (GAD) Modification for Transition State SamplingTo adapt ASBS for finding transition states (index-1 saddle points) instead of minima, the energy gradient $\nabla E(x)$ in the Adjoint Matching objective is replaced with a Gentlest Ascent Dynamics (GAD) vector field, $v_{GAD}(x)$. This modifies the drift regression target to transport mass towards saddle points using the formulation:
$$
v_{GAD}(x) = -\nabla E(x) + 2 (v \cdot \nabla E(x)) v
$$
where $H(x)$ is the Hessian of the energy and $v$ is the eigenvector corresponding to the smallest eigenvalue of $H(x)$ (the direction of negative curvature). Consequently, the modified Adjoint Matching objective becomes:
$$
u^{(k)} := \arg \min \mathbb{E} \left[ \left\| u_t(X_t) + \sigma_t (v_{GAD}(X_1) + h^{(k-1)}(X_1)) \right\|^2 \right]
$$
where $h^{(k-1)}$ remains the corrector from the previous stage, ensuring the sampler accounts for the arbitrary prior while targeting the stationary points defined by the GAD field.

Note that AS is a special case of ASBS with memoryless condition.
For amortized conformer generation, please check [here](https://github.com/facebookresearch/adjoint_sampling).
For improved ASBS with exploration using low-dimensional projections of atomic coordinates known as collective variables (CVs), Well-Tempered Adjoint Schrödinger Bridge Sampler (WT-ASBS), please check [here](https://github.com/facebookresearch/wt-asbs).




## Installation

```bash
uv venv .venv --python 3.11
source .venv/bin/activate

# uv run train.py experiment=dw4_asbs
```

## Run overview

Toy examples
```bash
uv run train.py experiment=dw4_asbs
uv run train.py experiment=dw4_as 
uv run train.py experiment=lj13_asbs 
uv run train.py experiment=lj13_as 
uv run train.py experiment=lj55_asbs 
uv run train.py experiment=lj55_as 
```

Molecules
```bash
uv run train.py experiment=c3h4_asbs
uv run train.py experiment=isopropanol_asbs
```


## Demo
Run [`scripts/demo.sh`](https://github.com/facebookresearch/adjoint_samplers/blob/main/scripts/demo.sh) to generate similar demo figure in ASBS paper.

![](./assets/demo.png)

## Training DW and LJ energies
Run the following script to download the necessary reference .npy test splits for the synthetic energies into ./data (DW4, LJ13, LJ55) from the DEM GitHub repo for evaluation purposes:
```bash
bash scripts/download.sh
```

Training scripts to generate similar results in the papers can be found under
[`scripts`](https://github.com/facebookresearch/adjoint_samplers/blob/main/scripts).
Checkpoints and figures are saved under the folder `results`.
```bash
# energy.gad=True
uv run train.py experiment={dw4,lj13,lj55}_{asbs,as} seed=0,1,2 -m
```


## Citation
If you find this repository helpful, please consider citing our paper:
```bibtex
@inproceedings{liu2025asbs,
  title={{Adjoint Schr{\"o}dinger bridge sampler}},
  author={Liu, Guan-Horng and Choi, Jaemoo and Chen, Yongxin and Miller, Benjamin Kurt and Chen, Ricky T. Q.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
}
```

## License
This repository is licensed under the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/),
with some portions of the project subject to separate license terms:
[DEM](https://github.com/jarridrb/DEM),
[DDS](https://github.com/franciscovargas/denoising_diffusion_samplers),
and [bgflow](https://github.com/noegroup/bgflow)
are each licensed under the MIT License. Please refer to the respective repositories for details.