# Transition state discovery with diffusion-based samplers

### Transition state discovery for unknown reactions
In applications like catalysis we would often like to know the reaction rates of unknown reactions. Unwanted side reactions can produce unintended molecules, poisoning the catalyst.

Transition state theory allows us to efficiently compute an approximation to the reaction rate from properties at the transition state, reactant, and product geometry. The crux is finding the transition states. From a transition state, one can get the reactant and product comparatively easily by running relaxations (internal reaction coordinate, IRC).

The problem with classical pipelines to find transition states is that they either presuppose that the reaction is already known, by requiring the reactant and product (in the case for double ended methods like NEB), or require reaction coordinates (also known as collective variables) to be tractable (like in the case of metadynamics or umbrella sampling).

Ideally, we want a method that
- Can find a diverse set of transition states
- Requires no prior knowledge, like reactant and product geometries
- Can generalise across molecules
- Can be trained without hard-to-get data


### Diffusion-based samplers
A similar problem of finding low-energy geometries without data has been tackled with diffusion-based samplers. In the following we show how a diffusion-based samplers can be used to find transition states.

Note that most samplers try to sample _proportional_ to the Boltzmann distribution. Here, sampling each mode with a well-calibrated probability is not of importance for us. We mainly care about discovering a diverse set of modes. We (ab)use the distribution formulation as an exploration tool. We could also use reinforcement learning with a binary reward (proposed structure is a TS or not).


### Transition states are index-1 saddle points of the potential energy surface
In two dimensions a saddle point is where the force (first derivative) is zero, and one Hessian eigenvalue is positive and one is negative (second derivative). That is to say we have a minimum in one direction but a maximum in another direction. In higher dimensions a transition state is an index-1 saddle point, meaning it is a maximum in exactly one direction and a minimum in all other directions. That is the same as having exactly one negative eigenvalue of the Hessian, or one negative frequency.


### A vector field to find transition states

Diffusion-based samplers aim to sample from a Boltzmann-style distribution of the form $\propto e^{-\gamma E}$. For low-energy geometries like conformers, $E(x)$ is the energy of a configuration. To sample index-1 saddle points instead of minima, we need a new pseudopotential $E(x)$ that is smooth and minimal exactly at the transition states.

Unfortunately, we do not have a pseudopotential for transition states. But we know about a vector field that locally converges to transition states as it's stable points: the gentlest ascent dynamics (GAD).
$$
\mathbf{F}_{GAD}(\mathbf{x})
= -\nabla E(\mathbf{x})
+ 2\,(\nabla E(\mathbf{x}), \mathbf{v}_1(\mathbf{x}))\,\mathbf{v}_1(\mathbf{x}),
$$
where $\mathbf{v}_1(\mathbf{x})$ is the eigenvector of the Hessian $\nabla^{2} E(\mathbf{x})$ associated with the smallest eigenvalue.

Note that the vector field and it's potential are only defined locally. The "potential" that produces the GAD vector field has discontinuities $\pm \inf$ where the smallest and second smallest eigenvalue of the Hessian change places.

Fortunately, some diffusion-based samplers only require the score $\Delta_x ln p(x) = -\gamma \Delta_x U(x) $ at the final timestep $t = 1 \hat{=} data$ without explicitly requiring the potential $U(x)$, like adjoint sampling:
$$
\mathcal{L}_{\mathrm{RAM}}(u)
= \int_0^1 \frac{1}{\sigma(t)^2}\,
\mathbb{E}_{X_t \sim p_{t|1}^{\text{base}},\, X_1 \sim p_1^{\bar{u}}}
\left[
\frac{1}{2}
\left\lVert
u(X_t, t) + \sigma(t)\nabla g(X_1)
\right\rVert^2
\right] dt
$$
$$
g(x) = \log p_1^{\text{base}}(x) + \beta E(x)
$$

Our insight is that we can directly regress our sampler onto the GAD vector field.
$$
\nabla g(X_1) = \nabla \log p_1^{\text{base}}(X_1) + \beta \mathbf{F}_{GAD}(\mathbf{x})
$$
We will gloss over points were the potential and vector field diverge by simply clipping the score. Score clipping is a common trick that is also necessary when sampling low-energy states, as the energy diverges to $\inf$ as atoms get close to each other.

References:
- Hessian Interatomic Potentials: https://arxiv.org/abs/2509.21624
- Adjoint sampling: https://arxiv.org/abs/2504.11713
- Gentlest ascent dynamics: https://arxiv.org/abs/1011.0042 (equation 18)
- Dimer method: https://pubs.aip.org/aip/jcp/article-abstract/111/15/7010/475160/A-dimer-method-for-finding-saddle-points-on-high

---

## Finding transition states in a 2D toy model

Here is a simple 2D example to demonstrate the idea.
The surface is a four wells potential centered at (0.0, 0.0).
Between the four wells there are four transition states. Note that the "center" at 0.0,0.0 is a maximum, not an index-1 saddle point.

Training takes about 30 minutes on a RTX 3060.

You can see the run here: https://api.wandb.ai/links/andreas-burger/eus8uepw

---

### Code

```bash
uv pip install torch torch_geometric torchmetrics omegaconf wandb tqdm matplotlib seaborn scikit-learn ipynb ipykernel hdbscan pandas
```

---

### Run the code

See the notebook `sgad2d.ipynb` for the full implementation and training loop.

---

## Making this work for more complex systems

There are a few challenges we haven't addressed yet:
1. We want the location of the mode peaks, not to sample around the region. So we might need to cluster the samples, or reduce the temperature at inference time, or use other techniques
2. In more dimensions (more atoms) the number of transition states explodes combinatorically. We are mainly interested in low-energy transition states
3. Depending on the context, we want transition states between conformers (same bond structure) or transitions of chemical reactions (changing bond structure). We will have to use some sort of conditioning/guidance or bond regularisation potential (https://arxiv.org/abs/2504.11713)
4. We are mainly interested in finding a diverse set of modes. We could encourage diverse sampling via conditioning (https://openreview.net/forum?id=U87XyMPrZp) or include off-policy exploration when populating the replay buffer

Please let me know your thoughts!

andreas.burger (at) mail.utoronto.ca
