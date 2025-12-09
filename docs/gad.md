# Gentlest Ascent Dynamics (GAD)

The gentlest ascent dynamics (GAD) is a vector field for finding index-1 saddle points of a smooth potential V(x).

$$
\dot x = -\nabla V(x)+2\langle \nabla V, v_1(x)\rangle v_1(x).
$$

Where v_1(x) is the eigenvector of the Hessian associated with the smallest eigenvalue 
$$
\nabla^2 V v_i = \lambda_i v_i
$$
$$
\lambda_1 < \lambda_2 < ... < \lambda_{3N}
$$

The stationary points of the GAD vector field are index-1 saddle points, a.k.a. transition states.

Adjoint Sampling (AS) and Adjoint Schroedinger Bridge Sampler (ASBS) sample low energy states by using a loss involving the force (negative gradient of the energy).

We will instead sample transition state regions using the GAD vector field instead of the force.