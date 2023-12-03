# High fidelity tomographic 3D printing
## Iterative projection calculation: forward model

This python implementation provides the optimization and projection generation framework used for volumetric additive manufacturing as described in the following publications:

[1] B. E. Kelly, I. Bhattacharya, H. Heidari, M. Shusteff, C. Spadaccini, H. K. Taylor, "Volumetric additive manufacturing via tomographic reconstruction", *Science*, 363 (6431), 1075-1079, 2019

[2] I. Bhattacharya, J. Toombs, H. K. Taylor, "High fidelity volumetric additive manufacturing", *Additive Manufacturing*, 47, 102299, 2021

[3] I. Bhattacharya, B. Kelly, M. Shusteff, C. Spadaccini, H. K. Taylor, "Computed axial lithography: volumetric 3D printing of arbitrary geometries", SPIE COSI, 2018

In computed axial lithography, a carefully calculated video is projected into a steadily rotating photocurable resin. The video is synchronized to the rotation of the resin so that specific images are projected at each rotated angle. This is modeled as an integral projection of the image through the resin container (in the low attenuation case). The accumulation of dose leads to a two step printing process. The first step, which consumes most of the dose is an induction phase where oxygen free radicals are neutralized by polymer free radicals. The second step is gelation where the excess polymer free radicals cross-link and solidify into a desired geometry. A detailed discussion and process model are provided in [1] and [2] above, but briefly, the forward model can be described as follows:

The 3D dose distribution within the photosensitive resin $f(\mathbf{r}, z)$ (Joules/cm3) arising from a set of projections $g(\rho, \theta, z)$ (Watt/cm2) can be expressed using the integral projection operation as:

$$ f(\mathbf{r}, z) = \frac{\alpha N_r}{\Omega} \int_{\theta = 0} ^{2\pi} g(\rho = \mathbf{r}.\hat{\theta}, \theta, z) e^{-\alpha \mathbf{r}. \hat{\theta}_\perp} d\theta $$

where $N_r$ denotes an integer number of rotations, $\alpha$ is the attenuation per unit length according to the Beer-Lambert law and $\Omega$ is the angular velocity of rotation of the resin container. Note that the expression excluding the pre-factor is the adjoint of the exponential radon tranform.

We are interested in achieving solidification in a region $R_1$ and no solidification in a subset of its compliment with respect to the region volume (which we denote as $R_2$). The dose requirement may be expressed in a simplified form as:

$$ R_1: f(\mathbf{r}, z) \ge d_h, R_2: f(\mathbf{r}, z) \le d_l $$ 

where $d_h > d_l$. Our goal is to calculate $g(\rho, \theta, z)$ that lead to $f(\mathbf{r}, z)$ that best satisfy the above conditions. 

## Optimization framework and software implementation

![CAL setup](githubRepo_schematic.png)


