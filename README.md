# High fidelity tomographic 3D printing
## Iterative projection generation implementation

This python implementation provides the optimization and projection generation framework used for volumetric additive manufacturing as described in the following publications:

[1] B. E. Kelly, I. Bhattacharya, H. Heidari, M. Shusteff, C. Spadaccini, H. K. Taylor, "Volumetric additive manufacturing via tomographic reconstruction", *Science*, 363 (6431), 1075-1079, 2019

[2] I. Bhattacharya, J. Toombs, H. K. Taylor, "High fidelity volumetric additive manufacturing", *Additive Manufacturing*, 47, 102299, 2021

[3] I. Bhattacharya, B. Kelly, M. Shusteff, C. Spadaccini, H. K. Taylor, "Computed axial lithography: volumetric 3D printing of arbitrary geometries", SPIE COSI, 2018

In computed axial lithography, a carefully calculated video is projected into a photocurable resin. The video is synchronized to the rotation of the resin so that specific images are projected at each rotated angle. This is modeled as an integral projection of the image through the resin container (in the low attenuation case). A detailed discussion and process model are provided in [1] and [2] above, but briefly, the forward model can be described as follows:

The 3D dose distribution within the photosensitive resin $f(\mathbf{r}, z)$ arising from a set of projections $g(\rho, \theta, z)$ can be expressed using the integral projection operation as:

$$ f(\mathbf{r}, z) = \frac{\alpha N_r}{\Omega} \int_{\theta = 0 to 2\pi} g(\rho = \mathbf{r}.\hat{\theta}, \theta, z) e^{-\alpha \mathbf{r}. \hat{theta}_\perp} d\theta $$