# GRaTeR-JAX
**GPU-accelerated modeling of scattered-light disks**

<p align="center">
  <img src="https://github.com/user-attachments/assets/c10f45e8-5449-4891-b6a7-33954cf6d954" width="220">
</p>

<<<<<<< HEAD
**GRaTeR-JAX** is a Python library for forward modeling of scattered-light images of circumstellar debris disks.  
It builds upon the *Generalized Radial Transporter (GRaTeR)* framework [(Augereau+ 1999)](https://arxiv.org/abs/astro-ph/9906429) using the **JAX** ecosystem to achieve fast, differentiable, GPU-accelerated computation for disk simulations, parameter inference, and image optimization.

Developed by the **UCSB Exoplanet Polarimetry Lab**, GRaTeR-JAX provides the foundation for analyzing debris disks using modern differentiable programming techniques.
=======
**GRaTeR-JAX** is a machine learning JAX-based implementation of the **Generalized Radial Transporter (GRaTeR)** framework [(Augereau+ 1999)](https://ui.adsabs.harvard.edu/abs/1999A%26A...348..557A/abstract), designed for modeling scattered light observations of debris disks. This repository provides tools for forward modeling, optimization, and parameter estimation of debris disk images using the **JAX** ecosystem to achieve fast, differentiable, GPU-accelerated disk simulations. 

Note that **GRaTeR** is ***not*** a ray-tracing code for circumstellar disk modeling, but instead an analytic model based on a dust distribution and scattering properties such as the scattering phase function (SPF), specifically intended for optically thin disks like debris disks; for modeling protoplanetary disk or general circumstellar disk models using ray tracing, other packages exist such as [MCFOST](https://mcfost.readthedocs.io/en/latest/overview.html) and [RADMC-3D](https://github.com/dullemond/radmc3d-2.0).

Developed by the [**UCSB Exoplanet Polarimetry Lab**](https://ucsb-exoplanet-polarimetry-lab.github.io/), **GRaTeR-JAX** provides a foundation for analyzing debris disks using modern differentiable programming techniques.
>>>>>>> 31f7623 (revised readme and index for read the docs)

---

## Features
* **GPU/TPU acceleration** through JAX
* **Differentiable physical modeling** for analytical gradient-based optimization
<<<<<<< HEAD
* **Spline SPF Modeling** for more dynamic and accurate scattering phase function (SPF) fitting
* **PSF convolution** for both static and dynamic PSFs, such as those from _JWST_
=======
* **Spline SPF Modeling** for more flexible scattering phase function (SPF) fitting
* **PSF convolution** for both static and dynamic PSFs
>>>>>>> 31f7623 (revised readme and index for read the docs)
* **Modular design** for more flexibility and control
* **Higher Level API** for more intuitive and user-friendly disk fitting
* **Image Processing Utilities** for ingesting data, making PSF models, and estimating noise

---

## Bugs and Feature Requests
Please use the [GitHub Issue Tracker](https://github.com/UCSB-Exoplanet-Polarimetry-Lab/GRaTeR-JAX/issues) to submit bug reports, documentation issues, or feature requests.  
Contributions from the community are always welcome.

---

## Attribution
The development of GRaTeR-JAX is led by **Mihir Kondapalli**, **Briley Lewis**, and **Max Millar-Blanchaer** 
with contributions from members of the **UCSB Exoplanet Polarimetry Lab** and the wider astrophysics software community.

If you build upon this package, please cite both GRaTeR-JAX (Kondapalli et al. in prep JOSS, Lewis et al. in prep AJ) and the original GRaTeR framework [(Augereau+ 1999)](https://ui.adsabs.harvard.edu/abs/1999A%26A...348..557A/abstract).

---

## Acknowledgments
This work was developed by the [**UCSB Exoplanet Polarimetry Lab**](https://ucsb-exoplanet-polarimetry-lab.github.io/).

---

## Contents

```{toctree}
:maxdepth: 2

guide/index
api/index
tutorials/index
