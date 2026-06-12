# AIES Seismic Impedance Inversion

This repository contains a MATLAB implementation of seismic acoustic impedance inversion using an Affine-Invariant Ensemble Sampler (AIES). The main example script is [`AIES-MCMC-Poststack-Inversion.m`](./AIES-MCMC-Poststack-Inversion.m), which demonstrates Bayesian impedance inversion on a synthetic single-trace benchmark.

## Overview

`AIES-MCMC-Poststack-Inversion.m` performs probabilistic impedance inversion from a seismic trace. It builds a low-frequency prior impedance model, generates or reads seismic observations, and samples the posterior impedance distribution using an optimized batch version of the AIES stretch move.

The script estimates:

- Posterior impedance realizations
- Posterior mean impedance
- MAP-like impedance estimate from kernel density estimation
- 5% and 95% posterior uncertainty bounds
- Acceptance-rate history
- Mean log-posterior convergence curve
- RMSE and correlation coefficient against the theoretical model

## Main Workflow

The inversion workflow is:

1. Load benchmark data from `data3.mat`.
2. Construct the true acoustic impedance:

   ```matlab
   true_imp = Vp' .* Rho';
   ```

3. Generate the reflection coefficient series.
4. Convolve the reflection series with a Ricker wavelet.
5. Add Gaussian noise to create observed seismic data.
6. Build a low-frequency prior impedance model by Butterworth filtering.
7. Construct a Gaussian prior covariance using temporal correlation.
8. Run AIES posterior sampling using vectorized stretch moves.
9. Plot posterior realizations, posterior mean, MAP estimate, uncertainty bounds, convergence history, and acceptance rate.

## Requirements

Recommended environment:

- MATLAB R2020b or newer
- Signal Processing Toolbox
  - `butter`
  - `filtfilt`
- Statistics and Machine Learning Toolbox
  - `mvnrnd`
  - `ksdensity`
  - `quantile`
- SeReM or another implementation of `RickerWavelet`

The current script expects:

- `data3.mat`
- `defaultAxes.m` optional, used only for figure formatting
- `RickerWavelet.m`, available in this repository under `SeReM/Inversion/Seismic/`

## Quick Start

Clone or download the repository, open MATLAB, and run:

```matlab
cd('path/to/your/repository');
addpath(genpath(pwd));
untitled_faster
```

The script will run the inversion and generate several figures.

## Important Parameters

The main parameters are defined near the top and middle of [`untitled_faster.m`](./untitled_faster.m):

```matlab
nd = 99;                 % Number of impedance samples
nfilt = 3;               % Butterworth filter order
dt = 0.001;              % Sampling interval, seconds
freq = 45;               % Ricker wavelet dominant frequency, Hz
ntw = 64;                % Wavelet length
nwalkers = 200;          % Number of AIES walkers
niter = 10000;           % Number of MCMC iterations
a = 2;                   % AIES stretch-move scale parameter
burnin = round(0.1*niter);
```

Noise level is controlled by:

```matlab
sigma = sqrt(0.1 * var(d));
d_obs = d + sigma * randn(size(d));
```

The prior model is controlled by the low-pass filter:

```matlab
cutofffr = 0.04;
[b, a_filter] = butter(nfilt, cutofffr);
m0 = filtfilt(b, a_filter, true_imp);
```

The prior covariance is controlled by:

```matlab
corrlength = 3 * dt;
sigma0 = max(var(true_imp - m0, 1), 1.0e-6);
sigmaprior = sigma0 * sigmatime;
```

## Outputs

The script produces:

- `figure(1)`: posterior impedance realizations, true impedance, prior model, posterior mean, MAP estimate, and uncertainty bounds
- `figure(2)`: negative mean log-posterior convergence curve
- `figure(3)`: AIES acceptance-rate history
- `figure(4)`: reference model curves for velocity, density, and impedance

It also computes useful scalar metrics:

```matlab
Coverage_Ratio
rmseImp_mean
rrmseImp_mean
rmseImp_map
rrmseImp_map
ccImp_mean
ccImp_map
```

## Algorithm Notes

The posterior probability is defined as:

```matlab
log posterior = log likelihood + log prior
```

The likelihood compares observed and synthetic seismic data:

```matlab
residual = d_syn - d_obs;
ll = -0.5 * misfit / sigma^2;
```

The prior constrains impedance samples around the low-frequency model:

```matlab
dm = models - m0;
prior = -0.5 * sum((dm * inv_sigmaprior) .* dm, 2);
```

The sampler uses the affine-invariant stretch move, which is efficient for correlated model parameters and avoids hand-tuning a proposal covariance.

## Repository Notes

`untitled_faster.m` is currently a research script rather than a packaged MATLAB function. For publication or wider reuse, consider renaming it to a descriptive name such as:

```text
AIES_impedance_inversion_demo.m
```

You may also want to separate the workflow into reusable functions:

- data loading
- forward seismic modeling
- prior construction
- AIES sampling
- posterior visualization
- metric calculation

## Citation

If you use this code in academic work, please cite the affine-invariant ensemble sampler method:

Goodman, J., & Weare, J. (2010). Ensemble samplers with affine invariance. *Communications in Applied Mathematics and Computational Science*, 5(1), 65-80.

If this repository accompanies a paper, thesis, or report, add your project citation here.

## License

Please add a license file before public release. Common choices are:

- MIT License for permissive open-source release
- GPL-3.0 if derivative work should remain open source
- Apache-2.0 for permissive release with explicit patent language

