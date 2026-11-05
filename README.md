![License](https://img.shields.io/badge/license-MPL--2.0-blue)
![CI](https://github.com/eriksen-lab/pyscf_opentrustregion/actions/workflows/main.yml/badge.svg)

# PySCF interface to OpenTrustRegion

This package is a [PySCF](https://github.com/pyscf/pyscf) extension that replaces
PySCF's own second-order orbital optimizers with
[OpenTrustRegion](https://github.com/eriksen-lab/opentrustregion), a second-order
trust region optimizer. Hartree–Fock, Kohn–Sham DFT, orbital localization and
CASSCF are supported.

Objects produced by this interface remain fully usable as the PySCF objects they were
built from, so they can be passed to any PySCF routine that expects the original class.

The methodology is documented in the following paper, which should be cited in any
work using OpenTrustRegion:

- Greiner, J.; Høyvik, I.-M.; Lehtola, S.; Eriksen, J. J.
  A Reusable Library for Second-Order Orbital Optimization Using the Trust Region
  Method. *Journal of Chemical Theory and Computation* **2026**, *22*(2), 881–895.
  DOI: [10.1021/acs.jctc.5c01576](https://doi.org/10.1021/acs.jctc.5c01576).
  arXiv: [2509.13931](https://arxiv.org/abs/2509.13931).

## Installation

The interface is a PySCF extension available from PyPI, which installs it together
with PySCF 2.14 or later and `pyopentrustregion`, the Python interface to the
OpenTrustRegion library:

```sh
pip install pyscf-opentrustregion
```

Confirm the OpenTrustRegion installation with

```sh
python3 -m pyopentrustregion.verify_install
```

The OpenTrustRegion wheels are built with 32-bit integers. To build it from source
instead, for example with 64-bit integers or from its development branch, follow the
[OpenTrustRegion installation instructions](https://github.com/eriksen-lab/opentrustregion#installation)
before installing the interface, which then keeps that build.

### Development version

To use or work on the development version, install the interface from a clone in
editable mode,

```sh
git clone https://github.com/eriksen-lab/pyscf_opentrustregion
pip install -e pyscf_opentrustregion
```

or, instead of installing it, point PySCF at the clone through the `PYSCF_EXT_PATH`
environment variable:

```sh
export PYSCF_EXT_PATH=$PWD/pyscf_opentrustregion:$PYSCF_EXT_PATH
```

See the PySCF documentation on [extensions](https://pyscf.org/user/extensions.html)
for details.

## Usage

### Hartree–Fock and DFT

`mf_to_otr` converts a PySCF mean-field object into its OpenTrustRegion counterpart:

```python
from pyscf import gto, scf
from pyscf.opentrustregion import mf_to_otr

mol = gto.M(atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", basis="cc-pVDZ")

mf = mf_to_otr(scf.RHF(mol))
mf.kernel()

stable, direction = mf.stability_check()
```

RHF, ROHF, UHF and GHF are supported, along with their KS counterparts and spinor X2C
(`scf.X2C(mol)`), Kramers-restricted and unrestricted; the appropriate class is selected
from the object that is passed in. Complex orbitals, as from a complex initial guess or with spin–orbit
coupling (`scf.GHF(mol).x2c()`), are optimized through the real and imaginary parts of
their rotations. Objects with a solvent model, such as `scf.RHF(mol).ddCOSMO()`, and
periodic calculations at the Γ point are supported as well. The converted object also
works in PySCF's scanners, which geometry optimizations and scans run, e.g. with
geomeTRIC:

```python
from pyscf.geomopt.geometric_solver import optimize

mol_eq = optimize(mf_to_otr(scf.RHF(mol)))
```

**Set the functional before the conversion.** The converted object evaluates the energy
and the gradient through the PySCF object it was converted from, but the Hessian
itself, as PySCF's own `newton()` does. A Kohn–Sham option defining the energy (`xc`,
`nlc`, `disp`, `grids`, `nlcgrids`, `small_rho_cutoff`) that is assigned after the
conversion therefore reaches at most the Hessian: the solver optimizes the energy of the
original functional, and the stability check can report instabilities that this energy
does not have. The converted object warns about such a difference when the solver or the
stability check starts.

```python
mf = scf.UKS(mol)
mf.xc = "B3LYP"
mf = mf_to_otr(mf)
```

A grid changed in place, as by `mf.grids.level = 4`, reaches both objects, since they
share it. To approximate the Hessian on purpose with a coarser grid, PySCF releases
after 2.14 provide `mf.set_second_grids()`, whose grid only the Hessian uses.

### Orbital localization

The localizers are used exactly like the PySCF classes they derive from:

```python
from pyscf.opentrustregion import BoysOTR, PipekMezeyOTR, EdmistonRuedenbergOTR

loc = BoysOTR(mol, mf.mo_coeff[:, : mol.nelec[0]])
mo_loc = loc.kernel()
```

### CASSCF

`casscf_to_otr` converts a PySCF CASSCF object:

```python
from pyscf import mcscf
from pyscf.opentrustregion import casscf_to_otr

mc = casscf_to_otr(mcscf.CASSCF(mf, 6, 6))
mc.kernel()
```

State-averaged CASSCF, also over several CI solvers (`mcscf.state_average_` and
`mcscf.state_average_mix_`), is converted the same way.

Further examples are in the [`examples`](https://github.com/eriksen-lab/pyscf_opentrustregion/tree/main/examples) directory.

## Settings

Solver settings are set as attributes on the converted object, using the names
documented in the
[OpenTrustRegion README](https://github.com/eriksen-lab/opentrustregion#usage).
Anything left unset keeps the solver's own default. The solver writes its output to the
PySCF log of the object, which `verbose` and the output file control as for PySCF's own
output.

```python
mf = mf_to_otr(scf.RHF(mol))
mf.conv_tol = 1e-7
mf.n_macro = 100
mf.line_search = True
mf.kernel()
```

Four details follow from PySCF and OpenTrustRegion sharing some attribute names.

**Set `conv_tol` after the conversion.** The name means different things on the two
sides. PySCF converges on the change in the energy, or in the localization cost
function, and keeps a separate `conv_tol_grad` for the gradient; OpenTrustRegion
converges on the RMS gradient itself. The defaults follow from that: `1e-9` for PySCF's
mean-field classes, `1e-7` for CASSCF and `1e-6` for the localizers, against `1e-5` for
the solver. Inheriting the PySCF value would therefore ask for an RMS gradient up to
four orders of magnitude tighter than anyone intended, so it is ignored and only a
value assigned after the conversion reaches the solver:

```python
mf = scf.RHF(mol)
mf.conv_tol = 1e-7           # ignored by the solver
mf = mf_to_otr(mf)
mf.conv_tol = 1e-7           # used by the solver, as an RMS gradient threshold
mf.kernel()
```

**Stability check settings take a `stability_` prefix.** The solver and the stability
check share several setting names, so the stability check reads the ones below from a
separate attribute; every other stability setting is read from its plain name:

| Stability check setting | Attribute to set |
|---|---|
| `conv_tol` | `stability_conv_tol` |
| `conv_check` | `stability_conv_check` |
| `n_random_trial_vectors` | `stability_n_random_trial_vectors` |
| `n_extra_trial_vectors` | `stability_n_extra_trial_vectors` |
| `get_extra_trial_vectors` | `stability_get_extra_trial_vectors` |
| `jacobi_davidson_start` | `stability_jacobi_davidson_start` |

**`conv_check` is a function here.** PySCF uses `conv_check` as a boolean flag whereas
OpenTrustRegion expects a convergence check function, so a boolean of that name is
ignored rather than passed on.

**`stability` is a setting here, not a method.** PySCF's mean-field and localizer
classes provide a `stability()` method, whereas OpenTrustRegion uses `stability` as a
boolean deciding whether the solver runs a stability check upon convergence. The
setting takes precedence, so the PySCF method is no longer callable; the equivalent is
the `stability_check()` method described below.

## Stability check

Every object exposes a `stability_check()` method, which examines the current point and
returns whether it is a minimum together with a direction out of the saddle point when
it is not. It only classifies stationary points and warns at any other point. The
mean-field objects examine the orbitals and occupations they hold, also without a
preceding `kernel()`, whereas the localizers and CASSCF need their `kernel()` to have
run, since it sets up the parameter space the check examines:

```python
mf.kernel()
stable, direction = mf.stability_check()
```

For complex orbitals, the direction holds the real parts of the rotations followed by
their imaginary parts. The ROHF Hessian of PySCF 2.14 and earlier is approximate also
at a stationary point, so that the check of an ROHF point can misjudge curvatures close
to zero there. Note that the solver escapes saddle points on its own, so a converged
result is normally already a minimum.

## Diagnostics

Each object counts the work the solver asked it for, over its whole lifetime:

```python
mf.n_update_orbs     # orbital updates, i.e. Fock builds or integral transformations
mf.n_hess_x          # Hessian linear transformations
```

## Testing

Check that PySCF picks the interface up and that OpenTrustRegion loads:

```sh
python3 -c "from pyscf.opentrustregion import mf_to_otr"
```

A `ModuleNotFoundError` means that the interface is neither installed nor on
`PYSCF_EXT_PATH`, and a `FileNotFoundError` means OpenTrustRegion was installed without
a loadable shared library. The scripts in
[`examples`](https://github.com/eriksen-lab/pyscf_opentrustregion/tree/main/examples) are a quick end-to-end check.

Then run the test suite:

```sh
python3 -m unittest pyscf.opentrustregion.test.test_opentrustregion_interface
```

It compares converged energies against the PySCF solvers this interface replaces and
checks the settings translation, the cached orbital updates, the derivatives and the
stability check. The solver itself is covered by the OpenTrustRegion testsuite.
