![License](https://img.shields.io/badge/license-MPL--2.0-blue)
![CI](https://github.com/eriksen-lab/pyscf_opentrustregion/actions/workflows/main.yml/badge.svg)

# PySCF interface to OpenTrustRegion

This package is a [PySCF](https://github.com/pyscf/pyscf) extension that replaces
PySCF's own second-order orbital optimizers with
[OpenTrustRegion](https://github.com/eriksen-lab/opentrustregion), a second-order
trust region optimizer. Hartree–Fock, Kohn–Sham DFT, orbital localization and
state-specific CASSCF are supported.

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

The interface needs `pyscf` and `pyopentrustregion`, the Python interface to the
OpenTrustRegion library. It is a PySCF extension and is not itself installed: PySCF
picks it up from the `PYSCF_EXT_PATH` environment variable.

### 1. Install OpenTrustRegion

OpenTrustRegion is a Fortran library that this interface reaches through `ctypes`.
Building it needs a Fortran compiler, CMake and BLAS/LAPACK.

```sh
git clone https://github.com/eriksen-lab/opentrustregion
cd opentrustregion
pip install .
```

Confirm the installation with

```sh
python3 -m pyopentrustregion.testsuite
```

A default build produces the solver as a static library and only the testsuite library
as a shared one, which is what `ctypes` then loads. If you build OpenTrustRegion with
its testsuite disabled, add `CMAKE_FLAGS='-DBUILD_SHARED_LIBS=ON'` so that a loadable
shared library is produced.

None of the optional OpenTrustRegion extensions are needed by this interface, which
only uses the core solver and stability check.

### 2. Point PySCF at the interface

PySCF discovers extensions through the `PYSCF_EXT_PATH` environment variable, so no
install step is needed:

```sh
git clone https://github.com/eriksen-lab/pyscf_opentrustregion
export PYSCF_EXT_PATH=$PWD/pyscf_opentrustregion:$PYSCF_EXT_PATH
```

See the PySCF documentation on
[extension modules](http://pyscf.org/pyscf/install.html#extension-modules) for details.

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

RHF, ROHF and UHF are supported, along with their KS counterparts; the appropriate
class is selected from the object that is passed in.

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

Further examples are in the [`examples`](examples) directory.

## Settings

Solver settings are set as attributes on the converted object, using the names
documented in the
[OpenTrustRegion README](https://github.com/eriksen-lab/opentrustregion#usage).
Anything left unset keeps the solver's own default.

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
| `n_random_trial_vectors` | `stability_n_random_trial_vectors` |
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
it is not. It works on an object whose `kernel()` has already run, since that is what
sets up the parameter space it examines:

```python
mf.kernel()
stable, direction = mf.stability_check()
```

Note that the solver escapes saddle points on its own, so a converged result is
normally already a minimum.

## Diagnostics

Each object counts the work the solver asked it for, over its whole lifetime:

```python
mf.n_update_orbs     # orbital updates, i.e. Fock builds or integral transformations
mf.n_hess_x          # Hessian linear transformations
```

## Testing

With `PYSCF_EXT_PATH` set as above, check that PySCF picks the interface up and that
OpenTrustRegion loads:

```sh
python3 -c "from pyscf.opentrustregion import mf_to_otr"
```

A `ModuleNotFoundError` means `PYSCF_EXT_PATH` is not set, and a `FileNotFoundError`
means OpenTrustRegion was installed without a loadable shared library. The scripts in
[`examples`](examples) are a quick end-to-end check.

Then run the test suite:

```sh
python3 -m unittest pyscf.opentrustregion.test.test_opentrustregion_interface
```

It compares converged energies against the PySCF solvers this interface replaces and
checks the settings translation, the cached orbital updates, the derivatives and the
stability check. The solver itself is covered by the OpenTrustRegion testsuite.
