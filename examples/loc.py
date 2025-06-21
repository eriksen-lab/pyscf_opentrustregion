from pyscf import gto, scf
from pyscf.opentrustregion import (
    PipekMezeyOTR,
    BoysOTR,
    EdmistonRuedenbergOTR,
)

# define molecule
mol = gto.Mole()
mol.build(
    atom="""
        O    0.000  0.000  0.000
        H    0.000 -0.757  0.587
        H    0.000  0.757  0.587
    """,
    basis="cc-pVDZ",
    symmetry=False,
)

# perform HF calculation
hf = scf.RHF(mol).run()

# orbitals
orbs = [hf.mo_coeff[:, : min(mol.nelec)], hf.mo_coeff[:, max(mol.nelec) :]]

### Pipek-Mezey localization

# loop over occupied and virtual subspaces
for mo_coeff in orbs:
    loc = PipekMezeyOTR(mol, mo_coeff)
    loc.line_search = True

    loc.kernel()

### Foster-Boys localization

# loop over occupied and virtual subspaces
for mo_coeff in orbs:
    loc = BoysOTR(mol, mo_coeff)
    loc.line_search = True

    loc.kernel()

### Edmiston-Ruedenberg

# loop over occupied and virtual subspaces
for mo_coeff in orbs:
    loc = EdmistonRuedenbergOTR(mol, mo_coeff)
    loc.line_search = True

    loc.kernel()
