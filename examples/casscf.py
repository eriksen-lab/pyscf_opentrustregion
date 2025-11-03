from pyscf import gto, scf, mcscf
from pyscf.opentrustregion import casscf_to_otr

# define molecule
mol = gto.M(
    atom = """
        N  0.000000  0.000000  0.7
        N  0.000000  0.000000 -0.7
    """,
    basis = "cc-pVDZ",
    symmetry = False,
)

# perform HF calculation
mf = scf.RHF(mol).run()

# run CASSCF calculation
mc = casscf_to_otr(mcscf.CASSCF(mf, 6, 6))
mc.kernel()

# call stability check
stable, direction = mc.stability_check()
