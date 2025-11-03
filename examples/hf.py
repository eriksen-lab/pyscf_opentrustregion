from pyscf import gto, scf, dft
from pyscf.opentrustregion import mf_to_otr

# define molecule
mol = gto.Mole()
mol.build(
    atom="""
        O    0.000  0.000  0.000
        H    0.000 -0.757  0.587
        H    0.000  0.757  0.587
    """,
    basis="cc-pVDZ",
    symmetry=True,
)

### run RHF

# use hf_to_otr function to convert to RHFOTR object
hf = mf_to_otr(scf.RHF(mol))

# call kernel
hf.kernel()

# call stability check
stable, direction = hf.stability_check()

### run RKS

# use hf_to_otr function to convert to RHFOTR object
hf = mf_to_otr(dft.RKS(mol, xc="b3lyp"))

# call kernel
hf.kernel()

# call stability check
stable, direction = hf.stability_check()

# set spin
mol.spin = 2

### run ROHF

# use hf_to_otr function to convert to ROHFOTR object
hf = mf_to_otr(scf.ROHF(mol))

# call kernel
hf.kernel()

# call stability check
stable, direction = hf.stability_check()

### run ROKS

# use hf_to_otr function to convert to ROHFOTR object
hf = mf_to_otr(dft.ROKS(mol, xc="b3lyp"))

# call kernel
hf.kernel()

# call stability check
stable, direction = hf.stability_check()

### run UHF

# use hf_to_otr function to convert to ROHFOTR object
hf = mf_to_otr(scf.UHF(mol))

# call kernel
hf.kernel()

# call stability check
stable, direction = hf.stability_check()

### run UKS

# use hf_to_otr function to convert to ROHFOTR object
hf = mf_to_otr(dft.UKS(mol, xc="b3lyp"))

# call kernel
hf.kernel()

# call stability check
stable, direction = hf.stability_check()
