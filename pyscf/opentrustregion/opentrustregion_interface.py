# Copyright (C) 2025- Jonas Greiner
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import annotations

import numpy as np
from functools import reduce
from pyscf import gto, scf, lo, lib
from pyscf.soscf import ciah, newton_ah
from pyscf.mcscf import casci, newton_casscf, addons
from pyopentrustregion import solver, stability_check
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple, Callable, Optional, Union


class OTR:
    _keys = {
        "precond",
        "conv_check",
        "stability",
        "line_search",
        "davidson",
        "jacobi_davidson",
        "prefer_jacobi_davidson",
        "conv_tol",
        "n_random_trial_vectors",
        "start_trust_radius",
        "n_macro",
        "n_micro",
        "global_red_factor",
        "local_red_factor",
        "seed",
        "verbose",
        "logger",
    }


class BoysOTR(OTR, lo.Boys):

    norb: int
    mo_coeff: np.ndarray

    # unpack matrix
    def unpack(self, kappa: np.ndarray) -> np.ndarray:
        matrix = np.zeros(2 * (self.norb,), dtype=np.float64)
        idx = np.tril_indices(self.norb, -1)
        matrix[idx] = kappa
        return matrix - matrix.conj().T

    # cost function
    def func(self, kappa: np.ndarray) -> float:
        u = ciah.expmat(self.unpack(kappa))
        return self.cost_function(u)

    # cost function, gradient, Hessian diagonal and Hessian linear transformation
    # function
    def update_orbs(
        self, kappa: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        u = ciah.expmat(self.unpack(kappa))
        func = self.cost_function(u)
        grad, hess_x, hdiag = self.gen_g_hop(u)
        self.mo_coeff = self.mo_coeff @ u
        
        return func, 2 * grad, 2 * hdiag, lambda x: 2 * hess_x(x)

    # kernel function
    def kernel(self, mo_coeff: Optional[np.ndarray] = None) -> np.ndarray:

        # set MO coefficients if provided
        if mo_coeff is not None:
            self.mo_coeff = np.asarray(mo_coeff, order="C")

        # emnsure MO coefficients are provided and orbitals can be optimized
        assert self.mo_coeff is not None
        if self.mo_coeff.shape[1] <= 1:
            return self.mo_coeff

        # number of orbitals
        self.norb = self.mo_coeff.shape[1]

        # number of parameters
        self.n_param = (self.norb - 1) * self.norb // 2

        # get initial guess
        if mo_coeff is None:
            if getattr(self, "mol", None) and self.mol.natm == 0:
                # For customized Hamiltonian
                u0 = self.get_init_guess("random")
            else:
                u0 = self.get_init_guess(self.init_guess)
        else:
            u0 = self.get_init_guess(None)
        self.mo_coeff = self.mo_coeff @ u0

        # call solver
        solver(
            self.func,
            self.update_orbs,
            self.n_param,
            getattr(self, "precond", None),
            getattr(self, "conv_check", None),
            getattr(self, "stability", None),
            getattr(self, "line_search", None),
            getattr(self, "davidson", None),
            getattr(self, "jacobi_davidson", None),
            getattr(self, "prefer_jacobi_davidson", None),
            getattr(self, "conv_tol", None),
            getattr(self, "n_random_trial_vectors", None),
            getattr(self, "start_trust_radius", None),
            getattr(self, "n_macro", None),
            getattr(self, "n_micro", None),
            getattr(self, "global_red_factor", None),
            getattr(self, "local_red_factor", None),
            getattr(self, "seed", None),
            getattr(self, "verbose", None),
            getattr(self, "logger", None),
        )

        return self.mo_coeff

    # stability check function
    def stability(self) -> Tuple[bool, np.ndarray]:
        _, _, h_diag, hess_x = self.update_orbs(
            np.zeros(self.n_param, dtype=np.float64)
        )
        return stability_check(
            h_diag,
            hess_x,
            self.n_param,
            getattr(self, "precond", None),
            getattr(self, "jacobi_davidson", None),
            getattr(self, "conv_tol", None),
            getattr(self, "n_random_trial_vectors", None),
            getattr(self, "n_iter", None),
            getattr(self, "verbose", None),
            getattr(self, "logger", None),
        )


class PipekMezeyOTR(lo.PipekMezey, BoysOTR):

    # cost function
    def func(self, kappa: np.ndarray) -> float:
        u = ciah.expmat(self.unpack(kappa))
        return -self.cost_function(u)

    # cost function, gradient, Hessian diagonal and Hessian linear transformation
    # function
    def update_orbs(
        self, kappa: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        u = ciah.expmat(self.unpack(kappa))
        func = self.cost_function(u)
        grad, hess_x, hdiag = self.gen_g_hop(u)
        self.mo_coeff = self.mo_coeff @ u
        return -func, 2 * grad, 2 * hdiag, lambda x: 2 * hess_x(x)


class EdmistonRuedenbergOTR(lo.EdmistonRuedenberg, PipekMezeyOTR):
    pass


class FourthMomentOTR(lo.FourthMoment, BoysOTR):
    pass


class SecondOrderOTR(OTR, newton_ah._CIAH_SOSCF):

    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    dm: np.ndarray
    vhf: np.ndarray

    def __init__(self, mf: scf.SCF):
        self.__dict__.update(mf.__dict__)
        self._scf = mf

    # energy function
    def func(self, kappa: np.ndarray) -> float:
        u = ciah.expmat(self.unpack(kappa))
        rot_mo_coeff = self.rotate_mo(self.mo_coeff, u)
        dm = self.make_rdm1(rot_mo_coeff, self.mo_occ)
        vhf = self._scf.get_veff(self._scf.mol, dm)
        return self._scf.energy_tot(dm, self.h1e, vhf)

    # energy, gradient, Hessian diagonal and Hessian linear transformation function
    def update_orbs(
        self, kappa: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        u = ciah.expmat(self.unpack(kappa))
        self.mo_coeff = self.rotate_mo(self.mo_coeff, u)
        dm = self.make_rdm1(self.mo_coeff, self.mo_occ)
        vhf = self._scf.get_veff(self._scf.mol, dm)
        self.dm, self.vhf = dm, vhf
        fock = self.get_fock(self.h1e, self.s1e, vhf, dm)
        grad, hess_x, h_diag = self.gen_g_hop(self.mo_coeff, self.mo_occ, fock)

        def hess_x_symm(x):
            x_full = np.zeros_like(self.mask_symm, dtype=np.float64)
            x_full[self.mask_symm] = x
            return 2 * hess_x(x_full)[self.mask_symm]

        return (
            self._scf.energy_tot(dm, self.h1e, vhf),
            2 * grad[self.mask_symm],
            2 * h_diag[self.mask_symm],
            hess_x_symm,
        )

    # kernel function
    def kernel(
        self,
        mo_coeff: Optional[np.ndarray] = None,
        mo_occ: Optional[np.ndarray] = None,
        dm: Optional[np.ndarray] = None,
    ) -> float:
        if dm is not None:
            if isinstance(dm, str):
                lib.logger.debug(
                    self, 
                    f"OpenTrustRegion solver reads density matrix from chkfile {dm}",
                )
                dm = self.from_chk(dm)

        elif mo_coeff is not None and mo_occ is None:
            lib.logger.warn(
                self, 
                "Newton solver expects mo_coeff with mo_occ as initial guess but "
                "mo_occ is not found in the arguments.",
            )
            lib.logger.warn(self, "The given argument is treated as density matrix.")
            dm = mo_coeff
            mo_coeff = mo_occ = None

        else:
            if mo_coeff is None:
                mo_coeff = self.mo_coeff
            if mo_occ is None:
                mo_occ = self.mo_occ

        self.build(self.mol)

        mol = self._scf.mol

        # call self._scf.get_hcore, self._scf.get_ovlp because they might be overloaded
        self.h1e = self._scf.get_hcore(mol)
        self.s1e = self._scf.get_ovlp(mol)

        # get initial guess
        if mo_coeff is not None and mo_occ is not None:
            dm = self.make_rdm1(mo_coeff, mo_occ)
            vhf = self._scf.get_veff(mol, dm)
            self.dm, self.vhf = dm, vhf
            fock = self.get_fock(self.h1e, self.s1e, vhf, dm, level_shift_factor=0)
            mo_energy, mo_tmp = self.eig(fock, self.s1e)
            self.get_occ(mo_energy, mo_tmp)
            mo_tmp = None

        else:
            if dm is None:
                lib.logger.debug(
                    self,
                    "Initial guess density matrix is not given. Generating initial "
                    f"guess from {self.init_guess}",
                )
                dm = self.get_init_guess(self._scf.mol, self.init_guess)
            vhf = self._scf.get_veff(mol, dm)
            self.dm, self.vhf = dm, vhf
            fock = self.get_fock(self.h1e, self.s1e, vhf, dm, level_shift_factor=0)
            mo_energy, mo_coeff = self.eig(fock, self.s1e)
            mo_occ = self.get_occ(mo_energy, mo_coeff)
            dm = self.make_rdm1(mo_coeff, mo_occ)
            vhf = self._scf.get_veff(mol, dm, dm_last=self.dm, vhf_last=self.vhf)
            self.dm, self.vhf = dm, vhf

        self.mo_coeff, self.mo_occ = mo_coeff, mo_occ

        # get indices of all mixed occupation combinations
        self.mask, self.mask_symm = self.get_indices()

        # number of parameters
        self.n_param = np.count_nonzero(self.mask_symm)

        # call solver
        solver(
            self.func,
            self.update_orbs,
            self.n_param,
            getattr(self, "precond", None),
            getattr(self, "conv_check", None) 
            if not isinstance(getattr(self, "conv_check", None), bool) 
            else None,
            getattr(self, "stability", None),
            getattr(self, "line_search", None),
            getattr(self, "davidson", None),
            getattr(self, "jacobi_davidson", None),
            getattr(self, "prefer_jacobi_davidson", None),
            getattr(self, "conv_tol", None),
            getattr(self, "n_random_trial_vectors", None),
            getattr(self, "start_trust_radius", None),
            getattr(self, "n_macro", None),
            getattr(self, "n_micro", None),
            getattr(self, "global_red_factor", None),
            getattr(self, "local_red_factor", None),
            getattr(self, "seed", None),
            getattr(self, "verbose", None),
            getattr(self, "logger", None),
        )

        # get canonical orbitals
        self.converged = True
        self.e_tot = self.func(np.zeros(self.n_param, dtype=np.float64))
        dm = self.make_rdm1(self.mo_coeff, self.mo_occ)
        vhf = self._scf.get_veff(self._scf.mol, dm)
        self.dm, self.vhf = dm, vhf
        fock = self.get_fock(self.h1e, self.s1e, vhf, dm, level_shift_factor=0)
        self.mo_energy, self.mo_coeff = self._scf.canonicalize(
            self.mo_coeff, self.mo_occ, fock
        )

        self._finalize()

        return self.e_tot

    # stability check function
    def stability(self) -> Tuple[bool, np.ndarray]:
        _, _, h_diag, hess_x = self.update_orbs(
            np.zeros(self.n_param, dtype=np.float64)
        )
        return stability_check(
            h_diag,
            hess_x,
            self.n_param,
            getattr(self, "precond", None),
            getattr(self, "jacobi_davidson", None),
            getattr(self, "conv_tol", None),
            getattr(self, "n_random_trial_vectors", None),
            getattr(self, "n_iter", None),
            getattr(self, "verbose", None),
            getattr(self, "logger", None),
        )


class RHFOTR(SecondOrderOTR, newton_ah._SecondOrderRHF):

    # get indices of all mixed occupation combinations
    def get_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        occidxa = self.mo_occ > 0
        occidxb = self.mo_occ == 2
        viridxa = ~occidxa
        viridxb = ~occidxb
        mask = (viridxa[:, None] & occidxa) | (viridxb[:, None] & occidxb)

        mask_symm = mask[mask]
        if self._scf.mol.symmetry:
            orbsym = self.get_orbsym(self.mo_coeff)
            sym_allow = orbsym[:, None] == orbsym
            mask_symm = sym_allow[mask]
            mask[mask] = mask_symm

        return mask, mask_symm

    # unpack matrix
    def unpack(self, kappa):
        matrix = np.zeros(2 * (self.mol.nao,), dtype=np.float64)
        matrix[self.mask] = kappa
        return matrix - matrix.T


class ROHFOTR(SecondOrderOTR, newton_ah._SecondOrderROHF):

    get_indices = RHFOTR.get_indices
    unpack = RHFOTR.unpack


class UHFOTR(SecondOrderOTR, newton_ah._SecondOrderUHF):

    # get indices of all mixed occupation combinations
    def get_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        occidxa = self.mo_occ[0] == 1
        occidxb = self.mo_occ[1] == 1
        viridxa = ~occidxa
        viridxb = ~occidxb
        mask = np.stack((viridxa[:, None] & occidxa, viridxb[:, None] & occidxb))

        mask_symm = mask[mask]
        if self._scf.mol.symmetry:
            orbsyma, orbsymb = self.get_orbsym(self.mo_coeff)
            sym_allowa = orbsyma[:, None] == orbsyma
            sym_allowb = orbsymb[:, None] == orbsymb
            sym_allow = np.stack((sym_allowa, sym_allowb))
            mask_symm = sym_allow[mask]
            mask[mask] = mask_symm

        return mask, mask_symm

    def rotate_mo(self, mo_coeff, u):
        return super().rotate_mo(
            mo_coeff,
            (
                u[: self.mol.nao, : self.mol.nao],
                u[self.mol.nao :, self.mol.nao :],
            ),
        )

    # unpack matrix
    def unpack(self, kappa):
        matrix = np.zeros(2 * (2 * self.mol.nao,), dtype=np.float64)
        matrix[: self.mol.nao, : self.mol.nao][self.mask[0]] = kappa[
            : np.count_nonzero(self.mask[0])
        ]
        matrix[self.mol.nao :, self.mol.nao :][self.mask[1]] = kappa[
            np.count_nonzero(self.mask[0]) :
        ]
        return matrix - matrix.T


def mf_to_otr(mf):

    if isinstance(mf, SecondOrderOTR):
        return mf

    assert isinstance(mf, scf.hf.SCF)

    if mf.istype("ROHF"):
        cls = ROHFOTR
    elif mf.istype("UHF"):
        cls = UHFOTR
    else:
        cls = RHFOTR

    mf = lib.set_class(cls(mf), (cls, mf.__class__))
    if hasattr(mf, "stability") and callable(mf.stability):
        mf.stability = None

    return mf


class CASSCFOTR(newton_casscf.CASSCF):

    def __init__(
        self,
        mf_or_mol: Union[gto.Mole, scf.RHF],
        ncas,
        nelecas,
        ncore=None,
        frozen=None,
    ):
        casci.CASBase.__init__(self, mf_or_mol, ncas, nelecas, ncore)
        self.frozen = frozen

        self.e_tot = None
        self.e_cas = None
        self.ci = None
        self.mo_coeff = self._scf.mo_coeff
        self.mo_energy = self._scf.mo_energy
        self.converged = False
        self._max_stepsize = None

    # energy function
    def func(self, x):
        u = ciah.expmat(self.unpack_uniq_var(x[: self.n_param_orb]))

        rot_mo_coeff = self.rotate_mo(self.mo_coeff, u)
        eris = self.ao2mo(rot_mo_coeff)

        idx_start = self.n_param_orb
        if self.fcisolver.nroots == 1:
            ci = self.ci + x[idx_start:]
            ci /= np.linalg.norm(ci)
        else:
            ci = []
            for c in self.ci:
                idx_stop = idx_start + c.size
                ci.append(c + x[idx_start:idx_stop])
                ci[-1] /= np.linalg.norm(ci[-1])
                idx_start = idx_stop

        return self.casci(rot_mo_coeff, ci, eris)[0]

    # energy, gradient, Hessian diagonal and Hessian linear transformation function
    def update_orbs(self, x):
        u = ciah.expmat(self.unpack_uniq_var(x[: self.n_param_orb]))
        self.mo_coeff = self.rotate_mo(self.mo_coeff, u)
        eris = self.ao2mo(self.mo_coeff)

        idx_start = self.n_param_orb
        if self.fcisolver.nroots == 1:
            ci = self.ci + x[idx_start:]
            ci /= np.linalg.norm(ci)
            self.ci = ci
        else:
            ci = []
            for c in self.ci:
                idx_stop = idx_start + c.size
                ci.append(c + x[idx_start:idx_stop])
                ci[-1] /= np.linalg.norm(ci[-1])
                idx_start = idx_stop
            self.ci = ci
            ci = [c.ravel() for c in ci]

        grad, _, hess_x, h_diag = newton_casscf.gen_g_hop(self, self.mo_coeff, ci, eris)

        return (
            self.casci(self.mo_coeff, self.ci, eris)[0],
            2 * grad,
            2 * h_diag,
            lambda x: 2 * hess_x(x),
        )

    def kernel(self, mo_coeff=None, ci0=None, callback=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if callback is None:
            callback = self.callback

        if ci0 is None:
            ci0 = self.ci

        # initial guess
        eris = self.ao2mo(mo_coeff)
        self.e_tot, self.e_cas, fcivec = self.casci(mo_coeff, ci0, eris)
        if self.fcisolver.nroots == 1:
            self.ci = fcivec.ravel()
        else:
            self.ci = [c.ravel() for c in fcivec]

        # number of unique orbital rotation parameters
        self.n_param_orb = np.count_nonzero(
            self.uniq_var_indices(
                self.mo_coeff.shape[1], self.ncore, self.ncas, self.frozen
            )
        )

        # number of CI parameters
        if self.fcisolver.nroots == 1:
            self.n_param_ci = self.ci.size
        else:
            self.n_param_ci = sum(c.size for c in self.ci)

        # number of parameters
        self.n_param = self.n_param_orb + self.n_param_ci

        # call solver
        solver(
            self.func,
            self.update_orbs,
            self.n_param,
            getattr(self, "precond", None),
            getattr(self, "conv_check", None),
            getattr(self, "stability", None),
            getattr(self, "line_search", None),
            getattr(self, "davidson", None),
            getattr(self, "jacobi_davidson", None),
            getattr(self, "prefer_jacobi_davidson", None),
            getattr(self, "conv_tol", None),
            getattr(self, "n_random_trial_vectors", None),
            getattr(self, "start_trust_radius", None),
            getattr(self, "n_macro", None),
            getattr(self, "n_micro", None),
            getattr(self, "global_red_factor", None),
            getattr(self, "local_red_factor", None),
            getattr(self, "seed", None),
            getattr(self, "verbose", None),
            getattr(self, "logger", None),
        )
        self.converged = True
        eris = self.ao2mo(self.mo_coeff)
        self.e_tot, self.e_cas, fcivec = self.casci(self.mo_coeff, self.ci, eris)
        if self.fcisolver.nroots == 1:
            self.ci = fcivec.ravel()
        else:
            self.ci = [c.ravel() for c in fcivec]

        if self.canonicalization:
            self.mo_coeff, self.ci, self.mo_energy = self.canonicalize(
                self.mo_coeff,
                self.ci,
                eris,
                self.sorting_mo_energy,
                self.natorb,
            )
        else:
            self.mo_energy = None

        self._finalize()

        return (
            self.converged,
            self.e_tot,
            self.e_cas,
            self.ci,
            self.mo_coeff,
            self.mo_energy,
        )

    def stability(self):
        _, _, h_diag, hess_x = self.update_orbs(
            np.zeros(self.n_param, dtype=np.float64)
        )
        return stability_check(
            h_diag,
            hess_x,
            self.n_param,
            getattr(self, "precond", None),
            getattr(self, "jacobi_davidson", None),
            getattr(self, "conv_tol", None),
            getattr(self, "n_random_trial_vectors", None),
            getattr(self, "n_iter", None),
            getattr(self, "verbose", None),
            getattr(self, "logger", None),
        )


def casscf_to_otr(casscf):
    if isinstance(casscf, CASSCFOTR):
        return casscf
    
    if not isinstance(casscf, newton_casscf.CASSCF):
        casscf = casscf.newton()

    casscf_otr = CASSCFOTR(casscf._scf, casscf.ncas, casscf.nelecas)
    casscf_otr.__dict__.update(casscf.__dict__)

    if isinstance(casscf, addons.StateAverageMCSCFSolver):
        wfnsym = getattr(casscf, "wfnsym", None)
        casscf_otr = casscf_otr.state_average_(casscf.weights, wfnsym)
        raise RuntimeError(
            "State-averaged CASSCF calculations currently do not work with the "
            "OpenTrustRegion solver"
        )

    return casscf_otr
