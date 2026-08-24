# Copyright (C) 2025- Jonas Greiner
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import annotations

import weakref
import numpy as np
import scipy as sc
from pyscf import gto, scf, lo, lib
from pyscf.soscf import ciah, newton_ah
from pyscf.mcscf import casci, newton_casscf, addons
from pyopentrustregion import SolverSettings, StabilitySettings, solver, stability_check
from pyopentrustregion.python_interface import SolverSettingsC, StabilitySettingsC
from pyopentrustregion.extensions.quasi_newton import (
    QNSettings,
    update_orbs_qn_factory,
    update_orbs_qn_deconstructor,
)
from pyopentrustregion.extensions.oao import OAOSettings, oao_factory, oao_deconstructor
from pyopentrustregion.extensions.arh import ARHSettings, arh_factory, arh_deconstructor
from pyopentrustregion.extensions.s_gek import (
    SGEKSettings,
    update_orbs_s_gek_factory,
    update_orbs_s_gek_deconstructor,
)
from pyopentrustregion.extensions.oao.python_interface import OAOSettingsC
from pyopentrustregion.extensions.arh.python_interface import ARHSettingsC
from pyopentrustregion.extensions.quasi_newton.python_interface import QNSettingsC
from pyopentrustregion.extensions.s_gek.python_interface import SGEKSettingsC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple, Callable, Optional, Union


solver_setting_fields = [
    field[0] for field in SolverSettingsC._fields_ if field[0] != "initialized"
]
stability_setting_fields = [
    field[0] for field in StabilitySettingsC._fields_ if field[0] != "initialized"
]
oao_setting_fields = [
    field[0] for field in OAOSettingsC._fields_ if field[0] != "initialized"
]
arh_setting_fields = [
    field[0] for field in ARHSettingsC._fields_ if field[0] != "initialized"
]
qn_setting_fields = [
    field[0] for field in QNSettingsC._fields_ if field[0] != "initialized"
]
s_gek_setting_fields = [
    field[0] for field in SGEKSettingsC._fields_ if field[0] != "initialized"
]


class OTR:
    _keys = set(
        solver_setting_fields
        + stability_setting_fields
        + oao_setting_fields
        + arh_setting_fields
        + qn_setting_fields
        + s_gek_setting_fields
        + [
            "saved_func",
            "saved_grad",
            "saved_h_diag",
            "saved_hess_x",
            "n_update_orbs",
            "n_hess_x",
            "oao",
            "oao_update_orbs_called",
            "arh",
            "s_gek",
            "pseudo_canonicalization",
        ]
    )

    def __init__(self):
        self.saved_func = None
        self.saved_grad = None
        self.saved_h_diag = None
        self.saved_hess_x = None
        self.n_update_orbs = 0
        self.n_hess_x = 0

    # stability check function
    def stability_check(self) -> Tuple[bool, np.ndarray]:
        # get Hessian diagonal and linear transformation at current point
        kappa = np.zeros(self.n_param, dtype=np.float64)
        grad = np.empty(self.n_param, dtype=np.float64)
        h_diag = np.empty(self.n_param, dtype=np.float64)
        _, hess_x = self.update_orbs(kappa, grad, h_diag)

        # initialize settings
        settings = StabilitySettings()
        for setting in stability_setting_fields:
            if hasattr(self, setting) and (
                setting != "conv_check"
                or not isinstance(getattr(self, "conv_check", None), bool)
            ):
                setattr(settings, setting, getattr(self, setting))

        # run stability check
        direction = np.empty(self.n_param, dtype=np.float64)
        stable = stability_check(h_diag, hess_x, self.n_param, settings, direction)

        return stable, direction


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
        self, kappa: np.ndarray, grad: np.ndarray, h_diag: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        if (
            np.sum(np.abs(kappa)) > 0.0
            or self.saved_func is None
            or self.saved_grad is None
            or self.saved_h_diag is None
            or self.saved_hess_x is None
        ):
            u = ciah.expmat(self.unpack(kappa))
            self.mo_coeff = self.mo_coeff @ u
            self.saved_func = self.cost_function(u)
            self.saved_grad, self.saved_hess_x, self.saved_h_diag = self.gen_g_hop(u)

            self.n_update_orbs += 1

        grad[:] = 2 * self.saved_grad
        h_diag[:] = 2 * self.saved_h_diag

        def hess_x(x: np.ndarray, hx: np.ndarray) -> None:
            hx[:] = 2 * self.saved_hess_x(x)
            self.n_hess_x += 1

        return self.saved_func, hess_x

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

        # initialize settings
        settings = SolverSettings()
        for setting in solver_setting_fields:
            if hasattr(self, setting) and (
                setting != "conv_check"
                or not isinstance(getattr(self, "conv_check", None), bool)
            ):
                setattr(settings, setting, getattr(self, setting))
        for setting in stability_setting_fields:
            if hasattr(self, setting) and (
                setting != "conv_check"
                or not isinstance(getattr(self, "conv_check", None), bool)
            ):
                setattr(settings.stability_settings, setting, getattr(self, setting))

        # call solver
        solver(self.func, self.update_orbs, self.n_param, settings)

        return self.mo_coeff


class PipekMezeyOTR(lo.PipekMezey, BoysOTR):

    # cost function
    def func(self, kappa: np.ndarray) -> float:
        u = ciah.expmat(self.unpack(kappa))
        return -self.cost_function(u)

    # cost function, gradient, Hessian diagonal and Hessian linear transformation
    # function
    def update_orbs(
        self, kappa: np.ndarray, grad: np.ndarray, h_diag: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        if (
            np.sum(np.abs(kappa)) > 0.0
            or self.saved_func is None
            or self.saved_grad is None
            or self.saved_h_diag is None
            or self.saved_hess_x is None
        ):
            u = ciah.expmat(self.unpack(kappa))
            self.mo_coeff = self.mo_coeff @ u
            self.saved_func = self.cost_function(u)
            self.saved_grad, self.saved_hess_x, self.saved_h_diag = self.gen_g_hop(u)

            self.n_update_orbs += 1

        grad[:] = 2 * self.saved_grad
        h_diag[:] = 2 * self.saved_h_diag

        def hess_x(x: np.ndarray, hx: np.ndarray) -> None:
            hx[:] = 2 * self.saved_hess_x(x)
            self.n_hess_x += 1

        return -self.saved_func, hess_x


class EdmistonRuedenbergOTR(lo.EdmistonRuedenberg, PipekMezeyOTR):
    pass


if hasattr(lo, "FourthMoment"):

    class FourthMomentOTR(lo.FourthMoment, BoysOTR):
        pass

else:

    class FourthMomentOTR:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "FourthMomentOTR requires PySCF with lo.FourthMoment. "
                "Please install a compatible PySCF version."
            )


class SecondOrderOTR(OTR, newton_ah._CIAH_SOSCF):

    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    dm: np.ndarray
    vhf: np.ndarray

    def __init__(self, mf: scf.SCF):
        OTR.__init__(self)

        self.__dict__.update(mf.__dict__)
        self._scf = mf
        self.pseudo_canonicalization = False
        self.oao_update_orbs_called = False

    # energy function
    def func(self, kappa: np.ndarray) -> float:
        u = self.exp_mat(self.unpack(kappa))
        rot_mo_coeff = self.rotate_mo(self.mo_coeff, u)
        dm = self.make_rdm1(rot_mo_coeff, self.mo_occ)
        vhf = self._scf.get_veff(self._scf.mol, dm)
        return self._scf.energy_tot(dm, self.h1e, vhf)

    # energy, gradient, Hessian diagonal and Hessian linear transformation function
    def update_orbs(
        self, kappa: np.ndarray, grad: np.ndarray, h_diag: np.ndarray
    ) -> Tuple[float, Callable[[np.ndarray], np.ndarray]]:
        if (
            np.sum(np.abs(kappa)) > 0.0
            or self.saved_func is None
            or self.saved_grad is None
            or self.saved_h_diag is None
            or self.saved_hess_x is None
        ):
            # perform orbital rotation and get new fock matrix if not already done in
            # step modification function
            if not self.pseudo_canonicalization:
                u = self.exp_mat(self.unpack(kappa))
                self.mo_coeff = self.rotate_mo(self.mo_coeff, u)
                self.dm = self.make_rdm1(self.mo_coeff, self.mo_occ)
                self.vhf = self._scf.get_veff(self._scf.mol, self.dm)
                self.fock = self.get_fock(self.h1e, self.s1e, self.vhf, self.dm)
            self.saved_func = self._scf.energy_tot(self.dm, self.h1e, self.vhf)
            self.saved_grad, self.saved_hess_x, self.saved_h_diag = self.gen_g_hop(
                self.mo_coeff, self.mo_occ, self.fock
            )

            self.n_update_orbs += 1

        grad[:] = 2 * self.saved_grad[self.kappa_mask]
        h_diag[:] = 2 * self.saved_h_diag[self.kappa_mask]

        def hess_x(x: np.ndarray, hx: np.ndarray) -> None:
            x_full = np.zeros_like(self.kappa_mask, dtype=np.float64)
            x_full[self.kappa_mask] = x
            hx[:] = 2 * self.saved_hess_x(x_full)[self.kappa_mask]
            self.n_hess_x += 1

        return self.saved_func, hess_x

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

        self.mo_coeff, self.mo_occ = np.asarray(mo_coeff), mo_occ

        # fix phases of MO coefficient to improve deterministic behavior
        self.fix_phase()

        # get indices of all mixed occupation combinations
        self.get_indices()

        # initialize settings (only add conv_check if it is a function, only add
        # modify_step if pseudo_canonicalization is True)
        settings = SolverSettings()
        for setting in solver_setting_fields:
            if (
                hasattr(self, setting)
                and (
                    setting != "conv_check"
                    or not isinstance(getattr(self, "conv_check", None), bool)
                )
                and setting != "modify_step"
            ) or (setting == "modify_step" and self.pseudo_canonicalization):
                setattr(settings, setting, getattr(self, setting))
        for setting in stability_setting_fields:
            if hasattr(self, setting) and (
                setting != "conv_check"
                or not isinstance(getattr(self, "conv_check", None), bool)
            ):
                setattr(settings.stability_settings, setting, getattr(self, setting))

        # set default values for OTR extensions
        if hasattr(self, "arh") and self.arh:
            # optimization is performed in orthogonal AO basis for ARH
            if hasattr(self, "oao") and not self.oao:
                raise RuntimeError(
                    "ARH can only be performed in the orthogonal AO basis."
                )
            self.oao = True
        if not hasattr(self, "oao"):
            self.oao = False
        if not hasattr(self, "arh"):
            self.arh = False
        if not hasattr(self, "hess_update_scheme"):
            self.hess_update_scheme = None
        if not hasattr(self, "s_gek"):
            self.s_gek = False

        # deallocate resources from a previous kernel() call on this instance before
        # registering a finalizer for the current one
        if getattr(self, "_finalizer", None) is not None and self._finalizer.alive:
            self._finalizer()
        self._finalizer = weakref.finalize(
            self, self._cleanup, self.arh, self.hess_update_scheme, self.s_gek, self.oao
        )

        # number of parameters
        if self.oao:
            # number of particles for ARH
            n_particle = 1 if isinstance(self, RHFOTR) else 2
            restricted = isinstance(self, RHFOTR) or isinstance(self, ROHFOTR)
            # number of parameters in AO basis
            self.n_param = self._scf.mol.nao * (self._scf.mol.nao - 1) // 2
            if not restricted:
                self.n_param *= n_particle
        else:
            self.n_param = np.count_nonzero(self.kappa_mask)

        # turn on automatic stability check for approximate Hessians
        if (not hasattr(self, "stability") or self.stability is None) and (
            self.hess_update_scheme is not None or self.arh or self.s_gek
        ):
            self.stability = True

        # check if orthogonal AO basis is to be used
        if self.oao:
            if isinstance(self, RHFOTR):
                dm_per_spin_ao = self.dm / 2
            else:
                dm_per_spin_ao = self.dm
            oao_settings = OAOSettings()
            oao_settings.restricted = restricted
            self.func, oao_update_orbs, settings.project = oao_factory(
                dm_per_spin_ao,
                self.s1e,
                n_particle,
                self._scf.mol.nao,
                self.get_energy,
                self.update_dm,
                oao_settings,
            )

            def wrapped_update_orbs(kappa, grad, h_diag):
                func, hess_x = oao_update_orbs(kappa, grad, h_diag)
                if np.sum(np.abs(kappa)) > 0.0 or not self.oao_update_orbs_called:
                    self.n_update_orbs += 1
                self.oao_update_orbs_called = True

                def wrapped_hess_x(x, hx):
                    hess_x(x, hx)
                    self.n_hess_x += 1

                return func, wrapped_hess_x

            self.update_orbs = wrapped_update_orbs

        # define callback functions for approximate Hessians
        if self.arh:
            if self.pseudo_canonicalization:
                raise RuntimeError("Pseudo-canonicalization is not supported for ARH.")
            arh_settings = ARHSettings()
            arh_settings.restricted = restricted
            if hasattr(self, "arh_type"):
                arh_settings.arh_type = self.arh_type
                settings.hess_symm = not self.arh_type == "standard"
            else:
                settings.hess_symm = True
            self.func, approx_update_orbs, settings.project = arh_factory(
                dm_per_spin_ao,
                self.s1e,
                n_particle,
                self._scf.mol.nao,
                self.get_energy,
                (
                    self.update_dm_jk
                    if isinstance(self, ROHFOTR) or isinstance(self, UHFOTR)
                    else self.update_dm
                ),
                arh_settings,
            )

            def wrapped_approx_update_orbs(kappa, grad, h_diag):
                func, hess_x = approx_update_orbs(kappa, grad, h_diag)
                if np.sum(np.abs(kappa)) > 0.0 or not self.oao_update_orbs_called:
                    self.n_update_orbs += 1
                self.oao_update_orbs_called = True
                return func, hess_x

            self.approx_update_orbs = wrapped_approx_update_orbs
        elif self.hess_update_scheme is not None:
            qn_settings = QNSettings()
            qn_settings.hess_update_scheme = self.hess_update_scheme
            if hasattr(self, "max_points"):
                qn_settings.max_points = self.max_points
            self.approx_update_orbs = update_orbs_qn_factory(
                self.update_orbs,
                self.transport,
                self.init_hess,
                self.n_param,
                qn_settings,
            )
        elif self.s_gek:
            if self.pseudo_canonicalization:
                raise RuntimeError(
                    "Pseudo-canonicalization is not supported for S-GEK."
                )
            if self.oao:
                raise RuntimeError(
                    "S-GEK cannot be performed in the orthogonal AO basis."
                )
            if isinstance(self, ROHFOTR):
                raise RuntimeError(
                    "ROHFOTR is not supported for S-GEK because history cannot be "
                    "transformed consistently to make redundant blocks vanish for flag "
                    "manifold."
                )
            s_gek_settings = SGEKSettings()
            if hasattr(self, "use_subspace"):
                s_gek_settings.use_subspace = self.use_subspace
            if hasattr(self, "max_points"):
                s_gek_settings.max_points = self.max_points
            self.approx_update_orbs = update_orbs_s_gek_factory(
                self.update_orbs, self.change_reference, self.n_param, s_gek_settings
            )

        # accelerate stability check with approximate Hessian information
        if (
            self.hess_update_scheme is not None or self.arh or self.s_gek
        ) and self.stability:
            kappa = np.zeros(self.n_param, dtype=np.float64)
            grad = np.empty(self.n_param, dtype=np.float64)
            h_diag = np.empty(self.n_param, dtype=np.float64)
            settings.stability_hess_x = self.update_orbs(kappa, grad, h_diag)[1]
            if not hasattr(self, "diag_solver"):
                settings.stability_settings.diag_solver = "jacobi-davidson"
                settings.stability_settings.jacobi_davidson_start = 0
            if settings.stability_settings.diag_solver == "jacobi-davidson":
                settings.stability_settings.approx_hess_x = self.approx_update_orbs(
                    kappa, grad, h_diag
                )[1]

        # call solver
        solver(
            self.func,
            (
                self.update_orbs
                if not hasattr(self, "approx_update_orbs")
                else self.approx_update_orbs
            ),
            self.n_param,
            settings,
        )

        # get canonical orbitals
        if self.oao:
            self.dm = 2 * dm_per_spin_ao if isinstance(self, RHFOTR) else dm_per_spin_ao
        self.converged = True
        if self.oao:
            self.mo_occ, self.mo_coeff = self.get_orth_mo_coeff()
        self.dm = self.make_rdm1(self.mo_coeff, self.mo_occ)
        vhf = self._scf.get_veff(self._scf.mol, self.dm)
        self.e_tot = self._scf.energy_tot(self.dm, self.h1e, vhf)
        fock = self._scf.get_fock(self.h1e, self.s1e, vhf, self.dm)
        self.mo_energy, self.mo_coeff = self._scf.canonicalize(
            self.mo_coeff, self.mo_occ, fock
        )
        self._finalize()

        return self.e_tot

    @staticmethod
    def _cleanup(arh, hess_update_scheme, s_gek, oao):
        # call deconstructor
        if arh:
            arh_deconstructor()
        elif hess_update_scheme is not None:
            update_orbs_qn_deconstructor()
        elif s_gek:
            update_orbs_s_gek_deconstructor()
        elif oao:
            oao_deconstructor()


class RHFOTR(SecondOrderOTR, newton_ah._SecondOrderRHF):

    # fix phases of MO coefficient to improve deterministic behavior
    def fix_phase(self):
        abs_mo_coeff = np.abs(self.mo_coeff)
        max_mo_coeff = np.max(abs_mo_coeff, axis=0)
        mask = np.isclose(abs_mo_coeff, max_mo_coeff, atol=1e-12)
        idx = np.argmax(mask, axis=0)
        cols = np.arange(self.mo_coeff.shape[1])
        vals = self.mo_coeff[idx, cols]
        signs = np.sign(vals)
        signs[signs == 0] = 1
        self.mo_coeff *= signs[np.newaxis, :]

    def get_indices(self):
        # get occupation indices
        self.occ_idx = self.mo_occ == 2
        self.virt_idx = ~self.occ_idx

        # get number of closed, open and virtual orbitals
        self.n_occ = np.count_nonzero(self.occ_idx)
        self.n_virt = np.count_nonzero(self.virt_idx)

        # full non-redundant mask for the full matrix
        self.rot_matrix_mask = self.virt_idx[:, None] & self.occ_idx

        # get non-redundant mask for the kappa vector which removes the redundancies
        # from occupation
        self.kappa_mask = self.rot_matrix_mask[self.rot_matrix_mask]

        # modifiy masks in case of symmetry
        if self._scf.mol.symmetry:
            orbsym = self.get_orbsym(self.mo_coeff)
            sym_allow = orbsym[:, None] == orbsym
            self.kappa_mask = sym_allow[self.rot_matrix_mask]
            self.rot_matrix_mask[self.rot_matrix_mask] = self.kappa_mask

        # get non-redundant mask for virtual-closed, virtual-open, and open-closed
        # blocks which only describes which of those rotations are redundant due to
        # symmetry
        self.rot_matrix_mask_vo = self.rot_matrix_mask[
            np.ix_(self.virt_idx, self.occ_idx)
        ]

    # function to compute exponential of anti-symmetric matrix
    def exp_mat(self, matrix):
        return ciah.expmat(matrix)

    # unpack matrix
    def unpack(self, kappa):
        matrix = np.zeros(2 * (self.mol.nao,), dtype=np.float64)
        matrix[self.rot_matrix_mask] = kappa
        return matrix - matrix.T

    # unpack matrix into virtual-occupied block
    def unpack_vo(self, kappa):
        matrix = np.zeros((self.n_virt, self.n_occ), dtype=np.float64)
        matrix[self.rot_matrix_mask_vo] = kappa
        return matrix

    # pack matrix from virtual occupied block
    def pack_vo(self, matrix):
        return matrix.ravel()[self.kappa_mask]

    # energy function from density matrix
    def get_energy(self, dm: np.ndarray) -> float:
        return self._scf.energy_tot(2.0 * dm, self.h1e)

    # update density matrix
    def update_dm(
        self, dm: np.ndarray, fock: np.ndarray
    ) -> Tuple[float, Callable[[np.ndarray, np.ndarray], None]]:
        vhf = self._scf.get_veff(self._scf.mol, 2.0 * dm)
        fock[:, :] = self.get_fock(self.h1e, self.s1e, vhf, 2.0 * dm)
        return self._scf.energy_tot(2.0 * dm, self.h1e, vhf), self.get_response_factory(
            self.gen_response(dm0=2.0 * dm, hermi=1, singlet=None)
        )

    def get_response_factory(
        self, gen_response: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray, np.ndarray], None]:
        def get_response(dm: np.ndarray, response: np.ndarray):
            response[:, :] = gen_response(2.0 * dm)

        return get_response

    # modify step to pseudo-canonical orbitals
    def modify_step(self, kappa: np.ndarray):
        # set new orbitals
        u = self.exp_mat(self.unpack(kappa))
        self.mo_coeff = self.rotate_mo(self.mo_coeff, u)
        mo_coeff_occ = self.mo_coeff[:, self.occ_idx]
        mo_coeff_virt = self.mo_coeff[:, self.virt_idx]

        # build Fock matrix at new orbitals and build pseudo-canonicalization
        # transformation
        self.dm = self.make_rdm1(self.mo_coeff, self.mo_occ)
        self.vhf = self._scf.get_veff(self._scf.mol, self.dm)
        self.fock = self.get_fock(self.h1e, self.s1e, self.vhf, self.dm)
        if self._scf.mol.symmetry:
            self.u_occ = np.zeros((self.n_occ, self.n_occ))
            self.u_virt = np.zeros((self.n_virt, self.n_virt))
            irreps = set(self.orbsym)
            for ir in irreps:
                ir_idx = self.orbsym == ir
                occ_ir_idx = ir_idx & self.occ_idx
                if np.count_nonzero(occ_ir_idx) > 0:
                    mo_coeff_occ_ir = self.mo_coeff[:, occ_ir_idx]
                    fock_occ = mo_coeff_occ_ir.T @ self.fock @ mo_coeff_occ_ir
                    local_irrep_idx = np.where(self.orbsym[self.occ_idx] == ir)[0]
                    self.u_occ[np.ix_(local_irrep_idx, local_irrep_idx)] = (
                        np.linalg.eigh(fock_occ)[1]
                    )
                virt_ir_idx = ir_idx & self.virt_idx
                if np.count_nonzero(virt_ir_idx) > 0:
                    mo_coeff_virt_ir = self.mo_coeff[:, virt_ir_idx]
                    fock_virt = mo_coeff_virt_ir.T @ self.fock @ mo_coeff_virt_ir
                    local_irrep_idx = np.where(self.orbsym[self.virt_idx] == ir)[0]
                    self.u_virt[np.ix_(local_irrep_idx, local_irrep_idx)] = (
                        np.linalg.eigh(fock_virt)[1]
                    )
        else:
            fock_oo = mo_coeff_occ.T @ self.fock @ mo_coeff_occ
            self.u_occ = np.linalg.eigh(fock_oo)[1]
            fock_vv = mo_coeff_virt.T @ self.fock @ mo_coeff_virt
            self.u_virt = np.linalg.eigh(fock_vv)[1]

        # rotate MO coefficients to pseudo-canonical orbitals
        self.mo_coeff[:, self.occ_idx] = self.rotate_mo(mo_coeff_occ, self.u_occ)
        self.mo_coeff[:, self.virt_idx] = self.rotate_mo(mo_coeff_virt, self.u_virt)

        # rotate step to pseudo-canonical orbitals
        kappa_vo = self.unpack_vo(kappa)
        kappa_vo = self.u_virt.T @ kappa_vo @ self.u_occ
        kappa[:] = self.pack_vo(kappa_vo)

    # transport tangent vector along geodesic
    def transport(self, geodesic: np.ndarray, tangent_vector: np.ndarray):
        # perform pseudo-canonicalization if enabled
        if self.pseudo_canonicalization:
            tangent_vector[:] = self.pack_vo(
                self.u_virt.T @ self.unpack_vo(tangent_vector) @ self.u_occ
            )

        return

    # Hessian initialization
    def init_hess(self, vector: np.ndarray):
        mo_coeff_occ = self.mo_coeff[:, self.occ_idx]
        mo_coeff_virt = self.mo_coeff[:, self.virt_idx]

        fock_oo = mo_coeff_occ.T @ self.fock @ mo_coeff_occ
        fock_vv = mo_coeff_virt.T @ self.fock @ mo_coeff_virt

        x_vo = self.unpack_vo(vector)
        x_vo = 2 * (fock_vv @ x_vo - x_vo @ fock_oo)
        vector[:] = self.pack_vo(x_vo)

    # change of reference
    def change_reference(
        self,
        new_ref: np.ndarray,
        kappa: np.ndarray,
        local_grad: np.ndarray,
        grad: np.ndarray,
    ):
        """
        this function performs a gauge transformation for the S-GEK extension which
        requires the history of orbital rotation parameters to be have vanishing
        occupied-occupied and virtual-virtual blocks such that the GEK metric is
        consistent
        """
        # get reference rotation matrix
        ref_rot_mat = self.exp_mat(self.unpack(new_ref))

        for i in range(kappa.shape[0]):
            # get matrix to be rotated
            rot_mat = self.exp_mat(self.unpack(kappa[i, :]))

            # get combined rotation matrix
            combined_rot = ref_rot_mat.T @ rot_mat
            combined_rot_oo = combined_rot[np.ix_(self.occ_idx, self.occ_idx)]
            combined_rot_ov = combined_rot[np.ix_(self.occ_idx, self.virt_idx)]
            combined_rot_vo = combined_rot[np.ix_(self.virt_idx, self.occ_idx)]
            combined_rot_vv = combined_rot[np.ix_(self.virt_idx, self.virt_idx)]

            # # perform sine cosine decomposition of combined rotation matrix
            # n_occ = np.count_nonzero(self.occ_idx)
            # n_virt = np.count_nonzero(self.virt_idx)
            # rank = min(n_occ, self.mol.nao - n_occ)
            # ((u1, u2), theta, (v1h, v2h)) = sc.linalg.cossin(
            #     [combined_rot_oo, combined_rot_ov, combined_rot_vo, combined_rot_vv],
            #     separate=True,
            # )
            # lower_kappa = np.zeros((n_virt, n_occ), dtype=np.float64)
            # lower_kappa[n_virt - rank :, n_occ - rank :] = np.diag(theta)
            # kappa[i, :] = (u2 @ lower_kappa @ u1.T).ravel()

            # # get rotation matrices to new basis
            # rot_occ = (u1 @ v1h).T
            # rot_virt = (u2 @ v2h).T

            # test gauge transformation
            # U_new = self.exp_mat(self.unpack(kappa[i, :]))
            # U_new[np.ix_(self.occ_idx, self.occ_idx)] = U_new[np.ix_(self.occ_idx, self.occ_idx)] @ rot_occ.T
            # U_new[np.ix_(self.occ_idx, self.virt_idx)] = U_new[np.ix_(self.occ_idx, self.virt_idx)] @ rot_virt.T
            # U_new[np.ix_(self.virt_idx, self.occ_idx)] = U_new[np.ix_(self.virt_idx, self.occ_idx)] @ rot_occ.T
            # U_new[np.ix_(self.virt_idx, self.virt_idx)] = U_new[np.ix_(self.virt_idx, self.virt_idx)] @ rot_virt.T
            # print("kappa_diff", np.linalg.norm(combined_rot - U_new))

            # perform SVD of occupied block
            u, s, vh = sc.linalg.svd(combined_rot_oo)
            # u2, s2, v2h = sc.linalg.svd(combined_rot_vv)

            # get rotation matrix to make occupied-occupied block vanish
            rot_occ = u @ vh  # vh.T @ u.T
            rot_virt = (
                combined_rot_vv
                - combined_rot_vo @ (vh.T / (s + 1)) @ u.T @ combined_rot_ov
            )
            # rot_virt = v2h.T @ u2.T

            # get combined rotation matrix with vanishing oo and vv blocks
            scos = np.ones_like(s)
            mask = np.abs(1 - s) > np.sqrt(np.finfo(float).eps)
            scos[mask] = np.arccos(s[mask]) / np.sqrt(1 - s[mask] ** 2)
            kappa[i, :] = self.pack_vo(combined_rot_vo @ (vh.T * scos) @ u.T)

            # # test gauge transformation
            # U_new = self.exp_mat(self.unpack(kappa[i, :]))
            # U_new[np.ix_(self.occ_idx, self.occ_idx)] = U_new[np.ix_(self.occ_idx, self.occ_idx)] @ rot_occ.T
            # U_new[np.ix_(self.occ_idx, self.virt_idx)] = U_new[np.ix_(self.occ_idx, self.virt_idx)] @ rot_virt.T
            # U_new[np.ix_(self.virt_idx, self.occ_idx)] = U_new[np.ix_(self.virt_idx, self.occ_idx)] @ rot_occ.T
            # U_new[np.ix_(self.virt_idx, self.virt_idx)] = U_new[np.ix_(self.virt_idx, self.virt_idx)] @ rot_virt.T
            # print("kappa_diff", np.linalg.norm(combined_rot - U_new))

            # transform local gradients to new orbitals
            n_occ = rot_occ.shape[0]
            n_virt = rot_virt.shape[0]
            occ_unit = np.allclose(rot_occ, np.eye(n_occ), atol=1.0e-10, rtol=0)
            virt_unit = np.allclose(rot_virt, np.eye(n_virt), atol=1.0e-10, rtol=0)
            if not occ_unit or not virt_unit:
                local_grad_2d = self.unpack_vo(local_grad[i, :])
                if not occ_unit:
                    local_grad_2d = local_grad_2d @ rot_occ
                if not virt_unit:
                    local_grad_2d = rot_virt.T @ local_grad_2d
                local_grad[i, :] = self.pack_vo(local_grad_2d)

            # # test gradient transformation
            # rot = self.exp_mat(self.unpack(kappa[i, :]))
            # mo_coeff = self.rotate_mo(self.mo_coeff, rot)
            # dm = self.make_rdm1(mo_coeff, self.mo_occ)
            # vhf = self._scf.get_veff(self._scf.mol, dm)
            # fock = self.get_fock(self.h1e, self.s1e, vhf, dm)
            # grad_full, _, _ = self.gen_g_hop(mo_coeff, self.mo_occ, fock)
            # grad_test = 2 * grad_full[self.kappa_mask]
            # print("local_grad_diff", np.linalg.norm(grad_test - local_grad[i, :]))

            # transform gradients to new reference
            u, s, vh = sc.linalg.svd(self.unpack_vo(kappa[i, :]), full_matrices=False)
            t0 = self.unpack_vo(local_grad[i, :])
            z = u.T @ t0 @ vh.T
            dm = 0.5 * np.sinc((s.reshape(-1, 1) - s) / np.pi)
            dp = 0.5 * np.sinc((s.reshape(-1, 1) + s) / np.pi)
            grad_2d = u @ ((z + z.T) * dm + (z - z.T) * dp) @ vh
            # decide whether the occupied orbitals or virtual orbitals define the rank
            # of the parameter matrix
            if n_occ <= n_virt:
                grad_2d += (
                    (np.eye(n_virt) - u @ u.T) @ t0 @ ((vh.T * np.sinc(s / np.pi)) @ vh)
                )
            else:
                grad_2d += (
                    ((u * np.sinc(s / np.pi)) @ u.T) @ t0 @ (np.eye(n_occ) - vh.T @ vh)
                )
            grad[i, :] = self.pack_vo(grad_2d)

        return

    # function to get orthogonal MO coefficients and occupations
    def get_orth_mo_coeff(self):
        eigvals_s, eigvecs_s = np.linalg.eigh(self.s1e)
        s_sqrt = eigvecs_s @ np.diag(np.sqrt(eigvals_s)) @ eigvecs_s.T
        s_inv_sqrt = eigvecs_s @ np.diag(1.0 / np.sqrt(eigvals_s)) @ eigvecs_s.T
        dm_orth = s_sqrt @ self.dm @ s_sqrt
        mo_occ, mo_coeff_oao = np.linalg.eigh(dm_orth)
        mo_occ = np.rint(mo_occ)
        mo_coeff = s_inv_sqrt @ mo_coeff_oao
        return mo_occ, mo_coeff


class ROHFOTR(SecondOrderOTR, newton_ah._SecondOrderROHF):

    fix_phase = RHFOTR.fix_phase

    def get_indices(self):
        # get occupation indices
        self.clos_idx = self.mo_occ == 2
        self.open_idx = self.mo_occ == 1
        self.virt_idx = self.mo_occ == 0

        # get number of closed, open and virtual orbitals
        self.n_clos = np.count_nonzero(self.clos_idx)
        self.n_open = np.count_nonzero(self.open_idx)
        self.n_virt = np.count_nonzero(self.virt_idx)

        # define 2D subspace masks with full matrix shape which represent the blocks of
        # the Flag Manifold
        self.vc_mask = self.virt_idx[:, None] & self.clos_idx
        self.vo_mask = self.virt_idx[:, None] & self.open_idx
        self.oc_mask = self.open_idx[:, None] & self.clos_idx

        # full non-redundant mask for the full matrix
        self.rot_matrix_mask = self.vc_mask | self.vo_mask | self.oc_mask

        # get non-redundant mask for the kappa vector which removes the redundancies
        # from occupation
        self.kappa_mask = self.rot_matrix_mask[self.rot_matrix_mask]

        # modifiy masks in case of symmetry
        if self._scf.mol.symmetry:
            orbsym = self.get_orbsym(self.mo_coeff)
            sym_allow = orbsym[:, None] == orbsym
            self.kappa_mask = sym_allow[self.rot_matrix_mask]
            self.rot_matrix_mask[self.rot_matrix_mask] = self.kappa_mask

        # get non-redundant mask for virtual-closed, virtual-open, and open-closed
        # blocks which only describes which of those rotations are redundant due to
        # symmetry
        self.rot_matrix_mask_vc = self.rot_matrix_mask[
            np.ix_(self.virt_idx, self.clos_idx)
        ]
        self.rot_matrix_mask_vo = self.rot_matrix_mask[
            np.ix_(self.virt_idx, self.open_idx)
        ]
        self.rot_matrix_mask_oc = self.rot_matrix_mask[
            np.ix_(self.open_idx, self.clos_idx)
        ]

        # get mask for specific blocks in kappa matrix
        self.kappa_mask_vc = self.vc_mask[self.rot_matrix_mask]
        self.kappa_mask_vo = self.vo_mask[self.rot_matrix_mask]
        self.kappa_mask_oc = self.oc_mask[self.rot_matrix_mask]

    exp_mat = RHFOTR.exp_mat
    unpack = RHFOTR.unpack

    # unpack matrix into virtual-closed, virtual-open, and open-closed blocks
    def unpack_voc(self, kappa):
        matrix_vc = np.zeros((self.n_virt, self.n_clos), dtype=np.float64)
        matrix_vo = np.zeros((self.n_virt, self.n_open), dtype=np.float64)
        matrix_oc = np.zeros((self.n_open, self.n_clos), dtype=np.float64)
        matrix_vc[self.rot_matrix_mask_vc] = kappa[self.kappa_mask_vc]
        matrix_vo[self.rot_matrix_mask_vo] = kappa[self.kappa_mask_vo]
        matrix_oc[self.rot_matrix_mask_oc] = kappa[self.kappa_mask_oc]
        return matrix_vc, matrix_vo, matrix_oc

    # pack matrix from virtual-occupied, virtual-open, and open-closed block
    def pack_voc(self, matrix_vc, matrix_vo, matrix_oc):
        kappa = np.empty(self.n_param, dtype=np.float64)
        kappa[self.kappa_mask_vc] = matrix_vc[self.rot_matrix_mask_vc].ravel()
        kappa[self.kappa_mask_vo] = matrix_vo[self.rot_matrix_mask_vo].ravel()
        kappa[self.kappa_mask_oc] = matrix_oc[self.rot_matrix_mask_oc].ravel()
        return kappa

    # energy function from density matrix
    def get_energy(self, dm: np.ndarray) -> float:
        return self._scf.energy_tot(dm, self.h1e)

    # update_density matrix
    def update_dm(
        self, dm: np.ndarray, fock: np.ndarray
    ) -> Tuple[float, Callable[[np.ndarray, np.ndarray], None]]:
        vhf = self._scf.get_veff(self._scf.mol, dm)
        eff_fock = self._scf.get_fock(self.h1e, self.s1e, vhf, dm)
        fock[0, :, :] = eff_fock.focka
        fock[1, :, :] = eff_fock.fockb
        return self._scf.energy_tot(dm, self.h1e, vhf), self.get_response_factory(
            self.gen_response(dm0=dm, hermi=1)
        )

    # update_density matrix with Coulomb and exchange contributions
    def update_dm_jk(
        self,
        dm: np.ndarray,
        fock: np.ndarray,
        coulomb: np.ndarray,
        exchange: np.ndarray,
    ) -> Tuple[float, Callable[[np.ndarray, np.ndarray], None]]:
        # get Coulomb and exchange matrices
        coulomb[:, :, :], exchange[:, :, :] = self._scf.get_jk(self.mol, dm)

        # construct mean-field potential
        vhf = coulomb[0] + coulomb[1] - exchange

        # construct Fock matrix in AO basis
        eff_fock = self._scf.get_fock(self.h1e, self.s1e, vhf, dm)
        fock[0, :, :] = eff_fock.focka
        fock[1, :, :] = eff_fock.fockb

        return self._scf.energy_tot(dm, self.h1e, vhf), self.get_response_factory(
            self.gen_response(dm0=dm, hermi=1)
        )

    def get_response_factory(
        self, gen_response: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray, np.ndarray], None]:
        def get_response(dm: np.ndarray, response: np.ndarray):
            response[:, :, :] = gen_response(dm)

        return get_response

    # modify step to pseudo-canonical orbitals
    def modify_step(self, kappa: np.ndarray):
        # set new orbitals
        u = self.exp_mat(self.unpack(kappa))
        self.mo_coeff = self.rotate_mo(self.mo_coeff, u)
        mo_coeff_clos = self.mo_coeff[:, self.clos_idx]
        mo_coeff_open = self.mo_coeff[:, self.open_idx]
        mo_coeff_virt = self.mo_coeff[:, self.virt_idx]

        # build Fock matrix at new orbitals and build pseudo-canonical orbital
        # transformation
        self.dm = self.make_rdm1(self.mo_coeff, self.mo_occ)
        self.vhf = self._scf.get_veff(self._scf.mol, self.dm)
        self.fock = self.get_fock(self.h1e, self.s1e, self.vhf, self.dm)
        if self._scf.mol.symmetry:
            irreps = set(self.orbsym)
            self.u_clos = np.zeros((self.n_clos, self.n_clos))
            self.u_open = np.zeros((self.n_open, self.n_open))
            self.u_virt = np.zeros((self.n_virt, self.n_virt))
            for ir in irreps:
                ir_idx = self.orbsym == ir
                for occ_idx, u_occ in zip(
                    [self.clos_idx, self.open_idx, self.virt_idx],
                    [self.u_clos, self.u_open, self.u_virt],
                ):
                    occ_ir_idx = ir_idx & occ_idx
                    if np.count_nonzero(occ_ir_idx) > 0:
                        mo_coeff_occ_ir = self.mo_coeff[:, occ_ir_idx]
                        fock_occ = mo_coeff_occ_ir.T @ self.fock @ mo_coeff_occ_ir
                        local_irrep_idx = np.where(self.orbsym[occ_idx] == ir)[0]
                        u_occ[np.ix_(local_irrep_idx, local_irrep_idx)] = (
                            np.linalg.eigh(fock_occ)[1]
                        )
        else:
            for mo_coeff_occ, u_occ in zip(
                [mo_coeff_clos, mo_coeff_open, mo_coeff_virt],
                [self.u_clos, self.u_open, self.u_virt],
            ):
                fock_occ = mo_coeff_occ.T @ self.fock @ mo_coeff_occ
                u_occ[:] = np.linalg.eigh(fock_occ)[1]

        # rotate MO coefficients to pseudo-canonical orbitals
        self.mo_coeff[:, self.clos_idx] = self.rotate_mo(mo_coeff_clos, self.u_clos)
        self.mo_coeff[:, self.open_idx] = self.rotate_mo(mo_coeff_open, self.u_open)
        self.mo_coeff[:, self.virt_idx] = self.rotate_mo(mo_coeff_virt, self.u_virt)

        # rotate step to pseudo-canonical orbitals
        kappa_vc, kappa_vo, kappa_oc = self.unpack_voc(kappa)
        kappa_vc = self.u_virt.T @ kappa_vc @ self.u_clos
        kappa_vo = self.u_virt.T @ kappa_vo @ self.u_open
        kappa_oc = self.u_open.T @ kappa_oc @ self.u_clos
        kappa[:] = self.pack_voc(kappa_vc, kappa_vo, kappa_oc)

    # transport tangent vector along geodesic
    def transport(self, geodesic: np.ndarray, tangent_vector: np.ndarray):
        """
        this function transports tangent_vector from the tangent space at a starting
        base to the tangent space at a new base along the geodesic. tangent_vector and
        geodesic have the same base. In other words, the displacement described by
        tangent_vector is defined in terms of the MO coefficients defined by
        the starting base, and this function transforms it to the MO coefficients
        define by the new base. The geodesic describes the rotation from the starting
        base to the new base.
        """
        # perform pseudo-canonicalization for history if enabled
        if self.pseudo_canonicalization:
            kappa_vc, kappa_vo, kappa_oc = self.unpack_voc(tangent_vector)
            kappa_vc = self.u_virt.T @ kappa_vc @ self.u_clos
            kappa_vo = self.u_virt.T @ kappa_vo @ self.u_open
            kappa_oc = self.u_open.T @ kappa_oc @ self.u_clos
            tangent_vector[:] = self.pack_voc(kappa_vc, kappa_vo, kappa_oc)

        # unpack into full matrix
        geodesic_full = self.unpack(geodesic)

        # define tolerance for truncation of series expansion
        tol = np.finfo(np.float64).eps

        # k = 0
        tangent_vector_full = self.unpack(tangent_vector)
        # k = 1
        k = 1
        current_term = -(1 / 2) * self.remove_diag_blocks(
            geodesic_full @ tangent_vector_full - tangent_vector_full @ geodesic_full
        )
        tangent_vector_full = tangent_vector_full + current_term
        # k > 1
        while (np.linalg.norm(current_term) > tol) and (k < 200):
            k += 1
            current_term = (-1 / (2 * k)) * self.remove_diag_blocks(
                geodesic_full @ current_term - current_term @ geodesic_full
            )
            tangent_vector_full = tangent_vector_full + current_term
        current_term_norm = np.linalg.norm(current_term)
        if current_term_norm > tol:
            raise RuntimeError(
                "Parallel transport trunctated before tolerance is reached. Norm of "
                f"the last term: {current_term_norm}."
            )

        # ensure the transported kappa and grad are in the tangent space of the new
        # reference
        tangent_vector_full = self.ensure_tangent(tangent_vector_full)

        # pack back into non-redundant parameters
        tangent_vector[:] = tangent_vector_full[self.rot_matrix_mask]

        return

    def remove_diag_blocks(self, matrix):
        """
        this function removes the redundant diagonal blocks of the matrix corresponding
        to closed, open, and virtual orbitals, which ensures the output is in the
        tangent space of the reference
        """
        matrix[np.ix_(self.clos_idx, self.clos_idx)] = 0.0
        matrix[np.ix_(self.open_idx, self.open_idx)] = 0.0
        matrix[np.ix_(self.virt_idx, self.virt_idx)] = 0.0
        return matrix

    def ensure_tangent(self, tangent_vector):
        """
        this function ensures the input tangent vector is in the tangent space of the
        reference by removing the redundant diagonal blocks and anti-symmetrizing the
        matrix
        """
        return self.remove_diag_blocks(0.5 * (tangent_vector - tangent_vector.T))

    # function to get orthogonal MO coefficients and occupations
    def get_orth_mo_coeff(self):
        eigvals_s, eigvecs_s = np.linalg.eigh(self.s1e)
        s_sqrt = eigvecs_s @ np.diag(np.sqrt(eigvals_s)) @ eigvecs_s.T
        s_inv_sqrt = eigvecs_s @ np.diag(1.0 / np.sqrt(eigvals_s)) @ eigvecs_s.T
        dm_orth = s_sqrt @ (self.dm[0] + self.dm[1]) @ s_sqrt
        mo_occ, mo_coeff_oao = np.linalg.eigh(dm_orth)
        mo_occ = np.rint(mo_occ)
        mo_coeff = s_inv_sqrt @ mo_coeff_oao
        return mo_occ, mo_coeff


class UHFOTR(SecondOrderOTR, newton_ah._SecondOrderUHF):

    # fix phases of MO coefficient to improve deterministic behavior
    def fix_phase(self):
        abs_mo_coeff = np.abs(self.mo_coeff)
        max_mo_coeff = np.max(abs_mo_coeff, axis=1)
        for i in range(2):
            mask = np.isclose(abs_mo_coeff[i], max_mo_coeff[i], atol=1e-12)
            idx = np.argmax(mask, axis=0)
            cols = np.arange(self.mo_coeff.shape[2])
            vals = self.mo_coeff[i, idx, cols]
            signs = np.sign(vals)
            signs[signs == 0] = 1
            self.mo_coeff[i] *= signs[np.newaxis, :]

    # energy function from density matrix
    def get_energy(self, dm: np.ndarray) -> float:
        return self._scf.energy_tot(dm, self.h1e)

    # get indices of all mixed occupation combinations
    def get_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        # get occupation indices for alpha and beta spins
        self.occ_idx = [self.mo_occ[0] == 1, self.mo_occ[1] == 1]
        self.virt_idx = [~self.occ_idx[0], ~self.occ_idx[1]]

        # get number of closed, open and virtual orbitals
        self.n_occ = [
            np.count_nonzero(self.occ_idx[0]),
            np.count_nonzero(self.occ_idx[1]),
        ]
        self.n_virt = [
            np.count_nonzero(self.virt_idx[0]),
            np.count_nonzero(self.virt_idx[1]),
        ]

        # get matrix mask for virtual-occupied rotations for alpha and beta spins
        self.rot_matrix_mask = np.stack(
            (
                self.virt_idx[0][:, None] & self.occ_idx[0],
                self.virt_idx[1][:, None] & self.occ_idx[1],
            )
        )

        # get kappa mask for parameters corresponding to virtual-occupied rotations
        self.kappa_mask = self.rot_matrix_mask[self.rot_matrix_mask]

        # modifiy masks in case of symmetry
        if self._scf.mol.symmetry:
            orbsyma, orbsymb = self.get_orbsym(self.mo_coeff)
            sym_allowa = orbsyma[:, None] == orbsyma
            sym_allowb = orbsymb[:, None] == orbsymb
            sym_allow = np.stack((sym_allowa, sym_allowb))
            self.kappa_mask = sym_allow[self.rot_matrix_mask]
            self.rot_matrix_mask[self.rot_matrix_mask] = self.kappa_mask

        # get non-redundant mask for virtual-closed, virtual-open, and open-closed
        # blocks which only describes which of those rotations are redundant due to
        # symmetry
        self.rot_matrix_mask_vo = [
            self.rot_matrix_mask[0][np.ix_(self.virt_idx[0], self.occ_idx[0])],
            self.rot_matrix_mask[1][np.ix_(self.virt_idx[1], self.occ_idx[1])],
        ]

    # function to compute exponential of anti-symmetric matrix
    def exp_mat(self, matrix):
        return [ciah.expmat(matrix[0]), ciah.expmat(matrix[1])]

    # unpack matrix
    def unpack(self, kappa):
        n_param_a = np.count_nonzero(self.rot_matrix_mask[0])
        matrix_list = [
            np.zeros(2 * (self.mol.nao,), dtype=np.float64),
            np.zeros(2 * (self.mol.nao,), dtype=np.float64),
        ]
        matrix_list[0][self.rot_matrix_mask[0]] = kappa[:n_param_a]
        matrix_list[1][self.rot_matrix_mask[1]] = kappa[n_param_a:]
        matrix_list[0] -= matrix_list[0].T
        matrix_list[1] -= matrix_list[1].T
        return matrix_list

    # unpack matrix into virtual-occupied block
    def unpack_vo(self, kappa):
        n_param_a = np.count_nonzero(self.rot_matrix_mask[0])
        matrix = [
            np.zeros((self.n_virt[0], self.n_occ[0]), dtype=np.float64),
            np.zeros((self.n_virt[1], self.n_occ[1]), dtype=np.float64),
        ]
        matrix[0][self.rot_matrix_mask_vo[0]] = kappa[:n_param_a]
        matrix[1][self.rot_matrix_mask_vo[1]] = kappa[n_param_a:]
        return matrix

    # pack matrix from virtual occupied block
    def pack_vo(self, matrix):
        return np.concatenate((matrix[0].ravel(), matrix[1].ravel()))[self.kappa_mask]

    # update density matrix
    def update_dm(
        self, dm: np.ndarray, fock: np.ndarray
    ) -> Tuple[float, Callable[[np.ndarray, np.ndarray], None]]:
        vhf = self._scf.get_veff(self._scf.mol, dm)
        fock[:, :, :] = self._scf.get_fock(self.h1e, self.s1e, vhf, dm)
        return self._scf.energy_tot(dm, self.h1e, vhf), self.get_response_factory(
            self.gen_response(dm0=dm, hermi=1)
        )

    # update density matrix with Coulomb and exchange contributions
    def update_dm_jk(
        self,
        dm: np.ndarray,
        fock: np.ndarray,
        coulomb: np.ndarray,
        exchange: np.ndarray,
    ) -> Tuple[float, Callable[[np.ndarray, np.ndarray], None]]:
        # get Coulomb and exchange matrices
        coulomb[:, :, :], exchange[:, :, :] = self._scf.get_jk(self.mol, dm)

        # construct mean-field potential
        vhf = coulomb[0] + coulomb[1] - exchange

        # construct Fock matrix in AO basis
        fock[:, :, :] = self._scf.get_fock(self.h1e, self.s1e, vhf, dm)

        return self._scf.energy_tot(dm, self.h1e, vhf), self.get_response_factory(
            self.gen_response(dm0=dm, hermi=1)
        )

    def get_response_factory(
        self, gen_response: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray, np.ndarray], None]:
        def get_response(dm: np.ndarray, response: np.ndarray):
            response[:, :, :] = gen_response(dm)

        return get_response

    # modify step to pseudo-canonical orbitals
    def modify_step(self, kappa: np.ndarray):
        # set new orbitals
        u = self.exp_mat(self.unpack(kappa))
        self.mo_coeff = self.rotate_mo(self.mo_coeff, u)
        mo_coeff_occ = [
            self.mo_coeff[0][:, self.occ_idx[0]],
            self.mo_coeff[1][:, self.occ_idx[1]],
        ]
        mo_coeff_virt = [
            self.mo_coeff[0][:, self.virt_idx[0]],
            self.mo_coeff[1][:, self.virt_idx[1]],
        ]

        # build Fock matrix at new orbitals and build pseudo-canonical orbital
        # transformation
        self.dm = self.make_rdm1(self.mo_coeff, self.mo_occ)
        self.vhf = self._scf.get_veff(self._scf.mol, self.dm)
        self.fock = self.get_fock(self.h1e, self.s1e, self.vhf, self.dm)
        if self._scf.mol.symmetry:
            self.u_occ = [
                np.zeros((self.n_occ[0], self.n_occ[0])),
                np.zeros((self.n_occ[1], self.n_occ[1])),
            ]
            self.u_virt = [
                np.zeros((self.n_virt[0], self.n_virt[0])),
                np.zeros((self.n_virt[1], self.n_virt[1])),
            ]
            for spin in [0, 1]:
                irreps = set(self.orbsym[spin])
                for ir in irreps:
                    ir_idx = self.orbsym[spin] == ir
                    occ_ir_idx = ir_idx & self.occ_idx[spin]
                    if np.count_nonzero(occ_ir_idx) > 0:
                        mo_coeff_occ_ir = self.mo_coeff[spin][:, occ_ir_idx]
                        fock_occ = mo_coeff_occ_ir.T @ self.fock[spin] @ mo_coeff_occ_ir
                        local_irrep_idx = np.where(
                            self.orbsym[spin][self.occ_idx[spin]] == ir
                        )[0]
                        self.u_occ[spin][np.ix_(local_irrep_idx, local_irrep_idx)] = (
                            np.linalg.eigh(fock_occ)[1]
                        )
                    virt_ir_idx = ir_idx & self.virt_idx[spin]
                    if np.count_nonzero(virt_ir_idx) > 0:
                        mo_coeff_virt_ir = self.mo_coeff[spin][:, virt_ir_idx]
                        fock_virt = (
                            mo_coeff_virt_ir.T @ self.fock[spin] @ mo_coeff_virt_ir
                        )
                        local_irrep_idx = np.where(
                            self.orbsym[spin][self.virt_idx[spin]] == ir
                        )[0]
                        self.u_virt[spin][np.ix_(local_irrep_idx, local_irrep_idx)] = (
                            np.linalg.eigh(fock_virt)[1]
                        )

        else:
            self.u_occ = []
            self.u_virt = []
            for spin in [0, 1]:
                fock_oo = mo_coeff_occ[spin].T @ self.fock[spin] @ mo_coeff_occ[spin]
                self.u_occ.append(np.linalg.eigh(fock_oo)[1])
                fock_vv = mo_coeff_virt[spin].T @ self.fock[spin] @ mo_coeff_virt[spin]
                self.u_virt.append(np.linalg.eigh(fock_vv)[1])

        # rotate MO coefficients to pseudo-canonical orbitals
        self.mo_coeff[0][:, self.occ_idx[0]] = mo_coeff_occ[0] @ self.u_occ[0]
        self.mo_coeff[1][:, self.occ_idx[1]] = mo_coeff_occ[1] @ self.u_occ[1]
        self.mo_coeff[0][:, self.virt_idx[0]] = mo_coeff_virt[0] @ self.u_virt[0]
        self.mo_coeff[1][:, self.virt_idx[1]] = mo_coeff_virt[1] @ self.u_virt[1]

        # rotate step to pseudo-canonical orbitals
        kappa_vo = self.unpack_vo(kappa)
        kappa_vo[0] = self.u_virt[0].T @ kappa_vo[0] @ self.u_occ[0]
        kappa_vo[1] = self.u_virt[1].T @ kappa_vo[1] @ self.u_occ[1]
        kappa[:] = self.pack_vo(kappa_vo)

    # change of reference
    def change_reference(
        self,
        new_ref: np.ndarray,
        kappa: np.ndarray,
        local_grad: np.ndarray,
        grad: np.ndarray,
    ):
        """
        this function performs a gauge transformation for the S-GEK extension which
        requires the history of orbital rotation parameters to be have vanishing
        occupied-occupied and virtual-virtual blocks such that the GEK metric is
        consistent
        """
        # get reference rotation matrix
        ref_rot_mat = self.exp_mat(self.unpack(new_ref))

        for i in range(kappa.shape[0]):
            # get matrix to be rotated
            rot_mat = self.exp_mat(self.unpack(kappa[i, :]))

            kappa_2d = []
            local_grad_2d = self.unpack_vo(local_grad[i, :])
            grad_2d = []
            for spin in [0, 1]:
                # get combined rotation matrix
                combined_rot = ref_rot_mat[spin].T @ rot_mat[spin]
                combined_rot_oo = combined_rot[
                    np.ix_(self.occ_idx[spin], self.occ_idx[spin])
                ]
                combined_rot_ov = combined_rot[
                    np.ix_(self.occ_idx[spin], self.virt_idx[spin])
                ]
                combined_rot_vo = combined_rot[
                    np.ix_(self.virt_idx[spin], self.occ_idx[spin])
                ]
                combined_rot_vv = combined_rot[
                    np.ix_(self.virt_idx[spin], self.virt_idx[spin])
                ]

                # perform SVD of occupied block
                u, s, vh = sc.linalg.svd(combined_rot_oo)

                # get rotation matrix to make occupied-occupied block vanish
                rot_occ = u @ vh
                rot_virt = (
                    combined_rot_vv
                    - combined_rot_vo @ (vh.T / (s + 1)) @ u.T @ combined_rot_ov
                )

                # get combined rotation matrix with vanishing oo and vv blocks
                scos = np.ones_like(s)
                mask = np.abs(1 - s) > np.sqrt(np.finfo(float).eps)
                scos[mask] = np.arccos(s[mask]) / np.sqrt(1 - s[mask] ** 2)
                kappa_2d.append(combined_rot_vo @ (vh.T * scos) @ u.T)

                # transform local gradients to new orbitals
                n_occ = rot_occ.shape[0]
                n_virt = rot_virt.shape[0]
                occ_unit = np.allclose(rot_occ, np.eye(n_occ), atol=1.0e-10, rtol=0)
                virt_unit = np.allclose(rot_virt, np.eye(n_virt), atol=1.0e-10, rtol=0)
                if not occ_unit or not virt_unit:
                    if not occ_unit:
                        local_grad_2d[spin] = local_grad_2d[spin] @ rot_occ
                    if not virt_unit:
                        local_grad_2d[spin] = rot_virt.T @ local_grad_2d[spin]

                # transform gradients to new reference
                u, s, vh = sc.linalg.svd(kappa_2d[spin], full_matrices=False)
                t0 = local_grad_2d[spin]
                z = u.T @ t0 @ vh.T
                dm = 0.5 * np.sinc((s.reshape(-1, 1) - s) / np.pi)
                dp = 0.5 * np.sinc((s.reshape(-1, 1) + s) / np.pi)
                grad_2d.append(u @ ((z + z.T) * dm + (z - z.T) * dp) @ vh)
                # decide whether the occupied orbitals or virtual orbitals define the rank
                # of the parameter matrix
                if n_occ <= n_virt:
                    grad_2d[spin] += (
                        (np.eye(n_virt) - u @ u.T)
                        @ t0
                        @ ((vh.T * np.sinc(s / np.pi)) @ vh)
                    )
                else:
                    grad_2d[spin] += (
                        ((u * np.sinc(s / np.pi)) @ u.T)
                        @ t0
                        @ (np.eye(n_occ) - vh.T @ vh)
                    )

            # pack back into vectors
            kappa[i, :] = self.pack_vo(kappa_2d)
            local_grad[i, :] = self.pack_vo(local_grad_2d)
            grad[i, :] = self.pack_vo(grad_2d)

        return

    # change of reference
    def transport(self, geodesic: np.ndarray, tangent_vector: np.ndarray):
        # perform pseudo-canonicalization for history if enabled
        if self.pseudo_canonicalization:
            tangent_vector_vo = self.unpack_vo(tangent_vector)
            tangent_vector[:] = self.pack_vo(
                [
                    self.u_virt[0].T @ tangent_vector_vo[0] @ self.u_occ[0],
                    self.u_virt[1].T @ tangent_vector_vo[1] @ self.u_occ[1],
                ]
            )

        return

    # function to get orthogonal MO coefficients and occupations
    def get_orth_mo_coeff(self):
        eigvals_s, eigvecs_s = np.linalg.eigh(self.s1e)
        s_sqrt = eigvecs_s @ np.diag(np.sqrt(eigvals_s)) @ eigvecs_s.T
        s_inv_sqrt = eigvecs_s @ np.diag(1.0 / np.sqrt(eigvals_s)) @ eigvecs_s.T
        dm_orth = s_sqrt @ self.dm @ s_sqrt
        mo_occ, mo_coeff_oao = np.linalg.eigh(dm_orth)
        mo_occ = np.rint(mo_occ)
        mo_coeff = s_inv_sqrt @ mo_coeff_oao
        return mo_occ, mo_coeff


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


class CASSCFOTR(OTR, newton_casscf.CASSCF):

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
    def update_orbs(self, x, grad, h_diag):
        if (
            np.sum(np.abs(x)) > 0.0
            or self.saved_func is None
            or self.saved_grad is None
            or self.saved_h_diag is None
            or self.saved_hess_x is None
        ):
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

            self.saved_func = self.casci(self.mo_coeff, self.ci, eris)[0]
            self.saved_grad, _, self.saved_hess_x, self.saved_h_diag = (
                newton_casscf.gen_g_hop(self, self.mo_coeff, ci, eris)
            )

            self.n_update_orbs += 1

        grad[:] = 2 * self.saved_grad
        h_diag[:] = 2 * self.saved_h_diag

        def hess_x(x: np.ndarray, hx: np.ndarray) -> None:
            hx[:] = 2 * self.saved_hess_x(x)
            self.n_hess_x += 1

        return self.saved_func, hess_x

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

        # initialize settings
        settings = SolverSettings()
        for setting in solver_setting_fields:
            if hasattr(self, setting) and (
                setting != "conv_check"
                or not isinstance(getattr(self, "conv_check", None), bool)
            ):
                setattr(settings, setting, getattr(self, setting))

        # call solver
        solver(self.func, self.update_orbs, self.n_param, settings)
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
