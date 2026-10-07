# Copyright (C) 2025- Jonas Greiner
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import annotations

import weakref
import numpy as np
from dataclasses import dataclass
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
from pyopentrustregion.extensions.mo import MOSettings, mo_factory, mo_deconstructor
from pyopentrustregion.extensions.oao import OAOSettings, oao_factory, oao_deconstructor
from pyopentrustregion.extensions.arh import (
    ARHSettings,
    arh_factory_mo,
    arh_factory_oao,
    arh_deconstructor,
)
from pyopentrustregion.extensions.s_gek import (
    SGEKSettings,
    update_orbs_s_gek_factory,
    update_orbs_s_gek_deconstructor,
)
from pyopentrustregion.extensions.mo.python_interface import MOSettingsC
from pyopentrustregion.extensions.oao.python_interface import OAOSettingsC
from pyopentrustregion.extensions.arh.python_interface import ARHSettingsC
from pyopentrustregion.extensions.quasi_newton.python_interface import QNSettingsC
from pyopentrustregion.extensions.s_gek.python_interface import SGEKSettingsC
from ctypes import Structure
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Tuple, Callable, Optional, Union


def setting_fields(settings_c: type[Structure]) -> list[str]:
    """
    this function collects the settings a user can assign on a settings object,
    skipping the initialization flag and any nested settings structure
    """
    return [
        field[0]
        for field in settings_c._fields_
        if field[0] != "initialized"
        and not (isinstance(field[1], type) and issubclass(field[1], Structure))
    ]


@dataclass(frozen=True, eq=False)
class LastResult:
    """
    this class holds the result computed at the last point, together with copies of the
    arrays that define that point, so that the result can be reused while the point is
    unchanged
    """

    key: Tuple[np.ndarray, ...] = ()
    value: Any = None

    @classmethod
    def of(cls, key: Tuple[np.ndarray, ...], value: Any) -> LastResult:
        return cls(tuple(np.array(array, copy=True) for array in key), value)

    def matches(self, key: Tuple[np.ndarray, ...]) -> bool:
        return (
            len(self.key) > 0
            and len(self.key) == len(key)
            and all(
                np.array_equal(saved, current) for saved, current in zip(self.key, key)
            )
        )


solver_setting_fields = setting_fields(SolverSettingsC)
stability_setting_fields = setting_fields(StabilitySettingsC)
mo_setting_fields = setting_fields(MOSettingsC)
oao_setting_fields = setting_fields(OAOSettingsC)
arh_setting_fields = setting_fields(ARHSettingsC)
qn_setting_fields = setting_fields(QNSettingsC)
s_gek_setting_fields = setting_fields(SGEKSettingsC)

# PySCF uses these field names itself, so their values are inherited from PySCF rather
# than chosen for the solver and must not silently override the solver's own defaults
solver_settings_shadowed_by_pyscf = ("conv_tol",)

# the solver and stability settings share several field names, so a setting meant for
# the solver would otherwise leak into the stability check. These fields are therefore
# only ever taken from a separate, explicitly stability-specific attribute
stability_setting_aliases = {
    "conv_tol": "stability_conv_tol",
    "n_random_trial_vectors": "stability_n_random_trial_vectors",
    "n_extra_trial_vectors": "stability_n_extra_trial_vectors",
    "get_extra_trial_vectors": "stability_get_extra_trial_vectors",
    "jacobi_davidson_start": "stability_jacobi_davidson_start",
}


def setting_was_set_by_user(obj, name) -> bool:
    """
    this function decides whether a setting whose name PySCF also uses was set
    deliberately, by comparing it against the value recorded when the object was
    constructed
    """
    if not hasattr(obj, name):
        return False
    inherited = getattr(obj, "_inherited_settings", {})
    if name not in inherited:
        return True
    return getattr(obj, name) != inherited[name]


def assign_stability_settings(obj, stability_settings) -> None:
    """
    this function copies the stability check settings from an object onto a stability
    settings object
    """
    for setting in stability_setting_fields:
        if setting in stability_setting_aliases:
            alias = stability_setting_aliases[setting]
            if hasattr(obj, alias):
                setattr(stability_settings, setting, getattr(obj, alias))
        elif hasattr(obj, setting) and (
            setting != "conv_check"
            or not isinstance(getattr(obj, "conv_check", None), bool)
        ):
            setattr(stability_settings, setting, getattr(obj, setting))


def assign_solver_settings(obj, settings) -> None:
    """
    this function copies the solver settings from an object onto a solver settings
    object, along with the settings of the stability check the solver runs itself
    """
    for setting in solver_setting_fields:
        if setting in solver_settings_shadowed_by_pyscf:
            if setting_was_set_by_user(obj, setting):
                setattr(settings, setting, getattr(obj, setting))
        elif hasattr(obj, setting) and (
            setting != "conv_check"
            or not isinstance(getattr(obj, "conv_check", None), bool)
        ):
            setattr(settings, setting, getattr(obj, setting))
    assign_stability_settings(obj, settings.stability_settings)


class OTR:
    _keys = set(
        solver_setting_fields
        + stability_setting_fields
        + list(stability_setting_aliases.values())
        + [
            "_inherited_settings",
            "_last_update",
            "_updated",
            "raw_hess_x",
            "n_update_orbs",
            "n_hess_x",
            "n_hess_x_stability",
        ]
    )

    # the cost function, gradient, Hessian diagonal and Hessian linear transformation
    # of the last orbital update, keyed by the point they were computed at and reused
    # when the solver asks for the same point again
    _last_update = LastResult()

    # whether the current kernel has performed an orbital update yet
    _updated = False

    # orbital updates and Hessian linear transformations performed on this object,
    # accumulated over its lifetime
    n_update_orbs = 0
    n_hess_x = 0
    n_hess_x_stability = 0

    def _moves(self, kappa: np.ndarray) -> bool:
        """
        this function returns whether an orbital update moves the optimization (when it
        takes a step, and when it is the first of a kernel, which starts the
        optimization at its initial point) and counts it if so
        """
        moves = bool(np.any(kappa)) or not self._updated
        self._updated = True
        if moves:
            self.n_update_orbs += 1

        return moves

    def _reset(self) -> None:
        """
        this function discards everything a previous kernel computed
        """
        self._updated = False
        self._last_update = LastResult()

    # Hessian linear transformation of the last orbital update
    raw_hess_x = None

    # Hessian linear transformation performed for the solver
    def count_hess_x(self, x: np.ndarray, hx: np.ndarray) -> None:
        self.raw_hess_x(x, hx)
        self.n_hess_x += 1

    # Hessian linear transformation performed for a stability check
    def count_stability_hess_x(self, x: np.ndarray, hx: np.ndarray) -> None:
        self.raw_hess_x(x, hx)
        self.n_hess_x_stability += 1

    def _orbs_state(self) -> Tuple[np.ndarray, ...]:
        """
        this function returns the arrays that define the point an orbital update
        describes
        """
        return (self.mo_coeff,)

    def _snapshot_shadowed_settings(self) -> None:
        """
        this function records the settings whose names PySCF also uses, so that a later
        assignment can be told apart from a value inherited from PySCF; it must run
        after any copying of another object's attributes
        """
        self._inherited_settings = {
            name: getattr(self, name)
            for name in solver_settings_shadowed_by_pyscf
            if hasattr(self, name)
        }

    # stability check function
    def stability_check(self) -> Tuple[bool, np.ndarray]:
        # get Hessian diagonal and linear transformation at current point
        kappa = np.zeros(self.n_param, dtype=np.float64)
        grad = np.empty(self.n_param, dtype=np.float64)
        h_diag = np.empty(self.n_param, dtype=np.float64)
        self.update_orbs(kappa, grad, h_diag)

        # initialize settings
        settings = StabilitySettings()
        assign_stability_settings(self, settings)

        # run stability check
        direction = np.empty(self.n_param, dtype=np.float64)
        stable, _ = stability_check(
            h_diag, self.count_stability_hess_x, self.n_param, settings, direction
        )

        return stable, direction


class LocalizerOTR(OTR):
    """
    this class is the OTR driver for the PySCF orbital localizers. This class must
    precede the pyscf.lo class in the base list of every concrete localizer below so
    that its kernel takes precedence over the PySCF one.
    """

    norb: int
    mo_coeff: np.ndarray

    # sign that turns the PySCF cost function into a minimization objective; PySCF's
    # gen_g_hop always returns derivatives of the minimization objective, so only the
    # cost function itself needs this factor
    cost_sign = 1.0

    # the boolean stability solver setting shadows the stability() method the PySCF
    # localizers provide; the OTR equivalent is the stability_check() method
    stability = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._snapshot_shadowed_settings()

    # unpack matrix
    def unpack(self, kappa: np.ndarray) -> np.ndarray:
        matrix = np.zeros(2 * (self.norb,), dtype=np.float64)
        idx = np.tril_indices(self.norb, -1)
        matrix[idx] = kappa
        return matrix - matrix.conj().T

    # cost function
    def func(self, kappa: np.ndarray) -> float:
        u = ciah.expmat(self.unpack(kappa))
        return self.cost_sign * self.cost_function(u)

    # cost function, gradient, Hessian diagonal and Hessian linear transformation
    # function
    def update_orbs(
        self, kappa: np.ndarray, grad: np.ndarray, h_diag: np.ndarray
    ) -> Tuple[float, Callable[[np.ndarray], np.ndarray]]:
        # reuse the cached update when the step is zero and nothing has moved the
        # orbitals since, so the cost function and its derivatives still apply
        self._moves(kappa)
        if np.any(kappa) or not self._last_update.matches(self._orbs_state()):
            u = ciah.expmat(self.unpack(kappa))
            func = self.cost_function(u)
            new_grad, hess_x, new_h_diag = self.gen_g_hop(u)
            self.mo_coeff = self.mo_coeff @ u
            self._last_update = LastResult.of(
                self._orbs_state(), (func, new_grad, new_h_diag, hess_x)
            )
        func, new_grad, new_h_diag, hess_x = self._last_update.value

        grad[:] = new_grad
        h_diag[:] = new_h_diag

        def raw_hess_x(x, hx):
            hx[:] = hess_x(x)

        self.raw_hess_x = raw_hess_x

        return self.cost_sign * func, self.count_hess_x

    # kernel function
    def kernel(self, mo_coeff: Optional[np.ndarray] = None) -> np.ndarray:
        # discard anything computed by a previous kernel
        self._reset()

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
        assign_solver_settings(self, settings)

        # call solver
        solver(self.func, self.update_orbs, self.n_param, settings)

        return self.mo_coeff


class BoysOTR(LocalizerOTR, lo.Boys):
    pass


class PipekMezeyOTR(LocalizerOTR, lo.PipekMezey):

    # the Pipek-Mezey cost function is maximized
    cost_sign = -1.0


class EdmistonRuedenbergOTR(LocalizerOTR, lo.EdmistonRuedenberg):

    # the Edmiston-Ruedenberg cost function is maximized
    cost_sign = -1.0


if hasattr(lo, "FourthMoment"):

    class FourthMomentOTR(LocalizerOTR, lo.FourthMoment):
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

    _keys = set(
        mo_setting_fields
        + oao_setting_fields
        + arh_setting_fields
        + qn_setting_fields
        + s_gek_setting_fields
        + [
            "_adopted_routines",
            "_last_potential",
            "_adopted_potential",
            "_canonical_rotation",
            "raw_approx_update_orbs",
            "raw_approx_hess_x",
            "n_jk_update",
            "n_jk_energy",
            "n_jk_response",
            "n_jk_response_stability",
            "oao",
            "arh",
            "s_gek",
        ]
    )

    # two-electron builds the optimization requires for effective potentials and for
    # the responses of exact Hessian linear transformations
    n_jk_update = 0
    n_jk_energy = 0
    n_jk_response = 0
    n_jk_response_stability = 0

    @property
    def n_jk(self) -> int:
        return self.n_jk_update + self.n_jk_energy + self.n_jk_response

    # the last built and the last adopted effective potential, keyed by their densities
    _last_potential = LastResult()
    _adopted_potential = None

    def _build_potential(self, dm: np.ndarray) -> np.ndarray:
        """
        this function builds the effective potential at a density
        """
        return self._scf.get_veff(self._scf.mol, dm)

    def _potential(self, dm: np.ndarray) -> np.ndarray:
        """
        this function returns the effective potential at a density, reusing the last
        one whenever the density is unchanged
        """
        if not self._last_potential.matches((dm,)):
            self._last_potential = LastResult.of((dm,), self._build_potential(dm))
            self.n_jk_energy += 1

        return self._last_potential.value

    def _count_update(self) -> None:
        """
        this function recounts the last build of the effective potential as one at a
        density the optimization moves to rather than a pure energy evaluation
        """
        if (
            self._last_potential.key
            and self._adopted_potential is not self._last_potential
        ):
            self._adopted_potential = self._last_potential
            self.n_jk_energy -= 1
            self.n_jk_update += 1

    # routines a kernel can replace or take over from the extension factories, and
    # those the last kernel did, which refer to the extension objects of that kernel
    # that the next kernel frees
    factory_routines = (
        "func",
        "update_orbs",
        "approx_update_orbs",
        "raw_approx_update_orbs",
        "project",
        "precond",
        "stability_get_extra_trial_vectors",
    )
    _adopted_routines: Tuple[str, ...] = ()

    # occupied-occupied and virtual-virtual blocks of the rotation of every spin from
    # the orbitals the MO objects hold to the final ones, with the masks of the
    # parameters of both, if the last kernel used the MO extension
    _canonical_rotation: Optional[list] = None

    def _reset(self) -> None:
        super()._reset()
        self._last_potential = LastResult()
        self._adopted_potential = None
        self._canonical_rotation = None
        for name in self._adopted_routines:
            self.__dict__.pop(name, None)
        self._adopted_routines = ()

    # raw and counted orbital updates and (approximate) Hessian linear transformations
    raw_approx_update_orbs = None
    raw_approx_hess_x = None

    def count_approx_hess_x(self, x: np.ndarray, hx: np.ndarray) -> None:
        self.raw_approx_hess_x(x, hx)
        self.n_hess_x += 1

    def count_approx_update_orbs(self, approx_update_orbs: Callable) -> Callable[
        [np.ndarray, np.ndarray, np.ndarray],
        Tuple[float, Callable[[np.ndarray, np.ndarray], None]],
    ]:
        def update_orbs(
            kappa: np.ndarray, grad: np.ndarray, h_diag: np.ndarray
        ) -> Tuple[float, Callable[[np.ndarray, np.ndarray], None]]:
            func, self.raw_approx_hess_x = approx_update_orbs(kappa, grad, h_diag)
            return func, self.count_approx_hess_x

        return update_orbs

    def count_stability_hess_x(self, x: np.ndarray, hx: np.ndarray) -> None:
        n_jk_response = self.n_jk_response
        super().count_stability_hess_x(x, hx)
        self.n_jk_response_stability += self.n_jk_response - n_jk_response

    def approx_stability_hess_x(self, x: np.ndarray, hx: np.ndarray) -> None:
        n_jk_response = self.n_jk_response
        self.raw_approx_update_orbs.hess_x(x, hx)
        self.n_hess_x_stability += 1
        self.n_jk_response_stability += self.n_jk_response - n_jk_response

    def __init__(self, mf: scf.SCF):
        self.__dict__.update(mf.__dict__)
        self._scf = mf
        self._snapshot_shadowed_settings()

    # energy function
    def func(self, kappa: np.ndarray) -> float:
        u = self.exp_mat(self.unpack(kappa))
        rot_mo_coeff = self.rotate_mo(self.mo_coeff, u)
        dm = self.make_rdm1(rot_mo_coeff, self.mo_occ)
        vhf = self._potential(dm)
        return self._scf.energy_tot(dm, self.h1e, vhf)

    def _save_update_orbs(self) -> None:
        """
        this function saves the energy, gradient, Hessian diagonal and Hessian linear
        transformation at the current orbitals, computed from their density matrix,
        effective potential and Fock matrix
        """
        energy = self._scf.energy_tot(self.dm, self.h1e, self.vhf)
        new_grad, hess_x, new_h_diag = self.gen_g_hop(
            self.mo_coeff, self.mo_occ, self.fock
        )
        self._last_update = LastResult.of(
            self._orbs_state(), (energy, new_grad, new_h_diag, hess_x)
        )

    # energy, gradient, Hessian diagonal and Hessian linear transformation function
    def update_orbs(
        self, kappa: np.ndarray, grad: np.ndarray, h_diag: np.ndarray
    ) -> Tuple[float, Callable[[np.ndarray], np.ndarray]]:
        # reuse the cached update when the step is zero and nothing has moved the
        # orbitals since, so the effective potential and Fock matrix need not be rebuilt
        self._moves(kappa)
        if np.any(kappa) or not self._last_update.matches(self._orbs_state()):
            # perform orbital rotation and get new fock matrix
            u = self.exp_mat(self.unpack(kappa))
            self.mo_coeff = self.rotate_mo(self.mo_coeff, u)
            self.dm = self.make_rdm1(self.mo_coeff, self.mo_occ)
            self.vhf = self._potential(self.dm)
            self._count_update()
            self.fock = self.get_fock(self.h1e, self.s1e, self.vhf, self.dm)
            self._save_update_orbs()
        energy, new_grad, new_h_diag, hess_x = self._last_update.value

        grad[:] = 2 * new_grad[self.kappa_mask]
        h_diag[:] = 2 * new_h_diag[self.kappa_mask]

        def raw_hess_x(x: np.ndarray, hx: np.ndarray) -> None:
            x_full = np.zeros_like(self.kappa_mask, dtype=np.float64)
            x_full[self.kappa_mask] = x
            self.n_jk_response += 1
            hx[:] = 2 * hess_x(x_full)[self.kappa_mask]

        self.raw_hess_x = raw_hess_x

        return energy, self.count_hess_x

    # kernel function
    def kernel(
        self,
        mo_coeff: Optional[np.ndarray] = None,
        mo_occ: Optional[np.ndarray] = None,
        dm: Optional[np.ndarray] = None,
    ) -> float:
        # discard anything computed by a previous kernel
        self._reset()

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

        # set default values for OTR extensions, ARH parameterizes the orbitals in the
        # MO basis unless the orthogonal AO basis is requested
        if not hasattr(self, "oao"):
            self.oao = False
        if not hasattr(self, "arh"):
            self.arh = False
        if isinstance(self, ROHFOTR) and (self.oao or self.arh):
            raise NotImplementedError(
                "OAO and ARH are not currently implemented for ROHF."
            )
        if self.oao and self._scf.mol.symmetry:
            raise NotImplementedError(
                "The OAO basis is not currently implemented for symmetry-adapted "
                "orbitals."
            )
        if not hasattr(self, "hess_update_scheme"):
            self.hess_update_scheme = None
        if not hasattr(self, "s_gek"):
            self.s_gek = False

        # exact orbital updates of RHF and UHF in the MO basis come from the MO
        # extension; the approximate Hessians build on them
        mo = not self.oao and not isinstance(self, ROHFOTR)

        # orthogonalize the AO basis for the eigensolver as PySCF's own kernel does,
        # which removes linearly dependent AOs, if this version of PySCF supports it
        eig_kwargs = (
            {"x": self.check_linear_dependency(self.s1e)}
            if hasattr(self, "check_linear_dependency")
            else {}
        )

        # the OAO basis rotates all AOs, so it cannot exclude linearly dependent ones
        if self.oao and "x" in eig_kwargs:
            if eig_kwargs["x"].shape[1] < self.s1e.shape[0]:
                raise NotImplementedError(
                    "The OAO basis is not currently implemented for linearly "
                    "dependent AOs."
                )

        # get initial guess
        if mo_coeff is not None and mo_occ is not None:
            dm = self.make_rdm1(mo_coeff, mo_occ)
            vhf = self._potential(dm)
            self._count_update()
            self.dm, self.vhf = dm, vhf
            fock = self.get_fock(self.h1e, self.s1e, vhf, dm, level_shift_factor=0)
            mo_energy, mo_tmp = self.eig(fock, self.s1e, **eig_kwargs)
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
            vhf = self._potential(dm)
            self._count_update()
            self.dm, self.vhf = dm, vhf
            fock = self.get_fock(self.h1e, self.s1e, vhf, dm, level_shift_factor=0)
            mo_energy, mo_coeff = self.eig(fock, self.s1e, **eig_kwargs)
            mo_occ = self.get_occ(mo_energy, mo_coeff)

            # the MO extension evaluates the density of these orbitals itself
            if not mo:
                dm = self.make_rdm1(mo_coeff, mo_occ)
                vhf = self._potential(dm)
                self._count_update()
                self.dm, self.vhf = dm, vhf

        # share the in-core integrals the wrapped object has just built for the
        # initial potential so that the responses, which go through this object,
        # neither cache them a second time nor, when a second copy does not fit, fall 
        # back to direct J/K builds
        if self.mol is self._scf.mol and not getattr(self, "with_df", None):
            self._eri = self._scf._eri

        self.mo_coeff, self.mo_occ = np.asarray(mo_coeff), mo_occ

        # fix phases of MO coefficients to improve deterministic behavior
        self.fix_phase()

        # the MO extension expects the occupied orbitals of every spin first
        if mo:
            order = np.argsort(np.asarray(self.mo_occ) == 0, axis=-1, kind="stable")
            self.mo_coeff = np.take_along_axis(
                self.mo_coeff, order[..., None, :], axis=-1
            )
            self.mo_occ = np.take_along_axis(np.asarray(self.mo_occ), order, axis=-1)

        # get indices of all mixed occupation combinations
        self.get_indices()

        # irreps of the MOs, whose occupied-virtual rotations between orbitals of the
        # same irrep are the parameters of the MO extension
        orbsym = (
            np.asarray(self.get_orbsym(self.mo_coeff)) if mo and mol.symmetry else None
        )

        # initialize settings
        settings = SolverSettings()
        assign_solver_settings(self, settings)

        # deallocate resources from a previous kernel() call on this instance before
        # registering a finalizer for the current one
        if getattr(self, "_finalizer", None) is not None and self._finalizer.alive:
            self._finalizer()
        self._finalizer = weakref.finalize(
            self,
            self._cleanup,
            self.arh,
            self.hess_update_scheme,
            self.s_gek,
            mo,
            self.oao,
        )

        # number of particles for OAO and ARH
        n_particle = 1 if isinstance(self, RHFOTR) else 2

        # number of parameters
        if self.oao:
            # number of parameters in AO basis
            self.n_param = n_particle * self._scf.mol.nao * (self._scf.mol.nao - 1) // 2
        else:
            self.n_param = np.count_nonzero(self.kappa_mask)

        # turn on automatic stability check for approximate Hessians
        if (not hasattr(self, "stability") or self.stability is None) and (
            self.hess_update_scheme is not None or self.arh or self.s_gek
        ):
            self.stability = settings.stability = True

        # routines before the extensions replace or take over any of them
        routines = {name: self.__dict__.get(name) for name in self.factory_routines}

        # get exact orbital updates from the MO or the OAO extension
        if mo or self.oao:
            if mo:
                mo_settings = MOSettings()
                for setting in mo_setting_fields:
                    if hasattr(self, setting):
                        setattr(mo_settings, setting, getattr(self, setting))
                # run factory, which rotates the MO coefficients in place
                self.func, exact_update_orbs = mo_factory(
                    self.mo_coeff,
                    self.s1e,
                    np.count_nonzero(self.mo_occ, axis=-1),
                    n_particle,
                    self._scf.mol.nao,
                    self.mo_coeff.shape[-1],
                    self.evaluate_dm,
                    settings,
                    mo_settings,
                    orbsym,
                )
            elif self.oao:
                if isinstance(self, RHFOTR):
                    dm_per_spin_ao = self.dm / 2
                else:
                    dm_per_spin_ao = self.dm
                oao_settings = OAOSettings()
                for setting in oao_setting_fields:
                    if hasattr(self, setting):
                        setattr(oao_settings, setting, getattr(self, setting))
                # run factory and set projection
                self.func, exact_update_orbs = oao_factory(
                    dm_per_spin_ao,
                    self.s1e,
                    n_particle,
                    self._scf.mol.nao,
                    self.evaluate_dm,
                    settings,
                    oao_settings,
                )
                self.project = settings.project

            # set preconditioner and extra trial vectors
            self.precond = settings.precond
            self.stability_get_extra_trial_vectors = (
                settings.stability_settings.get_extra_trial_vectors
            )

            def wrapped_update_orbs(kappa, grad, h_diag):
                n_jk = self.n_jk
                func, hess_x = exact_update_orbs(kappa, grad, h_diag)
                if self._moves(kappa) or self.n_jk > n_jk:
                    self._count_update()
                self.raw_hess_x = hess_x

                return func, self.count_hess_x

            self.update_orbs = wrapped_update_orbs

        # define callback functions for approximate Hessians
        if self.arh:
            arh_settings = ARHSettings()
            for setting in arh_setting_fields:
                if hasattr(self, setting):
                    setattr(arh_settings, setting, getattr(self, setting))
            evaluate_dm = (
                self.evaluate_dm_spin
                if isinstance(self, UHFOTR)
                else self.evaluate_dm_nonlinear
            )
            # run factory after the factory of its basis, with which it shares the
            # orbitals it updates in place and the orbital objects whose exact Hessian
            # the stability check uses
            if self.oao:
                self.func, approx_update_orbs = arh_factory_oao(
                    dm_per_spin_ao,
                    self.s1e,
                    n_particle,
                    self._scf.mol.nao,
                    evaluate_dm,
                    settings,
                    arh_settings,
                )
            else:
                self.func, approx_update_orbs = arh_factory_mo(
                    self.mo_coeff,
                    self.s1e,
                    np.count_nonzero(self.mo_occ, axis=-1),
                    n_particle,
                    self._scf.mol.nao,
                    self.mo_coeff.shape[-1],
                    evaluate_dm,
                    settings,
                    arh_settings,
                    orbsym,
                )

            # set the preconditioner, which inverts the approximate Hessian, and extra
            # trial vectors
            self.precond = settings.precond
            self.stability_get_extra_trial_vectors = (
                settings.stability_settings.get_extra_trial_vectors
            )

            def wrapped_approx_update_orbs(
                kappa: np.ndarray, grad: np.ndarray, h_diag: np.ndarray
            ) -> Tuple[float, Callable[[np.ndarray, np.ndarray], None]]:
                n_jk = self.n_jk
                func, hess_x = approx_update_orbs(kappa, grad, h_diag)
                if self._moves(kappa) or self.n_jk > n_jk:
                    self._count_update()
                return func, hess_x

            self.approx_update_orbs = wrapped_approx_update_orbs
            self.raw_approx_update_orbs = approx_update_orbs
        elif self.hess_update_scheme is not None:
            qn_settings = QNSettings()
            for setting in qn_setting_fields:
                if hasattr(self, setting):
                    setattr(qn_settings, setting, getattr(self, setting))
            self.approx_update_orbs = self.raw_approx_update_orbs = (
                update_orbs_qn_factory(
                    self.update_orbs,
                    self.transport,
                    self.init_hess,
                    self.n_param,
                    qn_settings,
                )
            )
        elif self.s_gek:
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
            for setting in s_gek_setting_fields:
                if hasattr(self, setting):
                    setattr(s_gek_settings, setting, getattr(self, setting))
            self.approx_update_orbs = self.raw_approx_update_orbs = (
                update_orbs_s_gek_factory(
                    self.update_orbs,
                    self.change_reference,
                    self.n_param,
                    s_gek_settings,
                )
            )

        # record the routines the extensions replaced or took over, so that the next
        # kernel drops them
        self._adopted_routines = tuple(
            name
            for name in self.factory_routines
            if self.__dict__.get(name) is not routines[name]
        )

        # set (approximate) Hessian linear transformation for stability check
        if self.stability:
            # get linear transformation function which is continuously updated as the
            # orbitals move
            kappa = np.zeros(self.n_param, dtype=np.float64)
            grad = np.empty(self.n_param, dtype=np.float64)
            h_diag = np.empty(self.n_param, dtype=np.float64)
            self.update_orbs(kappa, grad, h_diag)

            settings.stability_hess_x = self.count_stability_hess_x
            if hasattr(self, "approx_update_orbs"):
                settings.stability_settings.approx_hess_x = self.approx_stability_hess_x

        # call solver
        solver(
            self.func,
            (
                self.update_orbs
                if not hasattr(self, "approx_update_orbs")
                else self.count_approx_update_orbs(self.approx_update_orbs)
            ),
            self.n_param,
            settings,
        )

        # the MO objects keep the orbitals they hold now
        mo_coeff_evaluated = self.mo_coeff

        # get canonical orbitals
        if self.oao:
            self.dm = 2 * dm_per_spin_ao if isinstance(self, RHFOTR) else dm_per_spin_ao
        self.converged = True
        if self.oao:
            self.mo_occ, self.mo_coeff = self.get_orth_mo_coeff()
        self.dm = self.make_rdm1(self.mo_coeff, self.mo_occ)
        vhf = self._potential(self.dm)
        self._count_update()
        self.e_tot = self._scf.energy_tot(self.dm, self.h1e, vhf)
        fock = self._scf.get_fock(self.h1e, self.s1e, vhf, self.dm)
        # the orbitals the MO extension optimized keep the irreps they started with,
        # which canonicalization is told so that it diagonalizes the Fock matrix per
        # irrep and every orbital keeps its irrep, which the masks of the parameters
        # refer to
        if mo and orbsym is not None:
            self.mo_coeff = lib.tag_array(self.mo_coeff, orbsym=orbsym)
        self.mo_energy, self.mo_coeff = self._scf.canonicalize(
            self.mo_coeff, self.mo_occ, fock
        )
        masks_evaluated = self._vo_masks() if mo else []
        self._finalize()

        # finalizing may reorder the orbitals within the occupied and the virtual
        # ones, which with symmetry changes the irrep at every orbital index, so the
        # masks of the parameters are determined anew for the final orbitals
        if not self.oao:
            self.get_indices()

        # get the rotation from the orbitals of the MO objects to the final ones with
        # the masks of the parameters of either, with which a stability check after
        # the kernel starts from the rotations it would start from at the final
        # orbitals and returns its direction for these
        if mo:
            rotation = (
                np.swapaxes(mo_coeff_evaluated, -1, -2) @ self.s1e @ self.mo_coeff
            )
            occ_idx = self.occ_idx if isinstance(self, UHFOTR) else [self.occ_idx]
            virt_idx = self.virt_idx if isinstance(self, UHFOTR) else [self.virt_idx]
            rotation = rotation if isinstance(self, UHFOTR) else rotation[None]
            self._canonical_rotation = [
                (u[np.ix_(occ, occ)], u[np.ix_(virt, virt)], mask_evaluated, mask)
                for u, occ, virt, mask_evaluated, mask in zip(
                    rotation, occ_idx, virt_idx, masks_evaluated, self._vo_masks()
                )
            ]
            self.stability_get_extra_trial_vectors = self._canonical_trial_vectors

        return self.e_tot

    def _vo_masks(self) -> list:
        """
        this function returns the virtual-occupied blocks of the masks of the
        parameters of every spin, whose entries in row-major order are the parameters
        """
        if isinstance(self, UHFOTR):
            return list(self.rot_matrix_mask_vo)

        return [self.rot_matrix_mask_vo]

    def _rotate_canonical(self, vector: np.ndarray, to_canonical: bool) -> np.ndarray:
        """
        this function rotates an orbital rotation from the orbitals the MO objects hold
        to the final ones or back
        """
        offset, rotated = 0, []
        for u_occ, u_virt, mask_evaluated, mask in self._canonical_rotation:
            source, target = (
                (mask_evaluated, mask) if to_canonical else (mask, mask_evaluated)
            )
            x_vo = np.zeros(source.shape)
            x_vo[source] = vector[offset : offset + np.count_nonzero(source)]
            offset += int(np.count_nonzero(source))
            x_vo = u_virt.T @ x_vo @ u_occ if to_canonical else u_virt @ x_vo @ u_occ.T
            rotated.append(x_vo[target])

        return np.concatenate(rotated)

    def _canonical_trial_vectors(self, trial_vectors: np.ndarray) -> None:
        """
        this function returns the rotations along the lowest eigenvalue pairs of the
        static part of the Hessian at the final, canonical orbitals, the differences of
        the orbital energies of the parameters, in increasing order and rotated to the
        orbitals the MO objects hold
        """
        mo_energy = self.mo_energy if isinstance(self, UHFOTR) else [self.mo_energy]
        occ_idx = self.occ_idx if isinstance(self, UHFOTR) else [self.occ_idx]
        virt_idx = self.virt_idx if isinstance(self, UHFOTR) else [self.virt_idx]
        pairs = np.concatenate(
            [
                (energy[virt][:, None] - energy[occ])[mask]
                for energy, occ, virt, (*_, mask) in zip(
                    mo_energy, occ_idx, virt_idx, self._canonical_rotation
                )
            ]
        )
        for vector, idx in zip(trial_vectors, np.argsort(pairs, kind="stable")):
            unit_vector = np.zeros(pairs.size)
            unit_vector[idx] = 1.0
            vector[:] = self._rotate_canonical(unit_vector, False)

    def stability_check(self) -> Tuple[bool, np.ndarray]:
        stable, direction = super().stability_check()

        # the MO objects hold the orbitals before they were canonicalized and
        # reordered, so the direction is rotated to the final orbitals
        if self._canonical_rotation is not None:
            direction = self._rotate_canonical(direction, True)

        return stable, direction

    @staticmethod
    def _cleanup(arh, hess_update_scheme, s_gek, mo, oao):
        # call deconstructors
        if arh:
            arh_deconstructor()
        else:
            if hess_update_scheme is not None:
                update_orbs_qn_deconstructor()
            elif s_gek:
                update_orbs_s_gek_deconstructor()
            if mo:
                mo_deconstructor()
            elif oao:
                oao_deconstructor()


class RHFOTR(SecondOrderOTR, newton_ah._SecondOrderRHF):

    # fix phases of MO coefficients to improve deterministic behavior
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
        matrix = np.zeros(self.rot_matrix_mask.shape, dtype=np.float64)
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

    # evaluate quantities at a density matrix
    def evaluate_dm(
        self, dm: np.ndarray, fock: Optional[np.ndarray], get_response: bool
    ) -> Tuple[float, Optional[Callable[[np.ndarray, np.ndarray], None]]]:
        # anything the caller did not ask for is not built
        veff = self._potential(2.0 * dm)
        if fock is not None:
            self.fock = self.get_fock(self.h1e, self.s1e, veff, 2.0 * dm)
            fock[:, :] = self.fock
        response = (
            self.get_response_factory(
                self.gen_response(dm0=2.0 * dm, hermi=1, singlet=None)
            )
            if get_response
            else None
        )

        return self._scf.energy_tot(2.0 * dm, self.h1e, veff), response

    # evaluate quantities at a density matrix with a separate non-linear (XC)
    # contribution to the effective potential
    def evaluate_dm_nonlinear(
        self,
        dm: np.ndarray,
        fock: Optional[np.ndarray],
        v_nonlinear: Optional[np.ndarray],
    ) -> float:
        # anything the caller did not ask for is not built
        veff = self._potential(2.0 * dm)

        if v_nonlinear is not None:
            if isinstance(self, scf.hf.KohnShamDFT):
                v_linear = np.asarray(veff.vj)
                if getattr(veff, "vk", None) is not None:
                    v_linear = v_linear - 0.5 * veff.vk
                v_nonlinear[:, :] = np.asarray(veff) - v_linear
            else:
                # no non-linear contribution for Hartree-Fock
                v_nonlinear[:, :] = 0.0

        if fock is not None:
            fock[:, :] = self.get_fock(self.h1e, self.s1e, veff, 2.0 * dm)

        return self._scf.energy_tot(2.0 * dm, self.h1e, veff)

    def get_response_factory(
        self, gen_response: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray, np.ndarray], None]:
        def get_response(dm: np.ndarray, response: np.ndarray):
            self.n_jk_response += 1
            response[:, :] = gen_response(2.0 * dm)

        return get_response

    # transport tangent vector along geodesic
    def transport(self, geodesic: np.ndarray, tangent_vector: np.ndarray):
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

    exp_mat = RHFOTR.exp_mat
    unpack = RHFOTR.unpack

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

    # fix phases of MO coefficients to improve deterministic behavior
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

    def _build_potential(self, dm: np.ndarray) -> np.ndarray:
        """
        this function builds the effective potential at a density; unrestricted
        Hartree-Fock builds it from the Coulomb and exchange matrices of each spin and
        tags these onto the result, since its same-spin and opposite-spin potentials
        need them separately
        """
        if isinstance(self, scf.hf.KohnShamDFT):
            return super()._build_potential(dm)
        vj, vk = self._scf.get_jk(self._scf.mol, dm)

        return lib.tag_array(vj[0] + vj[1] - vk, vj=vj, vk=vk)

    def _spin_coulomb(self, dm: np.ndarray, veff: np.ndarray) -> np.ndarray:
        """
        this function returns the Coulomb potential of each spin separately; get_veff
        tags the potential of the total density onto its result, so only one spin has
        to be built and the other follows from it by linearity, which halves the
        Coulomb work this costs; the build is not counted: the potential at this
        density has been counted already, and this second pass exists only because
        get_veff hands back the total rather than the per-spin potentials
        """
        vj_alpha = self._scf.get_j(self._scf.mol, dm[0])

        return np.stack((vj_alpha, np.asarray(veff.vj) - vj_alpha))

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
            np.zeros(self.rot_matrix_mask.shape[1:], dtype=np.float64),
            np.zeros(self.rot_matrix_mask.shape[1:], dtype=np.float64),
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

    # evaluate quantities at a density matrix
    def evaluate_dm(
        self, dm: np.ndarray, fock: Optional[np.ndarray], get_response: bool
    ) -> Tuple[float, Optional[Callable[[np.ndarray, np.ndarray], None]]]:
        # anything the caller did not ask for is not built
        vhf = self._potential(dm)
        if fock is not None:
            fock[:, :, :] = self._scf.get_fock(self.h1e, self.s1e, vhf, dm)
        response = (
            self.get_response_factory(self.gen_response(dm0=dm, hermi=1))
            if get_response
            else None
        )

        return self._scf.energy_tot(dm, self.h1e, vhf), response

    # evaluate quantities at a density matrix with separate same-spin, opposite-spin
    # and non-linear (XC) contributions to the effective potential
    def evaluate_dm_spin(
        self,
        dm: np.ndarray,
        fock: Optional[np.ndarray],
        v_same_spin: Optional[np.ndarray],
        v_opposite_spin: Optional[np.ndarray],
        v_nonlinear: Optional[np.ndarray],
    ) -> float:
        # the channel potentials share one build, so they are all built as soon as any
        # of them is wanted; anything else the caller did not ask for is skipped
        channels = (
            v_same_spin is not None
            or v_opposite_spin is not None
            or v_nonlinear is not None
        )

        if isinstance(self, scf.hf.KohnShamDFT):
            # get KS effective potential
            veff = self._potential(dm)

            if channels:
                # get the coulomb potential of each spin
                vj = self._spin_coulomb(dm, veff)

                # linear part of the potential
                same_spin = np.array(vj)
                if veff.vk is not None:
                    same_spin -= veff.vk
                opposite_spin = vj[::-1]

                if v_same_spin is not None:
                    v_same_spin[:, :, :] = same_spin
                if v_opposite_spin is not None:
                    v_opposite_spin[:, :, :] = opposite_spin

                # non-linear (exchange-correlation) part
                if v_nonlinear is not None:
                    v_nonlinear[:, :, :] = np.asarray(veff) - same_spin - opposite_spin

        else:
            # get the mean-field potential, tagged with the Coulomb and exchange
            # matrices of each spin
            veff = self._potential(dm)
            vj, vk = veff.vj, veff.vk

            # construct same-spin and opposite-spin potentials, the entire
            # Hartree-Fock potential is linear in the density matrix
            if v_same_spin is not None:
                v_same_spin[:, :, :] = vj - vk
            if v_opposite_spin is not None:
                v_opposite_spin[:, :, :] = vj[::-1]

            # no non-linear contribution for Hartree-Fock
            if v_nonlinear is not None:
                v_nonlinear[:, :, :] = 0.0

        # construct Fock matrix
        if fock is not None:
            fock[:, :, :] = self._scf.get_fock(self.h1e, self.s1e, veff, dm)

        return self._scf.energy_tot(dm, self.h1e, veff)

    def get_response_factory(
        self, gen_response: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray, np.ndarray], None]:
        def get_response(dm: np.ndarray, response: np.ndarray):
            self.n_jk_response += 1
            response[:, :, :] = gen_response(dm)

        return get_response

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

    mf._snapshot_shadowed_settings()

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
        self._snapshot_shadowed_settings()

    def _orbs_state(self) -> Tuple[np.ndarray, ...]:
        if self.fcisolver.nroots == 1:
            return (self.mo_coeff, self.ci)
        return (self.mo_coeff, *self.ci)

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
        # reuse the cached update when the step is zero and nothing has moved the
        # orbitals or CI vector since, so the integrals need not be transformed again
        self._moves(x)
        if np.any(x) or not self._last_update.matches(self._orbs_state()):
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

            energy = self.casci(self.mo_coeff, self.ci, eris)[0]
            new_grad, _, hess_x, new_h_diag = newton_casscf.gen_g_hop(
                self, self.mo_coeff, ci, eris
            )
            self._last_update = LastResult.of(
                self._orbs_state(), (energy, new_grad, new_h_diag, hess_x)
            )
        energy, new_grad, new_h_diag, hess_x = self._last_update.value

        grad[:] = 2 * new_grad
        h_diag[:] = 2 * new_h_diag

        def raw_hess_x(x: np.ndarray, hx: np.ndarray) -> None:
            hx[:] = 2 * hess_x(x)

        self.raw_hess_x = raw_hess_x

        return energy, self.count_hess_x

    def kernel(self, mo_coeff=None, ci0=None, callback=None):
        # discard anything computed by a previous kernel
        self._reset()
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
        assign_solver_settings(self, settings)

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
    casscf_otr._snapshot_shadowed_settings()

    if isinstance(casscf, addons.StateAverageMCSCFSolver):
        wfnsym = getattr(casscf, "wfnsym", None)
        casscf_otr = casscf_otr.state_average_(casscf.weights, wfnsym)
        raise RuntimeError(
            "State-averaged CASSCF calculations currently do not work with the "
            "OpenTrustRegion solver"
        )

    return casscf_otr
