# Copyright (C) 2025- Jonas Greiner
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import annotations

import numpy as np
from pyscf import gto, scf, lo, lib
from pyscf.scf import hf_symm, uhf_symm, ghf_symm
from pyscf.soscf import ciah, newton_ah
from pyscf.x2c import x2c
from pyscf.x2c import newton_ah as x2c_newton_ah
from pyscf.mcscf import (
    casci,
    casci_symm,
    mc1step,
    newton_casscf,
    newton_casscf_symm,
    addons,
)
from pyopentrustregion import SolverSettings, StabilitySettings, solver, stability_check
from pyopentrustregion.python_interface import SolverSettingsC, StabilitySettingsC
from ctypes import Structure
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Tuple, Callable, List, Optional, Union

    PySCFHessX = Callable[[np.ndarray], np.ndarray]
    OTRHessX = Callable[[np.ndarray, np.ndarray], None]


# fields of the settings structures that are not settings of an object: the
# initialization flag and the host context of the C structures, and the logging
# function, which always writes to the PySCF log of the object
non_setting_fields = ("initialized", "context", "logger")


def setting_fields(settings_c: type[Structure]) -> list[str]:
    """
    this function collects the settings a user can assign on a settings object,
    skipping the fields that are not settings and any nested settings structure
    """
    return [
        field[0]
        for field in settings_c._fields_
        if field[0] not in non_setting_fields
        and not (isinstance(field[1], type) and issubclass(field[1], Structure))
    ]


solver_setting_fields = setting_fields(SolverSettingsC)
stability_setting_fields = setting_fields(StabilitySettingsC)

# PySCF uses these field names itself, so their values are inherited from PySCF rather
# than chosen for the solver and must not silently override the solver's own defaults
solver_settings_shadowed_by_pyscf = ("conv_tol",)

# the solver and stability settings share several field names, so a setting meant for
# the solver would otherwise leak into the stability check. These fields are therefore
# only ever taken from a separate, explicitly stability-specific attribute
stability_setting_aliases = {
    "conv_tol": "stability_conv_tol",
    "conv_check": "stability_conv_check",
    "n_random_trial_vectors": "stability_n_random_trial_vectors",
    "n_extra_trial_vectors": "stability_n_extra_trial_vectors",
    "get_extra_trial_vectors": "stability_get_extra_trial_vectors",
    "jacobi_davidson_start": "stability_jacobi_davidson_start",
}

# the error code, named as in OpenTrustRegion, with which the solver reports that it
# reached the maximum number of macro iterations, which pyopentrustregion states only
# in the message of the RuntimeError it raises
error_solver_max_iter = 102


def setting_was_set_by_user(obj: object, name: str) -> bool:
    """
    this function decides whether a setting whose name PySCF also uses was set
    deliberately, that is assigned after the object was constructed rather than
    inherited from PySCF, also when the value assigned equals the inherited one
    """
    return hasattr(obj, name) and name not in getattr(obj, "_inherited_settings", ())


def setting_is_assignable(obj: object, setting: str) -> bool:
    """
    this function decides whether an attribute that happens to carry a settings name
    can be handed on as that setting. PySCF uses conv_check as a boolean flag whereas
    the solver expects a convergence-check function, so a boolean of that name is
    dropped instead of being passed on, where it would raise inside the ctypes callback
    """
    return setting != "conv_check" or not isinstance(getattr(obj, setting, None), bool)


def assign_pyscf_logger(
    obj: object, settings: Union[SolverSettings, StabilitySettings]
) -> None:
    """
    this function hands the messages of the solver or the stability check to the PySCF
    log of an object, so that its output and verbosity control them as they control
    those of PySCF; the library hands on only the messages at its verbosity, which is
    that of the object
    """
    settings.logger = lambda message: lib.logger.log(obj, "%s", message)


def assign_stability_settings(
    obj: object, stability_settings: StabilitySettings
) -> None:
    """
    this function copies the stability check settings from an object onto a stability
    settings object
    """
    for setting in stability_setting_fields:
        if setting in stability_setting_aliases:
            alias = stability_setting_aliases[setting]
            if hasattr(obj, alias):
                setattr(stability_settings, setting, getattr(obj, alias))
        elif hasattr(obj, setting) and setting_is_assignable(obj, setting):
            setattr(stability_settings, setting, getattr(obj, setting))
    assign_pyscf_logger(obj, stability_settings)


def assign_solver_settings(obj: object, settings: SolverSettings) -> None:
    """
    this function copies the solver settings from an object onto a solver settings
    object, along with the settings of the stability check the solver runs itself
    """
    for setting in solver_setting_fields:
        if setting in solver_settings_shadowed_by_pyscf:
            if setting_was_set_by_user(obj, setting):
                setattr(settings, setting, getattr(obj, setting))
        elif hasattr(obj, setting) and setting_is_assignable(obj, setting):
            setattr(settings, setting, getattr(obj, setting))
    assign_pyscf_logger(obj, settings)
    assign_stability_settings(obj, settings.stability_settings)


class OTR:
    _keys = set(
        solver_setting_fields
        + stability_setting_fields
        + list(stability_setting_aliases.values())
        + [
            "_inherited_settings",
            "saved_func",
            "saved_grad",
            "saved_h_diag",
            "saved_hess_x",
            "saved_state",
            "n_update_orbs",
            "n_hess_x",
            "n_param",
            "converged",
        ]
    )

    # supplied by the concrete driver this mixin is combined with
    mo_coeff: np.ndarray

    # set while the kernel runs
    n_param: int
    converged: bool

    # cached result of the last orbital update, reused when the solver asks for the
    # same point again, together with the point it was computed at
    saved_func: float
    saved_grad: np.ndarray
    saved_h_diag: np.ndarray
    saved_hess_x: PySCFHessX
    saved_state: Optional[Tuple[np.ndarray, ...]] = None

    # orbital updates and Hessian linear transformations performed on this object,
    # accumulated over its lifetime
    n_update_orbs = 0
    n_hess_x = 0

    def _orbs_state(self) -> Tuple[np.ndarray, ...]:
        """
        this function returns the arrays that define the point a cached orbital update
        describes, so that the cache can tell whether it still applies
        """
        return (self.mo_coeff,)

    def _cache_is_valid(self) -> bool:
        """
        this function checks whether the cached orbital update still describes the
        current point. A zero step alone is not enough, since a kernel can move the
        orbitals after the solver has returned, for instance by canonicalizing them
        """
        if self.saved_state is None:
            return False
        state = self._orbs_state()
        return len(state) == len(self.saved_state) and all(
            np.array_equal(saved, current)
            for saved, current in zip(self.saved_state, state)
        )

    def _save_orbs_state(self) -> None:
        """
        this function records the point a cached orbital update was computed at, as
        copies, since the orbitals are also updated in place
        """
        self.saved_state = tuple(np.copy(array) for array in self._orbs_state())

    def _snapshot_shadowed_settings(self) -> None:
        """
        this function records the settings whose names PySCF also uses as inherited
        from PySCF until they are assigned; it must run after any copying of another
        object's attributes, which bypasses __setattr__
        """
        self._inherited_settings = frozenset(
            name for name in solver_settings_shadowed_by_pyscf if hasattr(self, name)
        )

    def __setattr__(self, name: str, value: Any) -> None:
        # an assignment makes a setting whose name PySCF also uses one chosen for the
        # solver, also when it assigns the value inherited from PySCF
        inherited = self.__dict__.get("_inherited_settings", frozenset())
        if name in inherited:
            super().__setattr__("_inherited_settings", inherited - {name})
        super().__setattr__(name, value)

    @staticmethod
    def _rotates_too_far(kappa: np.ndarray) -> bool:
        """
        this function checks whether a step rotates any pair of orbitals by more than
        pi / 4, for which the objective functions return an infinite value, so that the
        solver rejects the step
        """
        return bool(np.any(np.abs(kappa) > np.pi / 4))

    def _update_orbs(
        self, kappa: np.ndarray, grad: np.ndarray, h_diag: np.ndarray
    ) -> Tuple[float, OTRHessX]:
        """
        this function is supplied by every concrete driver below; it applies a step,
        writes the gradient and Hessian diagonal in place and returns the objective
        together with the Hessian linear transformation at the new point
        """
        raise NotImplementedError

    def _func(self, kappa: np.ndarray) -> float:
        """
        this function is supplied by every concrete driver below; it returns the
        objective at a step from the current point without applying the step
        """
        raise NotImplementedError

    def _solve(self) -> None:
        """
        this function runs the solver from the current point with the settings of this
        object and records whether it converged; like PySCF's own solvers, it stops at
        the last point without raising once the maximum number of macro iterations is
        reached, which the solver reports itself
        """
        settings = SolverSettings()
        assign_solver_settings(self, settings)
        self.converged = False
        try:
            solver(self._update_orbs, self._func, self.n_param, settings)
            self.converged = True
        except RuntimeError as error:
            if f"(code {error_solver_max_iter})" not in str(error):
                raise

    # stability check function
    def stability_check(self) -> Tuple[bool, np.ndarray]:
        # without parameters, as when symmetry forbids every rotation, there is no
        # direction to lower the objective along
        if self.n_param == 0:
            return True, np.empty(0, dtype=np.float64)

        # get Hessian diagonal and linear transformation at current point
        kappa = np.zeros(self.n_param, dtype=np.float64)
        grad = np.empty(self.n_param, dtype=np.float64)
        h_diag = np.empty(self.n_param, dtype=np.float64)
        _, hess_x = self._update_orbs(kappa, grad, h_diag)

        # the curvature tells a minimum from a saddle point only at a stationary point,
        # so a gradient the solver would not accept as converged is reported
        solver_settings = SolverSettings()
        assign_solver_settings(self, solver_settings)
        grad_rms = np.linalg.norm(grad) / np.sqrt(self.n_param)
        if not grad_rms < solver_settings.conv_tol:
            lib.logger.warn(
                self,
                "The stability check examines a point that is not stationary, since "
                f"its root mean square gradient {grad_rms:.2e} is not below conv_tol = "
                f"{solver_settings.conv_tol:.2e}. The objective decreases along the "
                "gradient there, so the check cannot tell whether the point is a "
                "minimum or a saddle point.",
            )

        # initialize settings
        settings = StabilitySettings()
        assign_stability_settings(self, settings)

        # run stability check
        direction = np.empty(self.n_param, dtype=np.float64)
        stable = stability_check(
            h_diag, hess_x, self.n_param, settings, kappa=direction
        )

        return stable, direction


class LocalizerOTR(OTR):
    """
    this class is the OTR driver for the PySCF orbital localizers. This class must
    precede the pyscf.lo class in the base list of every concrete localizer below so
    that its kernel takes precedence over the PySCF one.
    """

    # supplied by the pyscf.lo class this driver is combined with
    mol: gto.Mole
    init_guess: Optional[str]
    cost_function: Callable[..., float]
    get_init_guess: Callable[..., np.ndarray]
    gen_g_hop: Callable[..., Tuple[np.ndarray, PySCFHessX, np.ndarray]]

    # set while the kernel runs
    norb: int

    # sign that turns the PySCF cost function into a minimization objective; PySCF's
    # gen_g_hop always returns derivatives of the minimization objective, so only the
    # cost function itself needs this factor
    cost_sign = 1.0

    # the boolean stability solver setting shadows the stability() method the PySCF
    # localizers provide; the OTR equivalent is the stability_check() method
    stability = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._snapshot_shadowed_settings()

    # unpack matrix
    def unpack(self, kappa: np.ndarray) -> np.ndarray:
        matrix = np.zeros(2 * (self.norb,), dtype=np.float64)
        idx = np.tril_indices(self.norb, -1)
        matrix[idx] = kappa
        return matrix - matrix.conj().T

    # cost function
    def _func(self, kappa: np.ndarray) -> float:
        if self._rotates_too_far(kappa):
            return np.inf
        u = ciah.expmat(self.unpack(kappa))
        return self.cost_sign * self.cost_function(u)

    # cost function, gradient, Hessian diagonal and Hessian linear transformation
    # function
    def _update_orbs(
        self, kappa: np.ndarray, grad: np.ndarray, h_diag: np.ndarray
    ) -> Tuple[float, OTRHessX]:
        # reuse the cached update when the step is zero and nothing has moved the
        # orbitals since, so the cost function and its derivatives still apply
        if np.any(kappa) or not self._cache_is_valid():
            u = ciah.expmat(self.unpack(kappa))
            self.saved_func = self.cost_function(u)
            (
                self.saved_grad,
                self.saved_hess_x,
                self.saved_h_diag,
            ) = self.gen_g_hop(u)
            self.mo_coeff = self.mo_coeff @ u
            self.n_update_orbs += 1
            self._save_orbs_state()

        grad[:] = self.saved_grad
        h_diag[:] = self.saved_h_diag

        def hess_x(x: np.ndarray, hx: np.ndarray) -> None:
            hx[:] = self.saved_hess_x(x)
            self.n_hess_x += 1

        return self.cost_sign * self.saved_func, hess_x

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
        self._solve()

        return self.mo_coeff


class BoysOTR(LocalizerOTR, lo.Boys):
    pass


class PipekMezeyOTR(LocalizerOTR, lo.PipekMezey):

    # the Pipek-Mezey cost function is maximized
    cost_sign = -1.0


class EdmistonRuedenbergOTR(LocalizerOTR, lo.EdmistonRuedenberg):

    # the Edmiston-Ruedenberg cost function is maximized
    cost_sign = -1.0


# input options of the PySCF Kohn-Sham objects that define the energy. A converted
# object evaluates the energy and the gradient through the PySCF object it was
# converted from but the Hessian itself, so such an option assigned after the
# conversion reaches only the Hessian
ks_energy_options = ("xc", "nlc", "disp", "grids", "nlcgrids", "small_rho_cutoff")


def option_differs(value: Any, wrapped_value: Any) -> bool:
    """
    this function decides whether an option of a converted object differs from that of
    the object it was converted from, comparing names of functionals without regard to
    case, as PySCF parses them
    """
    if isinstance(value, str) and isinstance(wrapped_value, str):
        return value.upper() != wrapped_value.upper()
    return bool(value != wrapped_value)


class SecondOrderOTR(OTR, newton_ah._CIAH_SOSCF):

    # attributes the kernel sets, which PySCF would otherwise report as unknown
    _keys = {"h1e", "s1e", "dm", "vhf", "mask", "mask_symm", "complex_orbitals"}

    # supplied by the PySCF mean-field object this driver wraps
    mo_occ: np.ndarray

    # set while the kernel runs
    dm: np.ndarray
    vhf: np.ndarray
    mask: np.ndarray
    mask_symm: np.ndarray
    complex_orbitals: bool

    def __init__(self, mf: scf.SCF) -> None:
        self.__dict__.update(mf.__dict__)
        self._scf = mf
        self._snapshot_shadowed_settings()

    def _warn_on_differing_energy_options(self) -> None:
        """
        this function warns when an option defining the energy differs from that of the
        PySCF object this object was converted from, which evaluates the energy and the
        gradient, typically because the option was assigned after the conversion
        """
        differing = [
            option
            for option in ks_energy_options
            if hasattr(self._scf, option)
            and option_differs(getattr(self, option), getattr(self._scf, option))
        ]
        if differing:
            names = ", ".join(differing)
            verb = "differs" if len(differing) == 1 else "differ"
            lib.logger.warn(
                self,
                f"{names} of this object {verb} from the PySCF object passed to "
                "mf_to_otr, which evaluates the energy and the gradient, so the value "
                "assigned here reaches only the Hessian. Unless the Hessian is "
                f"approximated on purpose, set {names} before mf_to_otr: otherwise the "
                "solver optimizes the energy for the original value and the stability "
                "check can report instabilities that this energy does not have.",
            )

    def _set_parameters(self) -> None:
        """
        this function sets up the parameters of the current orbitals, the
        occupied-virtual rotations symmetry allows, which for complex orbitals are
        complex, so that their real and imaginary parts are the parameters
        """
        self.complex_orbitals = bool(np.iscomplexobj(self.mo_coeff))
        self.mask, self.mask_symm = self.get_indices()
        self.n_param = (1 + self.complex_orbitals) * int(
            np.count_nonzero(self.mask_symm)
        )

    def _rotation(self, kappa: np.ndarray) -> np.ndarray:
        """
        this function returns the rotation parameters of the allowed occupied-virtual
        pairs from the parameter vector, which holds their real parts followed by
        their imaginary parts for complex orbitals
        """
        if self.complex_orbitals:
            n_pairs = kappa.size // 2
            return kappa[:n_pairs] + 1j * kappa[n_pairs:]
        return kappa

    def _parameters(self, rotation: np.ndarray) -> np.ndarray:
        """
        this function returns the parameter vector of quantities over the allowed
        occupied-virtual pairs, the inverse of _rotation
        """
        if self.complex_orbitals:
            return np.concatenate((rotation.real, rotation.imag))
        return rotation

    def _embed(self, rotation: np.ndarray) -> np.ndarray:
        """
        this function returns the vector over the occupied-virtual pairs of PySCF's
        derivatives that the rotations correspond to
        """
        full = np.zeros(self.mask_symm.shape, dtype=self.saved_grad.dtype)
        full[self.mask_symm] = rotation
        return full

    def _restrict(self, full: np.ndarray) -> np.ndarray:
        """
        this function returns the derivatives with respect to the rotations from those
        over the occupied-virtual pairs of PySCF, the adjoint of _embed
        """
        return full[self.mask_symm]

    def _restrict_diagonal(self, h_diag: np.ndarray) -> np.ndarray:
        """
        this function returns the Hessian diagonal of the rotations from that over the
        occupied-virtual pairs of PySCF
        """
        return h_diag[self.mask_symm]

    def _canonicalize(self, fock: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns the energies and the canonical orbitals within the
        occupied and within the virtual orbitals, which the PySCF object determines
        """
        return self._scf.canonicalize(self.mo_coeff, self.mo_occ, fock)

    # energy function, which limits the modulus of the rotations
    def _func(self, kappa: np.ndarray) -> float:
        rotation = self._rotation(kappa)
        if self._rotates_too_far(rotation):
            return np.inf
        u = self.exp_mat(self.unpack(rotation))
        rot_mo_coeff = self.rotate_mo(self.mo_coeff, u)
        dm = self.make_rdm1(rot_mo_coeff, self.mo_occ)
        vhf = self._scf.get_veff(self._scf.mol, dm)
        return self._scf.energy_tot(dm, self.h1e, vhf)

    # the occupations and the orbitals define the density
    def _orbs_state(self) -> Tuple[np.ndarray, ...]:
        return (self.mo_coeff, self.mo_occ)

    # energy, gradient, Hessian diagonal and Hessian linear transformation function
    def _update_orbs(
        self, kappa: np.ndarray, grad: np.ndarray, h_diag: np.ndarray
    ) -> Tuple[float, OTRHessX]:
        # reuse the cached update when the step is zero and nothing has moved the
        # orbitals since, so the effective potential and Fock matrix need not be rebuilt
        if np.any(kappa) or not self._cache_is_valid():
            u = self.exp_mat(self.unpack(self._rotation(kappa)))
            self.mo_coeff = self.rotate_mo(self.mo_coeff, u)
            dm = self.make_rdm1(self.mo_coeff, self.mo_occ)
            vhf = self._scf.get_veff(self._scf.mol, dm)
            self.dm, self.vhf = dm, vhf
            fock = self.get_fock(self.h1e, self.s1e, vhf, dm)
            (
                self.saved_grad,
                self.saved_hess_x,
                self.saved_h_diag,
            ) = self.gen_g_hop(self.mo_coeff, self.mo_occ, fock)
            self.saved_func = self._scf.energy_tot(dm, self.h1e, vhf)
            self.n_update_orbs += 1
            self._save_orbs_state()

        # PySCF's quantities are half the derivatives with respect to the rotations;
        # for complex orbitals, the real and imaginary parts of the gradient are the
        # derivatives with respect to the real and imaginary parts of the rotations,
        # which share the diagonal of the Hessian
        grad[:] = 2 * self._parameters(self._restrict(self.saved_grad))
        h_diag_rotations = self._restrict_diagonal(self.saved_h_diag)
        h_diag[:] = 2 * np.tile(h_diag_rotations, 1 + self.complex_orbitals)

        def hess_x_symm(x: np.ndarray, hx: np.ndarray) -> None:
            x_full = self._embed(self._rotation(x))
            hx[:] = 2 * self._parameters(self._restrict(self.saved_hess_x(x_full)))
            self.n_hess_x += 1

        return self.saved_func, hess_x_symm

    # kernel function
    def kernel(
        self,
        mo_coeff: Optional[np.ndarray] = None,
        mo_occ: Optional[np.ndarray] = None,
        dm0: Optional[np.ndarray] = None,
    ) -> float:
        if dm0 is not None:
            if isinstance(dm0, str):
                lib.logger.debug(
                    self,
                    f"OpenTrustRegion solver reads density matrix from chkfile {dm0}",
                )
                dm0 = self.from_chk(dm0)

        elif mo_coeff is not None and mo_occ is None:
            lib.logger.warn(
                self,
                "Newton solver expects mo_coeff with mo_occ as initial guess but "
                "mo_occ is not found in the arguments.",
            )
            lib.logger.warn(self, "The given argument is treated as density matrix.")
            dm0 = mo_coeff
            mo_coeff = mo_occ = None

        else:
            if mo_coeff is None:
                mo_coeff = self.mo_coeff
            if mo_occ is None:
                mo_occ = self.mo_occ

        self.build(self.mol)

        # the energy and the gradient are evaluated with the options of the wrapped
        # object, whereas the Hessian is evaluated with those of this object
        self._warn_on_differing_energy_options()

        mol = self._scf.mol

        # call self._scf.get_hcore, self._scf.get_ovlp because they might be overloaded
        self.h1e = self._scf.get_hcore(mol)
        self.s1e = self._scf.get_ovlp(mol)

        # orthogonalization of the overlap, which removes the linear dependencies of
        # the basis from the orbitals of the initial guess
        x_orth = self._scf.check_linear_dependency(self.s1e)

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
            dm = dm0
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
            mo_energy, mo_coeff = self.eig(fock, self.s1e, x=x_orth)
            mo_occ = self.get_occ(mo_energy, mo_coeff)
            dm = self.make_rdm1(mo_coeff, mo_occ)
            vhf = self._scf.get_veff(mol, dm, dm_last=self.dm, vhf_last=self.vhf)
            self.dm, self.vhf = dm, vhf

        self.mo_coeff, self.mo_occ = np.asarray(mo_coeff), mo_occ

        # fix phases of MO coefficients to improve deterministic behavior
        self.fix_phase()

        # get the parameters, the occupied-virtual rotations, which are complex for
        # complex orbitals
        self._set_parameters()

        # call solver, unless symmetry forbids every rotation, which leaves the
        # orbitals stationary already
        if self.n_param > 0:
            self._solve()
        else:
            self.converged = True

        # get canonical orbitals
        self.e_tot = self._func(np.zeros(self.n_param, dtype=np.float64))
        dm = self.make_rdm1(self.mo_coeff, self.mo_occ)
        vhf = self._scf.get_veff(self._scf.mol, dm)
        self.dm, self.vhf = dm, vhf
        fock = self.get_fock(self.h1e, self.s1e, vhf, dm, level_shift_factor=0)
        self.mo_energy, self.mo_coeff = self._canonicalize(fock)

        self._finalize()

        # the parameters are set up anew for the orbitals the kernel returns, which a
        # symmetry-adapted object sorts by their energies across the irreducible
        # representations when it is finalized
        self._set_parameters()

        return self.e_tot

    # stability check function, which examines the Hessian evaluated with the options
    # of this object
    def stability_check(self) -> Tuple[bool, np.ndarray]:
        self._warn_on_differing_energy_options()

        # the one-electron Hamiltonian and the overlap are evaluated as in the kernel,
        # so that the check also examines orbitals assigned to an object whose kernel
        # never ran or ran for another molecule
        mol = self._scf.mol
        self.h1e = self._scf.get_hcore(mol)
        self.s1e = self._scf.get_ovlp(mol)

        # the parameters are the occupied-virtual rotations of the current orbitals,
        # which canonicalization reorders and which orbitals or occupations assigned
        # after the kernel replace
        self._set_parameters()

        return super().stability_check()


def unit_phases(values: np.ndarray) -> np.ndarray:
    """
    this function returns the phases of values, their signs for real values, with that
    of a vanishing value taken as one
    """
    phases = np.ones_like(values)
    nonzero = values != 0
    phases[nonzero] = values[nonzero] / np.abs(values[nonzero])
    return phases


class RHFOTR(SecondOrderOTR, newton_ah._SecondOrderRHF):

    # fix phases of MO coefficients to improve deterministic behavior
    def fix_phase(self) -> None:
        abs_mo_coeff = np.abs(self.mo_coeff)
        max_mo_coeff = np.max(abs_mo_coeff, axis=0)
        mask = np.isclose(abs_mo_coeff, max_mo_coeff, atol=1e-12)
        idx = np.argmax(mask, axis=0)
        cols = np.arange(self.mo_coeff.shape[1])
        vals = self.mo_coeff[idx, cols]
        self.mo_coeff *= unit_phases(vals).conj()[np.newaxis, :]

    # get indices of all mixed occupation combinations
    def get_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        occidxa = self.mo_occ > 0
        occidxb = self.mo_occ == 2
        viridxa = ~occidxa
        viridxb = ~occidxb
        mask = (viridxa[:, None] & occidxa) | (viridxb[:, None] & occidxb)

        mask_symm = mask[mask]
        if self._scf.mol.symmetry:
            # the orbitals are labeled as PySCF's Newton solver labels them, which also
            # works for an object that is not symmetry-adapted, as one built before the
            # symmetry of its molecule was switched on
            orbsym = hf_symm.get_orbsym(self._scf.mol, self.mo_coeff)
            sym_allow = orbsym[:, None] == orbsym
            mask_symm = sym_allow[mask]
            mask[mask] = mask_symm

        return mask, mask_symm

    # function to compute exponential of anti-symmetric matrix
    def exp_mat(self, matrix: np.ndarray) -> np.ndarray:
        return ciah.expmat(matrix)

    # unpack the rotations into an anti-Hermitian matrix over the orbitals
    def unpack(self, rotation: np.ndarray) -> np.ndarray:
        matrix = np.zeros(2 * (self.mo_coeff.shape[1],), dtype=rotation.dtype)
        matrix[self.mask] = rotation
        return matrix - matrix.conj().T


# PySCF's ROHF Hessian lacks the terms proportional to the gradient, which vanish at a
# stationary point, so that it is approximate away from one, which only slows the
# convergence; PySCF releases up to 2.14 also lack orbital-rotation terms that persist
# at a stationary point (pyscf#3448), so that the stability check of an ROHF point,
# like PySCF's own, is approximate with those releases
class ROHFOTR(SecondOrderOTR, newton_ah._SecondOrderROHF):

    fix_phase = RHFOTR.fix_phase
    get_indices = RHFOTR.get_indices
    exp_mat = RHFOTR.exp_mat
    unpack = RHFOTR.unpack


class UHFOTR(SecondOrderOTR, newton_ah._SecondOrderUHF):

    # fix phases of MO coefficients to improve deterministic behavior
    def fix_phase(self) -> None:
        abs_mo_coeff = np.abs(self.mo_coeff)
        max_mo_coeff = np.max(abs_mo_coeff, axis=1)
        for i in range(2):
            mask = np.isclose(abs_mo_coeff[i], max_mo_coeff[i], atol=1e-12)
            idx = np.argmax(mask, axis=0)
            cols = np.arange(self.mo_coeff.shape[2])
            vals = self.mo_coeff[i, idx, cols]
            self.mo_coeff[i] *= unit_phases(vals).conj()[np.newaxis, :]

    # get indices of all mixed occupation combinations
    def get_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        occidxa = self.mo_occ[0] == 1
        occidxb = self.mo_occ[1] == 1
        viridxa = ~occidxa
        viridxb = ~occidxb
        mask = np.stack((viridxa[:, None] & occidxa, viridxb[:, None] & occidxb))

        mask_symm = mask[mask]
        if self._scf.mol.symmetry:
            # labeled as the restricted orbitals are
            orbsyma, orbsymb = uhf_symm.get_orbsym(self._scf.mol, self.mo_coeff)
            sym_allowa = orbsyma[:, None] == orbsyma
            sym_allowb = orbsymb[:, None] == orbsymb
            sym_allow = np.stack((sym_allowa, sym_allowb))
            mask_symm = sym_allow[mask]
            mask[mask] = mask_symm

        return mask, mask_symm

    # function to compute exponential of anti-symmetric matrix
    def exp_mat(self, matrix: List[np.ndarray]) -> List[np.ndarray]:
        return [ciah.expmat(matrix[0]), ciah.expmat(matrix[1])]

    # unpack the rotations into an anti-Hermitian matrix over the orbitals of each spin
    def unpack(self, rotation: np.ndarray) -> List[np.ndarray]:
        n_param_a = np.count_nonzero(self.mask[0])
        matrices = [
            np.zeros(2 * (mo_coeff.shape[1],), dtype=rotation.dtype)
            for mo_coeff in self.mo_coeff
        ]
        matrices[0][self.mask[0]] = rotation[:n_param_a]
        matrices[1][self.mask[1]] = rotation[n_param_a:]
        return [matrix - matrix.conj().T for matrix in matrices]


class GHFOTR(SecondOrderOTR, newton_ah._SecondOrderGHF):

    fix_phase = RHFOTR.fix_phase
    exp_mat = RHFOTR.exp_mat

    # get indices of all occupied-virtual combinations of the spin orbitals
    def get_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        occidx = self.mo_occ > 0
        viridx = ~occidx
        mask = viridx[:, None] & occidx

        mask_symm = mask[mask]
        if self._scf.mol.symmetry:
            # labeled as PySCF's Newton solver labels the spin orbitals
            orbsym = ghf_symm.get_orbsym(self._scf.mol, self.mo_coeff)
            sym_allow = orbsym[:, None] == orbsym
            mask_symm = sym_allow[mask]
            mask[mask] = mask_symm

        return mask, mask_symm

    # unpack the rotations into an anti-Hermitian matrix over the spin orbitals, of
    # which there are twice as many as atomic orbitals
    def unpack(self, rotation: np.ndarray) -> np.ndarray:
        matrix = np.zeros(2 * (self.mo_coeff.shape[1],), dtype=rotation.dtype)
        matrix[self.mask] = rotation
        return matrix - matrix.conj().T


# Kramers-restricted X2C keeps every odd spinor the time reverse of the even one before
# it, which the antiunitary time reversal maintains for rotations that turn the even
# spinors by a matrix and their partners by its complex conjugate. For every pair of a
# virtual and an occupied Kramers pair, the rotations are A between the even spinors
# and C of the even occupied spinor into the odd virtual one, whose time reverse turns
# the odd occupied spinor into the even virtual one by -conj(C); PySCF's own
# second-order solver turns both spinors of a pair alike and omits C, which keeps the
# pairs only for real rotations
class X2CRHFOTR(SecondOrderOTR, x2c_newton_ah.SecondOrderX2CRHF):

    exp_mat = RHFOTR.exp_mat

    # fix phases of MO coefficients to improve deterministic behavior, giving the
    # partner of every even spinor the conjugate phase, which keeps it its time reverse
    def fix_phase(self) -> None:
        even = self.mo_coeff[:, ::2]
        abs_even = np.abs(even)
        mask = np.isclose(abs_even, np.max(abs_even, axis=0), atol=1e-12)
        idx = np.argmax(mask, axis=0)
        phases = unit_phases(even[idx, np.arange(even.shape[1])])
        self.mo_coeff[:, ::2] *= phases.conj()[np.newaxis, :]
        self.mo_coeff[:, 1::2] *= phases[np.newaxis, :]

    # get indices of all combinations of a virtual and an occupied Kramers pair, each
    # of which has the rotations A and C
    def get_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        occ_pairs = self.mo_occ[::2] > 0
        mask = ~occ_pairs[:, None] & occ_pairs
        return mask, np.ones(2 * int(np.count_nonzero(mask)), dtype=bool)

    def _vo_block(self, rotation: np.ndarray) -> np.ndarray:
        """
        this function returns the virtual-occupied block of the rotation generator over
        the spinors from the rotations A followed by the rotations C
        """
        n_occ = int(np.count_nonzero(self.mo_occ > 0))
        n_vir = self.mo_occ.size - n_occ
        n_rotations = rotation.size // 2
        a = rotation[:n_rotations].reshape(n_vir // 2, n_occ // 2)
        c = rotation[n_rotations:].reshape(n_vir // 2, n_occ // 2)
        block = np.empty((n_vir, n_occ), dtype=rotation.dtype)
        block[::2, ::2] = a
        block[1::2, 1::2] = a.conj()
        block[1::2, ::2] = c
        block[::2, 1::2] = -c.conj()
        return block

    # unpack the rotations into an anti-Hermitian matrix over the spinors
    def unpack(self, rotation: np.ndarray) -> np.ndarray:
        occidx = self.mo_occ > 0
        matrix = np.zeros(2 * (self.mo_occ.size,), dtype=rotation.dtype)
        matrix[np.ix_(~occidx, occidx)] = self._vo_block(rotation)
        return matrix - matrix.conj().T

    def _embed(self, rotation: np.ndarray) -> np.ndarray:
        return self._vo_block(rotation).ravel()

    def _restrict(self, full: np.ndarray) -> np.ndarray:
        n_occ = int(np.count_nonzero(self.mo_occ > 0))
        block = full.reshape(-1, n_occ)
        a = block[::2, ::2] + block[1::2, 1::2].conj()
        c = block[1::2, ::2] - block[::2, 1::2].conj()
        return np.concatenate((a.ravel(), c.ravel()))

    def _restrict_diagonal(self, h_diag: np.ndarray) -> np.ndarray:
        n_occ = int(np.count_nonzero(self.mo_occ > 0))
        block = h_diag.reshape(-1, n_occ)
        a = block[::2, ::2] + block[1::2, 1::2]
        c = block[1::2, ::2] + block[::2, 1::2]
        return np.concatenate((a.ravel(), c.ravel()))

    # canonical orbitals within the occupied and within the virtual Kramers pairs from
    # the quaternion eigensolver of the PySCF object, which returns them as pairs,
    # since PySCF's generic canonicalization mixes the degenerate partners of a pair
    def _canonicalize(self, fock: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mo_energy = np.empty(self.mo_occ.size)
        mo_coeff = np.empty_like(self.mo_coeff)
        for idx in (self.mo_occ > 0, self.mo_occ == 0):
            orbs = self.mo_coeff[:, idx]
            # the eigensolver takes the even spinors before their partners
            basis = np.hstack((orbs[:, ::2], orbs[:, 1::2]))
            mo_energy[idx], mo_coeff[:, idx] = self._scf._eigh(fock, self.s1e, x=basis)
        return mo_energy, mo_coeff


class X2CUHFOTR(SecondOrderOTR, x2c_newton_ah.SecondOrderX2CUHF):

    fix_phase = RHFOTR.fix_phase
    exp_mat = RHFOTR.exp_mat
    unpack = GHFOTR.unpack

    # get indices of all occupied-virtual combinations of the spinors, which
    # point-group symmetry can not restrict, since PySCF has no symmetry labels for
    # spinors
    def get_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        occidx = self.mo_occ > 0
        viridx = ~occidx
        mask = viridx[:, None] & occidx
        return mask, mask[mask]


def mf_to_otr(mf: scf.hf.SCF) -> SecondOrderOTR:

    if isinstance(mf, SecondOrderOTR):
        return mf

    assert isinstance(mf, scf.hf.SCF)

    # the spinor classes of X2C carry the names of the nonrelativistic classes, so they
    # are told apart by their type first, and an ROHF class also carries the name of
    # RHF, from which it derives; the four-component classes and those with k-points
    # are not supported
    cls: Optional[type[SecondOrderOTR]] = None
    if isinstance(mf, x2c.RHF):
        cls = X2CRHFOTR
    elif isinstance(mf, x2c.UHF):
        cls = X2CUHFOTR
    elif mf.istype("RHF"):
        cls = ROHFOTR if mf.istype("ROHF") else RHFOTR
    elif mf.istype("UHF"):
        cls = UHFOTR
    elif mf.istype("GHF"):
        cls = GHFOTR
    if cls is None:
        raise TypeError(
            "mf_to_otr supports RHF, ROHF, UHF and GHF and their Kohn-Sham "
            "counterparts, also at the Gamma point of periodic systems, and spinor "
            f"X2C, but not {type(mf).__module__}.{type(mf).__name__}."
        )

    mf = lib.set_class(cls(mf), (cls, mf.__class__))
    if hasattr(mf, "stability") and callable(mf.stability):
        mf.stability = None

    mf._snapshot_shadowed_settings()

    return mf


class CASSCFOTR(OTR, newton_casscf.CASSCF):

    # attributes the kernel sets, which PySCF would otherwise report as unknown
    _keys = {"n_param_orb", "n_param_ci"}

    # set while the kernel runs
    ci: Union[np.ndarray, List[np.ndarray]]
    e_tot: Optional[float]
    e_cas: Optional[float]
    n_param_orb: int
    n_param_ci: int

    def __init__(
        self,
        mf_or_mol: Union[gto.Mole, scf.RHF],
        ncas: int,
        nelecas: Union[int, Tuple[int, int]],
        ncore: Optional[int] = None,
        frozen: Optional[Union[int, List[int]]] = None,
    ) -> None:
        casci.CASBase.__init__(self, mf_or_mol, ncas, nelecas, ncore)
        self.frozen = frozen

        self.e_tot = None
        self.mo_coeff = self._scf.mo_coeff
        self.mo_energy = self._scf.mo_energy
        self.converged = False
        self._max_stepsize = None
        self._snapshot_shadowed_settings()

    @staticmethod
    def ci_vectors(ci: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
        """
        this function returns the CI vectors as a list
        """
        return [ci] if isinstance(ci, np.ndarray) else ci

    def _state_groups(
        self, ci: Union[np.ndarray, List[np.ndarray]]
    ) -> List[Tuple[Any, Union[int, Tuple[int, int]], np.ndarray]]:
        """
        this function returns, for every CI solver, the solver, its numbers of active
        electrons and the CI vectors of its states as the rows of a matrix; in a state
        average over several CI solvers, each solves for its own states, whose CI
        vectors belong to the space of that solver, independently of the others
        """
        vectors = [np.ravel(c) for c in self.ci_vectors(ci)]
        if not isinstance(self.fcisolver, addons.StateAverageMixFCISolver):
            return [(self.fcisolver, self.nelecas, np.array(vectors))]

        groups, start = [], 0
        for solver in self.fcisolver.fcisolvers:
            nelec = self.fcisolver._get_nelec(solver, self.nelecas)
            groups.append(
                (solver, nelec, np.array(vectors[start : start + solver.nroots]))
            )
            start += solver.nroots
        return groups

    def _ci_mask(self) -> np.ndarray:
        """
        this function returns which CI coefficients of all states are parameters, for a
        CI solver with point-group symmetry those of the determinants of the symmetry
        of its states, since the others vanish and its Hamiltonian does not act on them
        """
        masks = []
        for solver, _, states in self._state_groups(self.ci):
            allowed = getattr(solver, "sym_allowed_idx", None)
            if allowed is None:
                mask = np.ones(states.shape, dtype=bool)
            else:
                mask = np.zeros(states.shape, dtype=bool)
                mask[:, np.hstack(allowed)] = True
            masks.append(mask.ravel())
        return np.concatenate(masks)

    def _embed(self, x: np.ndarray) -> np.ndarray:
        """
        this function returns the vector over the orbital rotations and the CI
        coefficients of all states, as gen_g_hop orders them, that a vector over the
        parameters corresponds to
        """
        ci_mask = self._ci_mask()
        full = np.zeros(self.n_param_orb + ci_mask.size)
        full[: self.n_param_orb] = x[: self.n_param_orb]
        full[self.n_param_orb :][ci_mask] = x[self.n_param_orb :]
        return full

    def _restrict(self, full: np.ndarray) -> np.ndarray:
        """
        this function returns the vector over the parameters from one over the orbital
        rotations and the CI coefficients of all states, the adjoint of _embed
        """
        ci = full[self.n_param_orb :][self._ci_mask()]
        return np.concatenate((full[: self.n_param_orb], ci))

    def _orthogonal_ci(self, x: np.ndarray) -> np.ndarray:
        """
        this function removes the components along the states of the same CI solver
        from the CI part of a vector over the orbital rotations and the CI coefficients
        of all states; for every state, these only rescale it or mix it with the other
        states, which the orthonormalization of the displaced states and their
        diagonalization within the space they span undo, so that they are redundant
        directions, along which gen_g_hop's derivatives do not describe the energy
        """
        parts, start = [x[: self.n_param_orb]], self.n_param_orb
        for *_, states in self._state_groups(self.ci):
            x_ci = x[start : start + states.size].reshape(states.shape)
            parts.append((x_ci - (x_ci @ states.T) @ states).ravel())
            start += states.size
        return np.concatenate(parts)

    def displaced_ci(self, x: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
        """
        this function applies the CI part of a step to the current CI vectors, giving
        the CI vectors the energy and its derivatives are both evaluated at; the
        displaced states of every CI solver are orthonormalized symmetrically, which
        keeps them from collapsing onto one another, as normalizing each of them alone
        would not, and for steps orthogonal to these states agrees to second order with
        rotating the states into the steps, so that the energy agrees with the
        derivatives
        """
        x = self._orthogonal_ci(self._embed(x))
        displaced, start = [], self.n_param_orb
        for *_, states in self._state_groups(self.ci):
            moved = states + x[start : start + states.size].reshape(states.shape)
            overlap_values, overlap_vectors = np.linalg.eigh(moved @ moved.T)
            displaced.extend(
                (overlap_vectors / np.sqrt(overlap_values)) @ overlap_vectors.T @ moved
            )
            start += states.size
        return displaced[0] if isinstance(self.ci, np.ndarray) else displaced

    def eigenstates(
        self, mo_coeff: np.ndarray, ci: Union[np.ndarray, List[np.ndarray]], eris: Any
    ) -> Tuple[float, List[np.ndarray]]:
        """
        this function returns the energy of given CI vectors in given orbitals and the
        eigenstates of the active space Hamiltonian within the space the CI vectors of
        every CI solver span, in increasing energy for every solver; the energy is that
        of these eigenstates, weighted the way gen_g_hop weights its derivatives, whose
        gradient at the eigenstates is the derivative of this energy, and which equals
        the weighted energy of the CI vectors themselves for a single state of every
        solver or equal weights
        """
        # the active space Hamiltonian, reusing the integrals already transformed for
        # these orbitals
        fcasci = mc1step._fake_h_for_fast_casci(self, mo_coeff, eris)
        h1eff, energy_core = fcasci.get_h1eff()
        h2eff = fcasci.get_h2eff()

        # the Hamiltonian within the space the CI vectors of every solver span
        energies: List[float] = []
        eigenstates: List[np.ndarray] = []
        for solver, nelec, states in self._state_groups(ci):
            op = solver.absorb_h1e(h1eff, h2eff, self.ncas, nelec, 0.5)
            h_states = np.array(
                [np.ravel(solver.contract_2e(op, c, self.ncas, nelec)) for c in states]
            )
            group_energies, rotation = np.linalg.eigh(states @ h_states.T)
            energies.extend(group_energies)
            eigenstates.extend(rotation.T @ states)

        energy = energy_core + float(np.dot(getattr(self, "weights", [1.0]), energies))
        return energy, eigenstates

    # energy function, which limits only the orbital rotations, since the CI part of a
    # step is a displacement of the CI vectors rather than a rotation angle
    def _func(self, x: np.ndarray) -> float:
        if self._rotates_too_far(x[: self.n_param_orb]):
            return np.inf
        u = ciah.expmat(self.unpack_uniq_var(x[: self.n_param_orb]))

        rot_mo_coeff = self.rotate_mo(self.mo_coeff, u)
        eris = self.ao2mo(rot_mo_coeff)

        return self.eigenstates(rot_mo_coeff, self.displaced_ci(x), eris)[0]

    def _orbs_state(self) -> Tuple[np.ndarray, ...]:
        return (self.mo_coeff, *self.ci_vectors(self.ci))

    # energy, gradient, Hessian diagonal and Hessian linear transformation function
    def _update_orbs(
        self, x: np.ndarray, grad: np.ndarray, h_diag: np.ndarray
    ) -> Tuple[float, OTRHessX]:
        # reuse the cached update when the step is zero and nothing has moved the
        # orbitals or CI vectors since, so the integrals need not be transformed again
        if np.any(x) or not self._cache_is_valid():
            u = ciah.expmat(self.unpack_uniq_var(x[: self.n_param_orb]))
            self.mo_coeff = self.rotate_mo(self.mo_coeff, u)
            eris = self.ao2mo(self.mo_coeff)

            # the states move on as the eigenstates within the space the displaced CI
            # vectors span, at which gen_g_hop's derivatives are those of the energy
            self.saved_func, states = self.eigenstates(
                self.mo_coeff, self.displaced_ci(x), eris
            )
            self.ci = states[0] if isinstance(self.ci, np.ndarray) else states

            (
                self.saved_grad,
                _,
                self.saved_hess_x,
                self.saved_h_diag,
            ) = newton_casscf.gen_g_hop(self, self.mo_coeff, self.ci, eris)
            self.n_update_orbs += 1
            self._save_orbs_state()

        # gen_g_hop extends over all CI coefficients and takes the CI part of the
        # Hessian linear transformation of every state orthogonal only to that state,
        # whereas the steps are orthogonal to all states of its CI solver; its CI
        # gradient is orthogonal to all of them already, since the states are
        # eigenstates within the space they span
        grad[:] = self._restrict(self.saved_grad)
        h_diag[:] = self._restrict(self.saved_h_diag)

        def hess_x(x: np.ndarray, hx: np.ndarray) -> None:
            x_full = self._orthogonal_ci(self._embed(x))
            hx[:] = self._restrict(self._orthogonal_ci(self.saved_hess_x(x_full)))
            self.n_hess_x += 1

        return self.saved_func, hess_x

    def kernel(
        self,
        mo_coeff: Optional[np.ndarray] = None,
        ci0: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        callback: Optional[Callable[..., None]] = None,
    ) -> Tuple[
        Optional[float],
        Optional[float],
        Union[np.ndarray, List[np.ndarray]],
        np.ndarray,
        Optional[np.ndarray],
    ]:
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if callback is None:
            callback = self.callback

        if ci0 is None:
            ci0 = self.ci

        # a symmetry-adapted object labels the orbitals by symmetry first, as PySCF's
        # kernel does, since its orbital rotations and the CI solver need the labels
        if isinstance(self, newton_casscf_symm.CASSCF):
            mo_coeff = self.mo_coeff = casci_symm.label_symmetry_(self, mo_coeff)

        # initial guess
        eris = self.ao2mo(mo_coeff)
        self.e_tot, self.e_cas, fcivec = self.casci(mo_coeff, ci0, eris)
        if isinstance(fcivec, np.ndarray):
            self.ci = fcivec.ravel()
        else:
            self.ci = [c.ravel() for c in fcivec]

        # number of unique orbital rotation parameters
        self.n_param_orb = int(
            np.count_nonzero(
                self.uniq_var_indices(
                    self.mo_coeff.shape[1], self.ncore, self.ncas, self.frozen
                )
            )
        )

        # number of CI parameters
        self.n_param_ci = int(np.count_nonzero(self._ci_mask()))

        # number of parameters
        self.n_param = self.n_param_orb + self.n_param_ci

        # call solver
        self._solve()
        eris = self.ao2mo(self.mo_coeff)
        self.e_tot, self.e_cas, fcivec = self.casci(self.mo_coeff, self.ci, eris)
        if isinstance(fcivec, np.ndarray):
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

        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy


def casscf_to_otr(casscf: newton_casscf.CASSCF) -> CASSCFOTR:
    if isinstance(casscf, CASSCFOTR):
        return casscf

    if not isinstance(casscf, newton_casscf.CASSCF):
        casscf = casscf.newton()

    # the class of the PySCF object is mixed in as in mf_to_otr, e.g. for symmetry,
    # without a state average, which is applied again below so that it comes first
    pyscf_class = (
        casscf._base_class
        if isinstance(casscf, addons.StateAverageMCSCFSolver)
        else casscf.__class__
    )
    casscf_otr = lib.set_class(
        CASSCFOTR(casscf._scf, casscf.ncas, casscf.nelecas), (CASSCFOTR, pyscf_class)
    )
    casscf_otr.__dict__.update(casscf.__dict__)
    casscf_otr._snapshot_shadowed_settings()

    if isinstance(casscf, addons.StateAverageMCSCFSolver):
        wfnsym = getattr(casscf, "wfnsym", None)
        casscf_otr = casscf_otr.state_average_(casscf.weights, wfnsym)

    return casscf_otr
