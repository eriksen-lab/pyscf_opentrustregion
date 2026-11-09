# Copyright (C) 2025- Jonas Greiner
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import annotations

import functools
import inspect
import weakref
import numpy as np
from dataclasses import dataclass
import scipy as sc
from pyscf import gto, scf, lo, lib
from pyscf.pbc import gto as pbcgto
from pyscf.tools import mo_mapping
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
from pyopentrustregion import (
    SolverSettings,
    StabilitySettings,
    SolverState,
    solver,
    stability_check,
)
from pyopentrustregion.python_interface import (
    Settings,
    SolverSettingsC,
    StabilitySettingsC,
)
from pyopentrustregion.extensions.rotation_qn import (
    RotationQNSettings,
    rotation_qn_factory,
)
from pyopentrustregion.extensions.mo import MOSettings, mo_factory
from pyopentrustregion.extensions.oao import OAOSettings, oao_factory
from pyopentrustregion.extensions.density_qn import (
    DensityQNSettings,
    density_qn_factory,
)
from pyopentrustregion.extensions.s_gek import SGEKSettings, s_gek_factory
from pyopentrustregion.extensions.mo.python_interface import MOSettingsC
from pyopentrustregion.extensions.oao.python_interface import OAOSettingsC
from pyopentrustregion.extensions.density_qn.python_interface import DensityQNSettingsC
from pyopentrustregion.extensions.rotation_qn.python_interface import (
    RotationQNSettingsC,
)
from pyopentrustregion.extensions.s_gek.python_interface import SGEKSettingsC
from ctypes import Structure
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Tuple, Callable, List, Optional, Type, Union

    PySCFHessX = Callable[[np.ndarray], np.ndarray]
    OTRHessX = Callable[[np.ndarray, np.ndarray], None]


# fields of the settings structures that are not settings of an object: the
# initialization flag and the host context of the C structures, and the logging
# function, which always writes to the PySCF log of the object
non_setting_fields = ("initialized", "context", "logger")


def setting_fields(settings_c: Type[Structure]) -> List[str]:
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
        """
        this function returns the result computed at a point, keyed by copies of the
        arrays that define the point, so that changing these arrays in place does not
        change the key
        """
        return cls(tuple(np.array(array, copy=True) for array in key), value)

    def matches(self, key: Tuple[np.ndarray, ...]) -> bool:
        """
        this function returns whether a result is held that was computed at the point
        the arrays define
        """
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
density_qn_setting_fields = setting_fields(DensityQNSettingsC)
rotation_qn_setting_fields = setting_fields(RotationQNSettingsC)
s_gek_setting_fields = setting_fields(SGEKSettingsC)

# the quasi-Newton methods of the two extensions, whose names differ, so that the
# method alone selects the extension: the density QN methods update the Hessian in the
# space of the density matrices, the rotation QN methods in that of the orbital
# rotations
density_qn_methods = ("arh", "symm_arh", "ms_psb", "ms_sp", "ms_sr1")
rotation_qn_methods = ("sr1", "bfgs")

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


def weak_method(routine: Callable) -> Callable:
    """
    this function returns a function that calls a bound method through a weak reference
    to its object, and any other routine as it is; a driver hands its methods to the
    settings and the extension objects, which it keeps alive itself, only through such
    references, since the methods would otherwise keep the driver alive in a reference
    cycle, which only the garbage collector frees
    """
    if not inspect.ismethod(routine):
        return routine
    method: Any = weakref.WeakMethod(routine)

    def call(*args: Any) -> Any:
        return method()(*args)

    return call


def assign_pyscf_logger(obj: object, settings: Settings) -> None:
    """
    this function hands the messages of the solver, the stability check or an
    extension to the PySCF log of an object, so that its output and verbosity control
    them as they control those of PySCF; the library hands on only the messages at its
    verbosity, which is that of the object, which the settings refer to weakly, like
    the methods of a driver
    """
    reference = weakref.ref(obj)
    settings.logger = lambda message: lib.logger.log(reference(), "%s", message)


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


def assign_extension_settings(
    obj: object, settings: Settings, fields: List[str]
) -> None:
    """
    this function copies the settings of an extension from an object onto its settings
    object
    """
    for setting in fields:
        if hasattr(obj, setting):
            setattr(settings, setting, getattr(obj, setting))
    assign_pyscf_logger(obj, settings)


# the error code, named as in OpenTrustRegion, with which the solver reports that it
# reached the maximum number of macro iterations, which pyopentrustregion states only
# in the message of the RuntimeError it raises
error_solver_max_iter = 102


class OTR:
    """
    this class is the part common to all OTR drivers: it runs the solver and the
    stability check on the orbital update and the objective of a driver, with the
    settings taken from its attributes
    """

    _keys = set(
        solver_setting_fields
        + stability_setting_fields
        + list(stability_setting_aliases.values())
        + [
            "n_update_orbs",
            "n_hess_x",
            "n_hess_x_stability",
            "n_param",
            "converged",
        ]
    )

    # supplied by the concrete driver this mixin is combined with
    mo_coeff: np.ndarray

    # set while the kernel runs
    n_param: int
    converged: bool

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

    # Hessian linear transformation of the last orbital update
    _raw_hess_x: OTRHessX

    def _count_hess_x(self, x: np.ndarray, hx: np.ndarray) -> None:
        """
        this function performs the Hessian linear transformation of the last orbital
        update for the solver and counts it
        """
        self._raw_hess_x(x, hx)
        self.n_hess_x += 1

    def _count_stability_hess_x(self, x: np.ndarray, hx: np.ndarray) -> None:
        """
        this function performs the Hessian linear transformation of the last orbital
        update for a stability check and counts it
        """
        self._raw_hess_x(x, hx)
        self.n_hess_x_stability += 1

    def _drop_hess_x(self) -> None:
        """
        this function drops the Hessian linear transformations of the last orbital
        updates once the solver or the stability check has returned, since they can
        refer to this object, which would keep them alive in a reference cycle
        """
        self.__dict__.pop("_raw_hess_x", None)
        self.__dict__.pop("_raw_approx_hess_x", None)

    # the boolean stability setting of the solver shadows the stability() method of
    # PySCF's mean-field and localizer classes; the equivalent is stability_check()
    stability = False

    def _record_inherited_settings(self) -> None:
        """
        this function records the settings whose names PySCF also uses as inherited
        from PySCF until they are assigned; it must run after any copying of another
        object's attributes, which bypasses __setattr__
        """
        self._inherited_settings = frozenset(
            name for name in solver_settings_shadowed_by_pyscf if hasattr(self, name)
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """
        this function assigns an attribute; an assignment makes a setting whose name
        PySCF also uses one chosen for the solver, also when it assigns the value
        inherited from PySCF
        """
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

    def _solve(
        self,
        settings: Optional[SolverSettings] = None,
        state: Optional[SolverState] = None,
        update_orbs: Optional[Callable[..., Tuple[float, OTRHessX]]] = None,
    ) -> None:
        """
        this function runs the solver from the current point and records whether it
        converged, with the settings of this object and its orbital update unless the
        kernel built the settings and a state for the extensions and takes the orbital
        update of one; like PySCF's own solvers, it stops at the last point without
        raising once the maximum number of macro iterations is reached, which the
        solver reports itself
        """
        if settings is None:
            settings = SolverSettings()
            assign_solver_settings(self, settings)
        if update_orbs is None:
            update_orbs = self._update_orbs
        self.converged = False
        try:
            solver(update_orbs, self._func, self.n_param, settings, state)
            self.converged = True
        except RuntimeError as error:
            if f"(code {error_solver_max_iter})" not in str(error):
                raise
        finally:
            self._drop_hess_x()

    def _warn_on_callback(self, callback: Optional[Callable[..., None]]) -> None:
        """
        this function warns that a callback for PySCF's kernels is ignored, since it
        reads local variables of those kernels, which the solver does not have
        """
        if callback is not None:
            lib.logger.warn(
                self,
                "The callback is ignored, since it reads local variables of PySCF's "
                "kernel, which OpenTrustRegion does not have.",
            )

    def stability_check(self) -> Tuple[bool, np.ndarray]:
        """
        this function checks whether the current point is a minimum and returns this
        together with the direction along the lowest eigenvector of the Hessian, which
        leads out of a saddle point
        """
        # without parameters, as when symmetry forbids every rotation, there is no
        # direction to lower the objective along
        if self.n_param == 0:
            return True, np.empty(0, dtype=np.float64)

        # get Hessian diagonal and linear transformation at current point
        kappa = np.zeros(self.n_param, dtype=np.float64)
        grad = np.empty(self.n_param, dtype=np.float64)
        h_diag = np.empty(self.n_param, dtype=np.float64)
        self._update_orbs(kappa, grad, h_diag)

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
        try:
            stable = stability_check(
                h_diag,
                self._count_stability_hess_x,
                self.n_param,
                settings,
                kappa=direction,
            )
        finally:
            self._drop_hess_x()

        return stable, direction


class LocalizerOTR(OTR):
    """
    this class is the OTR driver for the PySCF orbital localizers. This class must
    precede the pyscf.lo class in the base list of every concrete localizer below so
    that its kernel takes precedence over the PySCF one.
    """

    # supplied by the pyscf.lo class this driver is combined with
    mol: gto.Mole
    verbose: int
    init_guess: Optional[str]
    pdim: int
    check_sanity: Callable[[], Any]
    unpack_uniq_var: Callable[[np.ndarray], np.ndarray]
    cost_function: Callable[..., float]
    get_init_guess: Callable[..., np.ndarray]
    gen_g_hop: Callable[..., Tuple[np.ndarray, PySCFHessX, np.ndarray]]

    # set while the kernel runs
    norb: int

    # sign that turns the PySCF cost function into a minimization objective; PySCF's
    # gen_g_hop always returns derivatives of the minimization objective, so only the
    # cost function itself needs this factor
    cost_sign = 1.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._record_inherited_settings()

    def _func(self, kappa: np.ndarray) -> float:
        """
        this function returns the cost function, as a minimization objective, at a
        step from the current orbitals
        """
        if self._rotates_too_far(kappa):
            return np.inf
        u = ciah.expmat(self.unpack_uniq_var(kappa))
        return self.cost_sign * self.cost_function(u)

    def _update_orbs(
        self, kappa: np.ndarray, grad: np.ndarray, h_diag: np.ndarray
    ) -> Tuple[float, OTRHessX]:
        """
        this function rotates the orbitals by a step and returns the cost function, its
        gradient, Hessian diagonal and Hessian linear transformation there
        """
        self._moves(kappa)
        u = ciah.expmat(self.unpack_uniq_var(kappa))
        func = self.cost_function(u)
        new_grad, hess_x, new_h_diag = self.gen_g_hop(u)
        self.mo_coeff = self.mo_coeff @ u

        grad[:] = new_grad
        h_diag[:] = new_h_diag

        def raw_hess_x(x: np.ndarray, hx: np.ndarray) -> None:
            hx[:] = hess_x(x)

        self._raw_hess_x = raw_hess_x

        return self.cost_sign * func, self._count_hess_x

    def kernel(
        self,
        mo_coeff: Optional[np.ndarray] = None,
        callback: Optional[Callable[..., None]] = None,
        verbose: Optional[int] = None,
    ) -> np.ndarray:
        """
        this function localizes the orbitals passed in or those of this object and
        returns them, as PySCF's localizer kernel does
        """
        # a verbosity passed to the kernel holds for its run
        if verbose is not None:
            with lib.temporary_env(self, verbose=verbose):
                return self.kernel(mo_coeff, callback)

        # discard anything computed by a previous kernel
        self._reset()

        # set MO coefficients if provided
        if mo_coeff is not None:
            self.mo_coeff = np.asarray(mo_coeff, order="C")

        # ensure MO coefficients are provided
        assert self.mo_coeff is not None

        # number of orbitals
        self.norb = self.mo_coeff.shape[1]

        # number of parameters
        self.n_param = self.pdim

        # a single orbital leaves nothing to optimize
        if self.n_param == 0:
            self.converged = True
            return self.mo_coeff

        # warn about attributes no class registers, such as misspelled settings, as
        # PySCF's kernel does
        if self.verbose >= lib.logger.WARN:
            self.check_sanity()
        self._warn_on_callback(callback)

        # the input orbitals, against which the localized orbitals are sorted
        mo_in = self.mo_coeff

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

        # sort the localized orbitals as PySCF does, so that each is closest to the
        # input orbital of the same index
        u = np.linalg.lstsq(mo_in, self.mo_coeff, rcond=None)[0]
        self.mo_coeff = self.mo_coeff[:, mo_mapping.mo_1to1map(u)]

        return self.mo_coeff


class BoysOTR(LocalizerOTR, lo.Boys):
    pass


class PipekMezeyOTR(LocalizerOTR, lo.PipekMezey):

    # the Pipek-Mezey cost function is maximized
    cost_sign = -1.0


class EdmistonRuedenbergOTR(LocalizerOTR, lo.EdmistonRuedenberg):

    # the Edmiston-Ruedenberg cost function is maximized
    cost_sign = -1.0


# the drivers of the PySCF localizers, matched by their exact class, since a subclass
# such as PipekMezeyComplex can change the cost function
localizer_drivers = {
    lo.Boys: BoysOTR,
    lo.PipekMezey: PipekMezeyOTR,
    lo.EdmistonRuedenberg: EdmistonRuedenbergOTR,
}


def _localizer_to_otr(loc: Any) -> LocalizerOTR:
    """
    this function converts a PySCF localizer into an object of its driver class
    """
    loc_otr = localizer_drivers[type(loc)](loc.mol, loc.mo_coeff)
    loc_otr.__dict__.update(loc.__dict__)
    loc_otr._record_inherited_settings()

    return loc_otr


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
    """
    this class is the OTR driver for the PySCF mean-field objects, which evaluates the
    energy and the gradient through the PySCF object it was converted from and the
    Hessian itself, as PySCF's second-order solver does; the classes below supply the
    rotations of the orbitals of every kind of mean-field object
    """

    # supplied by the PySCF mean-field object this driver wraps
    mo_occ: np.ndarray

    # quasi-Newton method, which selects density QN or rotation QN, if any
    qn_method: Optional[str]

    # set while the kernel runs
    dm: np.ndarray
    vhf: np.ndarray
    complex_orbitals: bool

    _keys = set(
        mo_setting_fields
        + oao_setting_fields
        + density_qn_setting_fields
        + rotation_qn_setting_fields
        + s_gek_setting_fields
        + [
            "n_jk_update",
            "n_jk_energy",
            "n_jk_response",
            "n_jk_response_stability",
            "oao",
            "s_gek",
            "h1e",
            "s1e",
            "dm",
            "vhf",
            "fock",
            "kappa_mask",
            "rot_matrix_mask",
            "rot_matrix_mask_vo",
            "occ_idx",
            "virt_idx",
            "n_occ",
            "n_virt",
            "clos_idx",
            "open_idx",
            "n_clos",
            "n_open",
            "vc_mask",
            "vo_mask",
            "oc_mask",
            "complex_orbitals",
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
        """
        this property returns the number of two-electron builds of every kind,
        including the response builds of the stability check
        """
        return self.n_jk_update + self.n_jk_energy + self.n_jk_response

    # the last built and the last adopted effective potential, keyed by their densities
    _last_potential = LastResult()
    _adopted_potential = None

    def _check_exchange_kernel(self) -> None:
        """
        this function raises if the exact exchange of the functional is not a positive
        semidefinite kernel: PySCF builds the exchange matrix as hyb K + (alpha - hyb)
        K_lr = hyb K_sr + alpha K_lr, and the full, short-range and long-range exchange
        kernels are positive semidefinite, so that this holds exactly when hyb and, for
        range-separated hybrids, alpha are not negative
        """
        if not isinstance(self, scf.hf.KohnShamDFT):
            return
        omega, alpha, hyb = self._numint.rsh_and_hybrid_coeff(
            self.xc, spin=self._scf.mol.spin
        )
        if hyb < 0.0 or (omega != 0.0 and alpha < 0.0):
            raise NotImplementedError(
                f"The functional {self.xc} has a negative fraction of exact exchange "
                f"(short-range {hyb}, long-range {alpha}), which multisecant SR1 does "
                "not support since its exact-exchange kernel is not positive "
                "semidefinite."
            )

    def _build_potential(self, dm: np.ndarray) -> lib.NPArrayWithTag:
        """
        this function builds the effective potential at a density
        """
        return self._scf.get_veff(self._scf.mol, dm)

    def _potential(self, dm: np.ndarray) -> lib.NPArrayWithTag:
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
    # and keep them alive and which the next kernel drops, so that its factories wrap
    # the routines of this object
    _factory_routines = (
        "_func",
        "_update_orbs",
        "_approx_update_orbs",
        "_raw_approx_update_orbs",
        "project",
        "precond",
        "stability_get_extra_trial_vectors",
    )
    _adopted_routines: Tuple[str, ...] = ()

    # occupied-occupied and virtual-virtual blocks of the rotation of every spin from
    # the orbitals the MO objects hold to the final ones, with the masks of the
    # parameters of both, if the last kernel used the MO extension
    _canonical_rotation: Optional[list] = None

    # the orbitals the last kernel left, which the extension objects of the routines it
    # adopted describe, if it adopted any, with a weak reference to the object whose
    # kernel adopted them, which a copy of the object shares the routines with
    _kernel_orbitals = LastResult()

    def _adopt(self, name: str, routine: Callable) -> None:
        """
        this function lets a routine of an extension factory take the place of the
        method of this object of that name, as an entry of the instance that shadows
        the method until the routines the kernel adopted are dropped
        """
        self.__dict__[name] = routine

    def _response_function(
        self, dm0: np.ndarray, scale: float
    ) -> Callable[[np.ndarray, np.ndarray], None]:
        """
        this function returns the response function of the extensions at the density
        matrices of every spin dm0, which scaled are those of the PySCF object, and
        counts its two-electron builds; the extension objects keep it alive, so it
        refers to this object weakly and PySCF's response function, which refers to the
        object it is built on, is built on a view of this object without the routines
        it adopted from the extension factories and the Hessian linear transformations
        of its last orbital updates
        """
        view = lib.view(self, type(self))
        view._drop_hess_x()
        for name in self._factory_routines:
            view.__dict__.pop(name, None)
        gen_response = view.gen_response(dm0=scale * dm0, hermi=1)
        driver: Any = weakref.proxy(self)

        def get_response(dm: np.ndarray, response: np.ndarray) -> None:
            driver.n_jk_response += 1
            response[...] = gen_response(scale * dm)

        return get_response

    def _drop_adopted_routines(self) -> None:
        """
        this function drops the routines the last kernel adopted from the extension
        factories, so that the routines of this object apply again
        """
        self._canonical_rotation = None
        self._kernel_orbitals = LastResult()
        for name in self._adopted_routines:
            self.__dict__.pop(name, None)
        self._adopted_routines = ()

    def _reset(self) -> None:
        """
        this function discards everything a previous kernel computed, including the
        effective potentials and the routines it adopted from the extension factories
        """
        super()._reset()
        self._last_potential = LastResult()
        self._adopted_potential = None
        self._drop_adopted_routines()

    def _follow_orbitals(self) -> None:
        """
        this function keeps the routines the last kernel adopted while the orbitals and
        occupations are those the kernel left, which the extension objects of these
        routines keep, and otherwise, or on a copy of the object whose kernel adopted
        them, whose extension objects refer to that object, drops them and determines
        the parameters of the current orbitals and occupations, on which the routines of
        this object work
        """
        if (
            self._adopted_routines
            and self._kernel_orbitals.matches(self._orbs_state())
            and self._kernel_orbitals.value() is self
        ):
            return

        self._drop_adopted_routines()
        self._set_parameters()

    # raw and counted orbital updates and (approximate) Hessian linear transformations;
    # the raw orbital update is the object a factory returns, whose Hessian linear
    # transformation the stability check uses
    _raw_approx_update_orbs: Any = None
    _raw_approx_hess_x: OTRHessX

    def _count_approx_hess_x(self, x: np.ndarray, hx: np.ndarray) -> None:
        """
        this function performs the approximate Hessian linear transformation of the
        last orbital update for the solver and counts it
        """
        self._raw_approx_hess_x(x, hx)
        self.n_hess_x += 1

    def _count_approx_update_orbs(self, approx_update_orbs: Callable) -> Callable[
        [np.ndarray, np.ndarray, np.ndarray],
        Tuple[float, Callable[[np.ndarray, np.ndarray], None]],
    ]:
        """
        this function wraps an approximate orbital update, so that the approximate
        Hessian linear transformations the solver performs with it are counted
        """

        def update_orbs(
            kappa: np.ndarray, grad: np.ndarray, h_diag: np.ndarray
        ) -> Tuple[float, Callable[[np.ndarray, np.ndarray], None]]:
            func, self._raw_approx_hess_x = approx_update_orbs(kappa, grad, h_diag)
            return func, self._count_approx_hess_x

        return update_orbs

    def _count_stability_hess_x(self, x: np.ndarray, hx: np.ndarray) -> None:
        """
        this function performs the Hessian linear transformation of the last orbital
        update for a stability check and counts it together with the two-electron
        response builds it requires
        """
        n_jk_response = self.n_jk_response
        super()._count_stability_hess_x(x, hx)
        self.n_jk_response_stability += self.n_jk_response - n_jk_response

    def _approx_stability_hess_x(self, x: np.ndarray, hx: np.ndarray) -> None:
        """
        this function performs the approximate Hessian linear transformation of the
        quasi-Newton or S-GEK extension for a stability check and counts it together
        with the two-electron response builds it requires
        """
        n_jk_response = self.n_jk_response
        self._raw_approx_update_orbs.hess_x(x, hx)
        self.n_hess_x_stability += 1
        self.n_jk_response_stability += self.n_jk_response - n_jk_response

    def __init__(self, mf: scf.SCF) -> None:
        self.__dict__.update(mf.__dict__)
        self._scf = mf
        self._record_inherited_settings()

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
                "to_otr, which evaluates the energy and the gradient, so the value "
                "assigned here reaches only the Hessian. Unless the Hessian is "
                f"approximated on purpose, set {names} before to_otr: otherwise the "
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
        self._get_indices()
        self.n_param = (1 + self.complex_orbitals) * int(
            np.count_nonzero(self.kappa_mask)
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
        full = np.zeros(self.kappa_mask.shape, dtype=rotation.dtype)
        full[self.kappa_mask] = rotation
        return full

    def _restrict(self, full: np.ndarray) -> np.ndarray:
        """
        this function returns the derivatives with respect to the rotations from those
        over the occupied-virtual pairs of PySCF, the adjoint of _embed
        """
        return full[self.kappa_mask]

    def _restrict_diagonal(self, h_diag: np.ndarray) -> np.ndarray:
        """
        this function returns the Hessian diagonal of the rotations from that over the
        occupied-virtual pairs of PySCF
        """
        return h_diag[self.kappa_mask]

    def _canonicalize(self, fock: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns the energies and the canonical orbitals within the
        occupied and within the virtual orbitals, which the PySCF object determines
        """
        return self._scf.canonicalize(self.mo_coeff, self.mo_occ, fock)

    def _func(self, kappa: np.ndarray) -> float:
        """
        this function returns the energy at a step from the current orbitals, which is
        infinite for a rotation with a modulus beyond pi / 4
        """
        rotation = self._rotation(kappa)
        if self._rotates_too_far(rotation):
            return np.inf
        u = self._exp_mat(self._unpack(rotation))
        rot_mo_coeff = self.rotate_mo(self.mo_coeff, u)
        dm = self.make_rdm1(rot_mo_coeff, self.mo_occ)
        vhf = self._potential(dm)
        return self._scf.energy_tot(dm, self.h1e, vhf)

    def _orbs_state(self) -> Tuple[np.ndarray, ...]:
        """
        this function returns the orbitals and the occupations, which define the
        point the extension objects of the routines the last kernel adopted describe
        """
        return (self.mo_coeff, self.mo_occ)

    def _update_orbs(
        self, kappa: np.ndarray, grad: np.ndarray, h_diag: np.ndarray
    ) -> Tuple[float, OTRHessX]:
        """
        this function rotates the orbitals by a step and returns the energy, its
        gradient, Hessian diagonal and Hessian linear transformation there
        """
        self._moves(kappa)

        # perform orbital rotation and get new fock matrix
        u = self._exp_mat(self._unpack(self._rotation(kappa)))
        self.mo_coeff = self.rotate_mo(self.mo_coeff, u)
        self.dm = self.make_rdm1(self.mo_coeff, self.mo_occ)
        self.vhf = self._potential(self.dm)
        self._count_update()
        self.fock = self.get_fock(self.h1e, self.s1e, self.vhf, self.dm)
        energy = self._scf.energy_tot(self.dm, self.h1e, self.vhf)
        new_grad, hess_x, new_h_diag = self.gen_g_hop(
            self.mo_coeff, self.mo_occ, self.fock
        )

        # PySCF's quantities are half the derivatives with respect to the rotations;
        # for complex orbitals, the real and imaginary parts of the gradient are the
        # derivatives with respect to the real and imaginary parts of the rotations,
        # which share the diagonal of the Hessian
        grad[:] = 2 * self._parameters(self._restrict(new_grad))
        h_diag_rotations = self._restrict_diagonal(new_h_diag)
        h_diag[:] = 2 * np.tile(h_diag_rotations, 1 + self.complex_orbitals)

        def raw_hess_x(x: np.ndarray, hx: np.ndarray) -> None:
            x_full = self._embed(self._rotation(x))
            self.n_jk_response += 1
            hx[:] = 2 * self._parameters(self._restrict(hess_x(x_full)))

        self._raw_hess_x = raw_hess_x

        # the rotation QN and S-GEK objects keep the Hessian linear transformation the
        # orbital update of their host returned last, so it refers to this object weakly
        return energy, weak_method(self._count_hess_x)

    def kernel(
        self,
        mo_coeff: Optional[np.ndarray] = None,
        mo_occ: Optional[np.ndarray] = None,
        dm0: Optional[np.ndarray] = None,
    ) -> float:
        """
        this function optimizes the orbitals from an initial guess, which it takes as
        PySCF's second-order kernel takes it, canonicalizes them and returns the total
        energy
        """
        # discard anything computed by a previous kernel
        self._reset()

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
        self._warn_on_callback(self.callback)

        mol = self._scf.mol

        # call self._scf.get_hcore, self._scf.get_ovlp because they might be overloaded
        self.h1e = self._scf.get_hcore(mol)
        self.s1e = self._scf.get_ovlp(mol)

        # set default values for OTR extensions, density QN parameterizes the orbitals
        # in the MO basis unless the orthogonal AO basis is requested, and the
        # quasi-Newton method, in any letter case, selects density QN or rotation QN
        if not hasattr(self, "oao"):
            self.oao = False
        if not hasattr(self, "qn_method"):
            self.qn_method = None
        qn_method = None if self.qn_method is None else self.qn_method.lower()
        if qn_method is not None and qn_method not in (
            density_qn_methods + rotation_qn_methods
        ):
            raise ValueError(
                f"Unknown quasi-Newton method {self.qn_method!r}: the density QN "
                f"methods are {', '.join(density_qn_methods)} and the rotation QN "
                f"methods {', '.join(rotation_qn_methods)}."
            )
        density_qn = qn_method in density_qn_methods
        rotation_qn = qn_method in rotation_qn_methods
        if isinstance(self, ROHFOTR) and (self.oao or density_qn):
            raise NotImplementedError(
                "OAO and density QN are not currently implemented for ROHF."
            )
        if self.oao and self._scf.mol.symmetry:
            raise NotImplementedError(
                "The OAO basis is not currently implemented for symmetry-adapted "
                "orbitals."
            )
        if not hasattr(self, "s_gek"):
            self.s_gek = False

        # the extensions work on the real orbitals of molecular RHF, ROHF and UHF,
        # which the orbitals of the initial guess show to be real
        periodic = isinstance(mol, pbcgto.Cell)
        extensions = self.oao or qn_method is not None or self.s_gek
        if extensions and (periodic or not isinstance(self, (RHFOTR, ROHFOTR, UHFOTR))):
            raise NotImplementedError(
                "OAO, the quasi-Newton methods and S-GEK are not currently "
                "implemented for GHF, X2C and periodic systems."
            )

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

        # get initial guess, either orbitals with their density or the eigenvectors of
        # the Fock matrix of a density
        if mo_coeff is not None and mo_occ is not None:
            density_guess = False
            dm = self.make_rdm1(mo_coeff, mo_occ)
            vhf = self._potential(dm)
            self._count_update()
            self.dm, self.vhf = dm, vhf
            fock = self.get_fock(self.h1e, self.s1e, vhf, dm, level_shift_factor=0)
            mo_energy, mo_tmp = self.eig(fock, self.s1e, **eig_kwargs)
            self.get_occ(mo_energy, mo_tmp)
            mo_tmp = None

        else:
            density_guess = True
            dm = dm0
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

        # exact orbital updates of RHF and UHF in the MO basis come from the MO
        # extension, which rotates real orbitals and evaluates responses at a density,
        # which PySCF's response functions of periodic systems do not take; the
        # approximate Hessians build on them
        complex_orbitals = np.iscomplexobj(mo_coeff)
        mo = (
            not self.oao
            and isinstance(self, (RHFOTR, UHFOTR))
            and not periodic
            and not complex_orbitals
        )
        if extensions and complex_orbitals:
            raise NotImplementedError(
                "OAO, the quasi-Newton methods and S-GEK are not currently "
                "implemented for complex orbitals."
            )

        # the MO extension evaluates the density of the orbitals of a density guess
        # itself, whereas the routines of this object start from it
        if density_guess and not mo:
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

        self.mo_coeff, self.mo_occ = np.array(mo_coeff), mo_occ

        # fix phases of MO coefficients to improve deterministic behavior
        self._fix_phase()

        # the MO extension expects the occupied orbitals of every spin first
        if mo:
            order = np.argsort(np.asarray(self.mo_occ) == 0, axis=-1, kind="stable")
            self.mo_coeff = np.take_along_axis(
                self.mo_coeff, order[..., None, :], axis=-1
            )
            self.mo_occ = np.take_along_axis(np.asarray(self.mo_occ), order, axis=-1)

        # get the parameters, the occupied-virtual rotations, which are complex for
        # complex orbitals
        self._set_parameters()

        # irreps of the MOs, whose occupied-virtual rotations between orbitals of the
        # same irrep are the parameters of the MO extension
        orbsym = None
        if mo and mol.symmetry:
            # labeled as in _get_indices
            symm_module = uhf_symm if isinstance(self, UHFOTR) else hf_symm
            orbsym = np.asarray(symm_module.get_orbsym(mol, self.mo_coeff))

        # initialize settings and the state of the solver, which the extension
        # objects read
        settings = SolverSettings()
        assign_solver_settings(self, settings)
        state = SolverState()

        # number of particles for OAO and density QN
        n_particle = 1 if isinstance(self, RHFOTR) else 2

        # number of parameters in AO basis
        if self.oao:
            self.n_param = n_particle * self._scf.mol.nao * (self._scf.mol.nao - 1) // 2

        # turn on the stability check of the solver for approximate Hessians, unless
        # stability was set on this object
        if "stability" not in vars(self) and (qn_method is not None or self.s_gek):
            self.stability = settings.stability = True

        # without parameters, as when symmetry forbids every rotation, the orbitals are
        # stationary already, so neither an extension object is built nor the solver
        # called
        if self.n_param > 0:
            # routines before the extensions replace or take over any of them
            routines = {
                name: self.__dict__.get(name) for name in self._factory_routines
            }

            # the routines this object adopts refer to it weakly, like the methods it
            # hands to the factories and the settings
            driver: Any = weakref.proxy(self)

            # the extensions replace or take over routines of this object, which are
            # recorded also if a factory or a check after one raises, so that the next
            # kernel drops them
            try:
                # get exact orbital updates from the MO or the OAO extension
                if mo or self.oao:
                    if mo:
                        mo_settings = MOSettings()
                        assign_extension_settings(self, mo_settings, mo_setting_fields)
                        # run factory, which rotates the MO coefficients in place
                        exact_update_orbs, obj_func = mo_factory(
                            self.mo_coeff,
                            self.s1e,
                            np.count_nonzero(self.mo_occ, axis=-1),
                            n_particle,
                            self._scf.mol.nao,
                            self.mo_coeff.shape[-1],
                            weak_method(self._evaluate_dm),
                            settings,
                            state,
                            mo_settings,
                            orbsym,
                        )
                        self._adopt("_func", obj_func)
                    elif self.oao:
                        if isinstance(self, RHFOTR):
                            dm_per_spin_ao = self.dm / 2
                        else:
                            dm_per_spin_ao = self.dm
                        oao_settings = OAOSettings()
                        assign_extension_settings(
                            self, oao_settings, oao_setting_fields
                        )
                        # run factory and set projection
                        exact_update_orbs, obj_func = oao_factory(
                            dm_per_spin_ao,
                            self.s1e,
                            n_particle,
                            self._scf.mol.nao,
                            weak_method(self._evaluate_dm),
                            settings,
                            state,
                            oao_settings,
                        )
                        self._adopt("_func", obj_func)
                        self.project = settings.project

                    # set preconditioner and extra trial vectors
                    self.precond = settings.precond
                    self.stability_get_extra_trial_vectors = (
                        settings.stability_settings.get_extra_trial_vectors
                    )

                    # the rotation QN and S-GEK factories accept a wrapper of the
                    # returned orbital updating function if it is marked as one
                    @functools.wraps(exact_update_orbs)
                    def wrapped_update_orbs(
                        kappa: np.ndarray, grad: np.ndarray, h_diag: np.ndarray
                    ) -> Tuple[float, OTRHessX]:
                        n_jk = driver.n_jk
                        func, hess_x = exact_update_orbs(kappa, grad, h_diag)
                        if driver._moves(kappa) or driver.n_jk > n_jk:
                            driver._count_update()
                        driver._raw_hess_x = hess_x

                        return func, weak_method(driver._count_hess_x)

                    self._adopt("_update_orbs", wrapped_update_orbs)

                # define callback functions for approximate Hessians
                if density_qn:
                    density_qn_settings = DensityQNSettings()
                    assign_extension_settings(
                        self, density_qn_settings, density_qn_setting_fields
                    )
                    # multisecant SR1 splits the linear potential into a Coulomb and an
                    # exact-exchange term, which requires an exact-exchange kernel that
                    # is positive semidefinite
                    if qn_method == "ms_sr1":
                        self._check_exchange_kernel()
                    # run factory after the factory of its basis, whose object it builds
                    # on, which holds the orbitals it updates in place and provides the
                    # exact Hessian the stability check uses
                    approx_update_orbs, obj_func = density_qn_factory(
                        weak_method(self._evaluate_dm_density_qn),
                        settings,
                        state,
                        density_qn_settings,
                    )
                    self._adopt("_func", obj_func)

                    # set the preconditioner, which inverts the approximate Hessian, and
                    # extra trial vectors
                    self.precond = settings.precond
                    self.stability_get_extra_trial_vectors = (
                        settings.stability_settings.get_extra_trial_vectors
                    )

                    def wrapped_approx_update_orbs(
                        kappa: np.ndarray, grad: np.ndarray, h_diag: np.ndarray
                    ) -> Tuple[float, Callable[[np.ndarray, np.ndarray], None]]:
                        n_jk = driver.n_jk
                        func, hess_x = approx_update_orbs(kappa, grad, h_diag)
                        if driver._moves(kappa) or driver.n_jk > n_jk:
                            driver._count_update()
                        return func, hess_x

                    self._approx_update_orbs = wrapped_approx_update_orbs
                    self._raw_approx_update_orbs = approx_update_orbs
                elif rotation_qn:
                    rotation_qn_settings = RotationQNSettings()
                    assign_extension_settings(
                        self, rotation_qn_settings, rotation_qn_setting_fields
                    )
                    self._approx_update_orbs, obj_func = rotation_qn_factory(
                        self.n_param,
                        weak_method(self._update_orbs),
                        weak_method(self._func),
                        weak_method(self._transport),
                        weak_method(self._init_hess),
                        settings,
                        state,
                        rotation_qn_settings,
                    )
                    self._adopt("_func", obj_func)
                    self._raw_approx_update_orbs = self._approx_update_orbs
                elif self.s_gek:
                    if self.oao:
                        raise RuntimeError(
                            "S-GEK cannot be performed in the orthogonal AO basis."
                        )
                    if isinstance(self, ROHFOTR):
                        raise RuntimeError(
                            "ROHFOTR is not supported for S-GEK because the history "
                            "cannot be transformed consistently to make the redundant "
                            "blocks vanish on the flag manifold."
                        )
                    s_gek_settings = SGEKSettings()
                    assign_extension_settings(
                        self, s_gek_settings, s_gek_setting_fields
                    )
                    self._approx_update_orbs, obj_func = s_gek_factory(
                        self.n_param,
                        weak_method(self._update_orbs),
                        weak_method(self._func),
                        weak_method(self._change_reference),
                        settings,
                        state,
                        s_gek_settings,
                    )
                    self._adopt("_func", obj_func)
                    self._raw_approx_update_orbs = self._approx_update_orbs
            finally:
                self._adopted_routines = tuple(
                    name
                    for name in self._factory_routines
                    if self.__dict__.get(name) is not routines[name]
                )

            # set (approximate) Hessian linear transformation for stability check
            if self.stability:
                # get linear transformation function which is continuously updated as
                # the orbitals move
                kappa = np.zeros(self.n_param, dtype=np.float64)
                grad = np.empty(self.n_param, dtype=np.float64)
                h_diag = np.empty(self.n_param, dtype=np.float64)
                self._update_orbs(kappa, grad, h_diag)

                settings.stability_hess_x = weak_method(self._count_stability_hess_x)
                if hasattr(self, "_approx_update_orbs"):
                    settings.stability_settings.approx_hess_x = weak_method(
                        self._approx_stability_hess_x
                    )

            # call solver
            self._solve(
                settings,
                state,
                (
                    self._update_orbs
                    if not hasattr(self, "_approx_update_orbs")
                    else self._count_approx_update_orbs(self._approx_update_orbs)
                ),
            )

            # the MO objects keep the orbitals they hold now
            mo_coeff_evaluated = self.mo_coeff
        else:
            self.converged = True

        # get canonical orbitals
        if self.oao:
            self.dm = 2 * dm_per_spin_ao if isinstance(self, RHFOTR) else dm_per_spin_ao
            self.mo_occ, self.mo_coeff = self._get_orth_mo_coeff()
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
        self.mo_energy, self.mo_coeff = self._canonicalize(fock)
        masks_evaluated = self._vo_masks() if mo else []

        # write the result to the chkfile as PySCF's kernel does, from the attributes
        # that hold the values it passes
        self.dump_chk(vars(self))

        self._finalize()

        # finalizing may reorder the orbitals within the occupied and the virtual
        # ones, which with symmetry changes the irrep at every orbital index, so the
        # parameters are set up anew for the final orbitals
        if not self.oao:
            self._set_parameters()

        # get the rotation from the orbitals of the MO objects to the final ones with
        # the masks of the parameters of either, with which a stability check after
        # the kernel starts from the rotations it would start from at the final
        # orbitals and returns its direction for these
        if mo and self.n_param > 0:
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
            self.stability_get_extra_trial_vectors = weak_method(
                self._canonical_trial_vectors
            )

        # the adopted routines describe the orbitals the kernel leaves, which are
        # recorded with this object so that the routines are dropped once these
        # orbitals are replaced and kept by none of its copies
        if self._adopted_routines:
            self._kernel_orbitals = LastResult.of(self._orbs_state(), weakref.ref(self))

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
        assert self._canonical_rotation is not None
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
        assert self._canonical_rotation is not None
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
        """
        this function checks whether the orbitals and occupations of this object are a
        minimum, with the Hessian evaluated with the options of this object
        """
        self._warn_on_differing_energy_options()

        # the one-electron Hamiltonian and the overlap are evaluated as in the kernel,
        # so that the check also examines orbitals assigned to an object whose kernel
        # never ran or ran for another molecule
        mol = self._scf.mol
        self.h1e = self._scf.get_hcore(mol)
        self.s1e = self._scf.get_ovlp(mol)

        # the check examines the current orbitals and occupations, also when they
        # replace the ones the last kernel left
        self._follow_orbitals()
        stable, direction = super().stability_check()

        # the MO objects hold the orbitals before they were canonicalized and
        # reordered, so the direction is rotated to the final orbitals
        if self._canonical_rotation is not None:
            direction = self._rotate_canonical(direction, True)

        return stable, direction


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
    """
    this class is the OTR driver for restricted mean-field objects, whose rotations
    are those between orbitals of different occupations that symmetry allows
    """

    def _build_potential(self, dm: np.ndarray) -> lib.NPArrayWithTag:
        """
        this function builds the effective potential at a density; restricted
        Hartree-Fock builds it as RHF.get_veff does, from one Coulomb and exchange
        build, but tags both matrices onto the result as Kohn-Sham's potential carries
        them, since its Coulomb and exact-exchange potentials need them separately
        """
        if isinstance(self, scf.hf.KohnShamDFT):
            return super()._build_potential(dm)
        vj, vk = self._scf.get_jk(self._scf.mol, dm)

        return lib.tag_array(vj - 0.5 * vk, vj=vj, vk=vk)

    def _fix_phase(self) -> None:
        """
        this function fixes the phases of the orbitals, which makes the results
        deterministic, so that the first of the largest coefficients of every orbital
        is real and positive
        """
        abs_mo_coeff = np.abs(self.mo_coeff)
        max_mo_coeff = np.max(abs_mo_coeff, axis=0)
        mask = np.isclose(abs_mo_coeff, max_mo_coeff, atol=1e-12)
        idx = np.argmax(mask, axis=0)
        cols = np.arange(self.mo_coeff.shape[1])
        vals = self.mo_coeff[idx, cols]
        self.mo_coeff *= unit_phases(vals).conj()[np.newaxis, :]

    def _get_indices(self) -> None:
        """
        this function sets up the masks of the occupied-virtual pairs over the
        orbitals, of the parameters among them, which symmetry restricts, and of their
        virtual-occupied block
        """
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
            # the orbitals are labeled as PySCF's Newton solver labels them, which also
            # works for an object that is not symmetry-adapted, as one built before the
            # symmetry of its molecule was switched on
            orbsym = hf_symm.get_orbsym(self._scf.mol, self.mo_coeff)
            sym_allow = orbsym[:, None] == orbsym
            self.kappa_mask = sym_allow[self.rot_matrix_mask]
            self.rot_matrix_mask[self.rot_matrix_mask] = self.kappa_mask

        # get non-redundant mask for virtual-closed, virtual-open, and open-closed
        # blocks which only describes which of those rotations are redundant due to
        # symmetry
        self.rot_matrix_mask_vo = self.rot_matrix_mask[
            np.ix_(self.virt_idx, self.occ_idx)
        ]

    def _exp_mat(self, matrix: np.ndarray) -> np.ndarray:
        """
        this function returns the exponential of an anti-Hermitian matrix
        """
        return ciah.expmat(matrix)

    def _unpack(self, rotation: np.ndarray) -> np.ndarray:
        """
        this function unpacks the rotations into an anti-Hermitian matrix over the
        orbitals
        """
        matrix = np.zeros(self.rot_matrix_mask.shape, dtype=rotation.dtype)
        matrix[self.rot_matrix_mask] = rotation
        return matrix - matrix.conj().T

    def _unpack_vo(self, kappa: np.ndarray) -> np.ndarray:
        """
        this function unpacks parameters into the virtual-occupied block of a rotation
        matrix
        """
        matrix = np.zeros((self.n_virt, self.n_occ), dtype=np.float64)
        matrix[self.rot_matrix_mask_vo] = kappa
        return matrix

    def _pack_vo(self, matrix: np.ndarray) -> np.ndarray:
        """
        this function packs the virtual-occupied block of a rotation matrix into
        parameters
        """
        return matrix.ravel()[self.kappa_mask]

    def _evaluate_dm(
        self, dm: np.ndarray, fock: Optional[np.ndarray], get_response: bool
    ) -> Tuple[float, Optional[Callable[[np.ndarray, np.ndarray], None]]]:
        """
        this function evaluates the energy at a density matrix of one spin for the MO
        and OAO extensions and, if asked for, the Fock matrix and the response function
        """
        # anything the caller did not ask for is not built
        veff = self._potential(2.0 * dm)
        if fock is not None:
            self.fock = self.get_fock(self.h1e, self.s1e, veff, 2.0 * dm)
            fock[:, :] = self.fock
        response = self._response_function(dm, 2.0) if get_response else None

        return self._scf.energy_tot(2.0 * dm, self.h1e, veff), response

    def _evaluate_dm_density_qn(
        self,
        dm: np.ndarray,
        fock: Optional[np.ndarray],
        v_coulomb: Optional[np.ndarray],
        v_exchange: Optional[np.ndarray],
        v_nonlinear: Optional[np.ndarray],
    ) -> float:
        """
        this function evaluates the energy at a density matrix of one spin for the
        density QN extension and, if asked for, the Fock matrix and the Coulomb,
        exact-exchange and non-linear (exchange-correlation) parts of the effective
        potential
        """
        # anything the caller did not ask for is not built
        veff = self._potential(2.0 * dm)

        # linear (Coulomb and exact exchange) part of the potential, the Coulomb minus
        # the exact-exchange potential, which is half the exchange matrix tagged onto
        # the potential, already weighted by the exact-exchange fractions of the
        # functional, and vanishes without exact exchange
        vk = getattr(veff, "vk", None)
        exchange = 0.0 if vk is None else 0.5 * np.asarray(vk)
        if v_coulomb is not None:
            v_coulomb[:, :] = veff.vj
        if v_exchange is not None:
            v_exchange[:, :] = exchange

        # non-linear (exchange-correlation) part, which Hartree-Fock does not have
        if v_nonlinear is not None:
            if isinstance(self, scf.hf.KohnShamDFT):
                v_nonlinear[:, :] = np.asarray(veff) - (np.asarray(veff.vj) - exchange)
            else:
                v_nonlinear[:, :] = 0.0

        if fock is not None:
            fock[:, :] = self.get_fock(self.h1e, self.s1e, veff, 2.0 * dm)

        return self._scf.energy_tot(2.0 * dm, self.h1e, veff)

    def _transport(self, geodesic: np.ndarray, tangent_vector: np.ndarray) -> None:
        """
        this function transports a tangent vector along a geodesic for the rotation QN
        extension, which leaves its parameters unchanged: the commutator of two
        occupied-virtual rotations has only occupied-occupied and virtual-virtual
        blocks, which are redundant, so that every term of the series in
        ROHFOTR._transport vanishes
        """
        return

    def _init_hess(self, vector: np.ndarray) -> None:
        """
        this function applies the initial Hessian of the rotation QN extension to
        parameters in place, the approximation 2 (F_vv x - x F_oo) that neglects the
        response of the effective potential to the rotation
        """
        mo_coeff_occ = self.mo_coeff[:, self.occ_idx]
        mo_coeff_virt = self.mo_coeff[:, self.virt_idx]

        fock_oo = mo_coeff_occ.T @ self.fock @ mo_coeff_occ
        fock_vv = mo_coeff_virt.T @ self.fock @ mo_coeff_virt

        x_vo = self._unpack_vo(vector)
        x_vo = 2 * (fock_vv @ x_vo - x_vo @ fock_oo)
        vector[:] = self._pack_vo(x_vo)

    def _change_reference(
        self,
        new_ref: np.ndarray,
        kappa: np.ndarray,
        local_grad: np.ndarray,
        grad: np.ndarray,
    ) -> None:
        """
        this function performs a gauge transformation for the S-GEK extension which
        requires the history of orbital rotation parameters to have vanishing
        occupied-occupied and virtual-virtual blocks such that the GEK metric is
        consistent
        """
        # get reference rotation matrix
        ref_rot_mat = self._exp_mat(self._unpack(new_ref))

        for i in range(kappa.shape[0]):
            # get matrix to be rotated
            rot_mat = self._exp_mat(self._unpack(kappa[i, :]))

            # get combined rotation matrix
            combined_rot = ref_rot_mat.T @ rot_mat
            combined_rot_oo = combined_rot[np.ix_(self.occ_idx, self.occ_idx)]
            combined_rot_ov = combined_rot[np.ix_(self.occ_idx, self.virt_idx)]
            combined_rot_vo = combined_rot[np.ix_(self.virt_idx, self.occ_idx)]
            combined_rot_vv = combined_rot[np.ix_(self.virt_idx, self.virt_idx)]

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
            kappa[i, :] = self._pack_vo(combined_rot_vo @ (vh.T * scos) @ u.T)

            # transform local gradients to new orbitals
            n_occ = rot_occ.shape[0]
            n_virt = rot_virt.shape[0]
            occ_unit = np.allclose(rot_occ, np.eye(n_occ), atol=1.0e-10, rtol=0)
            virt_unit = np.allclose(rot_virt, np.eye(n_virt), atol=1.0e-10, rtol=0)
            if not occ_unit or not virt_unit:
                local_grad_2d = self._unpack_vo(local_grad[i, :])
                if not occ_unit:
                    local_grad_2d = local_grad_2d @ rot_occ
                if not virt_unit:
                    local_grad_2d = rot_virt.T @ local_grad_2d
                local_grad[i, :] = self._pack_vo(local_grad_2d)

            # transform gradients to new reference
            u, s, vh = sc.linalg.svd(self._unpack_vo(kappa[i, :]), full_matrices=False)
            t0 = self._unpack_vo(local_grad[i, :])
            z = u.T @ t0 @ vh.T
            dm = 0.5 * np.sinc((s.reshape(-1, 1) - s) / np.pi)
            dp = 0.5 * np.sinc((s.reshape(-1, 1) + s) / np.pi)
            grad_2d = u @ ((z + z.T) * dm + (z - z.T) * dp) @ vh
            # decide whether the occupied or virtual orbitals define the rank of the
            # parameter matrix
            if n_occ <= n_virt:
                grad_2d += (
                    (np.eye(n_virt) - u @ u.T) @ t0 @ ((vh.T * np.sinc(s / np.pi)) @ vh)
                )
            else:
                grad_2d += (
                    ((u * np.sinc(s / np.pi)) @ u.T) @ t0 @ (np.eye(n_occ) - vh.T @ vh)
                )
            grad[i, :] = self._pack_vo(grad_2d)

        return

    def _get_orth_mo_coeff(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns the occupations and the natural orbitals of the density
        matrix, which define the orbitals after the OAO extension optimized the density
        matrix
        """
        eigvals_s, eigvecs_s = np.linalg.eigh(self.s1e)
        s_sqrt = eigvecs_s @ np.diag(np.sqrt(eigvals_s)) @ eigvecs_s.T
        s_inv_sqrt = eigvecs_s @ np.diag(1.0 / np.sqrt(eigvals_s)) @ eigvecs_s.T
        dm_orth = s_sqrt @ self.dm @ s_sqrt
        mo_occ, mo_coeff_oao = np.linalg.eigh(dm_orth)

        # PySCF orders the orbitals with the occupied ones first, so the natural
        # orbitals are taken in decreasing occupation, which is rounded to a
        # nonnegative integer
        mo_occ = np.abs(np.rint(mo_occ[..., ::-1]))
        mo_coeff = s_inv_sqrt @ mo_coeff_oao[..., ::-1]
        return mo_occ, mo_coeff


class ROHFOTR(SecondOrderOTR, newton_ah._SecondOrderROHF):
    """
    this class is the OTR driver for restricted open-shell mean-field objects, whose
    rotations are those of the restricted ones. PySCF's ROHF Hessian lacks the terms
    proportional to the gradient, which vanish at a stationary point, so that it is
    approximate away from one, which only slows the convergence; PySCF releases up to
    2.14 also lack orbital-rotation terms that persist at a stationary point
    (pyscf#3448), so that the stability check of an ROHF point, like PySCF's own, is
    approximate with those releases
    """

    _fix_phase = RHFOTR._fix_phase

    def _get_indices(self) -> None:
        """
        this function sets up the masks of the virtual-closed, virtual-open and
        open-closed pairs over the orbitals, the blocks of the flag manifold, and of
        the parameters among them, which symmetry restricts
        """
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
            # labeled as the restricted orbitals are
            orbsym = hf_symm.get_orbsym(self._scf.mol, self.mo_coeff)
            sym_allow = orbsym[:, None] == orbsym
            self.kappa_mask = sym_allow[self.rot_matrix_mask]
            self.rot_matrix_mask[self.rot_matrix_mask] = self.kappa_mask

    _exp_mat = RHFOTR._exp_mat
    _unpack = RHFOTR._unpack

    def _transport(self, geodesic: np.ndarray, tangent_vector: np.ndarray) -> None:
        """
        this function transports tangent_vector from the tangent space at a starting
        base to the tangent space at a new base along the geodesic. tangent_vector and
        geodesic have the same base. In other words, the displacement described by
        tangent_vector is defined in terms of the MO coefficients defined by
        the starting base, and this function transforms it to the MO coefficients
        defined by the new base. The geodesic describes the rotation from the starting
        base to the new base.
        """
        # unpack into full matrix
        geodesic_full = self._unpack(geodesic)

        # define tolerance for truncation of series expansion
        tol = np.finfo(np.float64).eps

        # k = 0
        tangent_vector_full = self._unpack(tangent_vector)
        # k = 1
        k = 1
        current_term = -(1 / 2) * self._remove_diag_blocks(
            geodesic_full @ tangent_vector_full - tangent_vector_full @ geodesic_full
        )
        tangent_vector_full = tangent_vector_full + current_term
        # k > 1
        while (np.linalg.norm(current_term) > tol) and (k < 200):
            k += 1
            current_term = (-1 / (2 * k)) * self._remove_diag_blocks(
                geodesic_full @ current_term - current_term @ geodesic_full
            )
            tangent_vector_full = tangent_vector_full + current_term
        current_term_norm = np.linalg.norm(current_term)
        if current_term_norm > tol:
            raise RuntimeError(
                "Parallel transport truncated before tolerance is reached. Norm of "
                f"the last term: {current_term_norm}."
            )

        # ensure the transported kappa and grad are in the tangent space of the new
        # reference
        tangent_vector_full = self._ensure_tangent(tangent_vector_full)

        # pack back into non-redundant parameters
        tangent_vector[:] = tangent_vector_full[self.rot_matrix_mask]

        return

    def _remove_diag_blocks(self, matrix: np.ndarray) -> np.ndarray:
        """
        this function removes the redundant diagonal blocks of the matrix corresponding
        to closed, open, and virtual orbitals, which ensures the output is in the
        tangent space of the reference
        """
        matrix[np.ix_(self.clos_idx, self.clos_idx)] = 0.0
        matrix[np.ix_(self.open_idx, self.open_idx)] = 0.0
        matrix[np.ix_(self.virt_idx, self.virt_idx)] = 0.0
        return matrix

    def _ensure_tangent(self, tangent_vector: np.ndarray) -> np.ndarray:
        """
        this function ensures the input tangent vector is in the tangent space of the
        reference by removing the redundant diagonal blocks and anti-symmetrizing the
        matrix
        """
        return self._remove_diag_blocks(0.5 * (tangent_vector - tangent_vector.T))

    def _get_orth_mo_coeff(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns the occupations and the natural orbitals of the total
        density matrix, which define the orbitals after the OAO extension optimized the
        density matrix
        """
        eigvals_s, eigvecs_s = np.linalg.eigh(self.s1e)
        s_sqrt = eigvecs_s @ np.diag(np.sqrt(eigvals_s)) @ eigvecs_s.T
        s_inv_sqrt = eigvecs_s @ np.diag(1.0 / np.sqrt(eigvals_s)) @ eigvecs_s.T
        dm_orth = s_sqrt @ (self.dm[0] + self.dm[1]) @ s_sqrt
        mo_occ, mo_coeff_oao = np.linalg.eigh(dm_orth)

        # PySCF orders the orbitals with the occupied ones first, so the natural
        # orbitals are taken in decreasing occupation, which is rounded to a
        # nonnegative integer
        mo_occ = np.abs(np.rint(mo_occ[..., ::-1]))
        mo_coeff = s_inv_sqrt @ mo_coeff_oao[..., ::-1]
        return mo_occ, mo_coeff


class UHFOTR(SecondOrderOTR, newton_ah._SecondOrderUHF):
    """
    this class is the OTR driver for unrestricted mean-field objects, whose rotations
    are those between occupied and virtual orbitals of either spin that symmetry
    allows
    """

    def _fix_phase(self) -> None:
        """
        this function fixes the phases of the orbitals of both spins, as for the
        restricted orbitals
        """
        abs_mo_coeff = np.abs(self.mo_coeff)
        max_mo_coeff = np.max(abs_mo_coeff, axis=1)
        for i in range(2):
            mask = np.isclose(abs_mo_coeff[i], max_mo_coeff[i], atol=1e-12)
            idx = np.argmax(mask, axis=0)
            cols = np.arange(self.mo_coeff.shape[2])
            vals = self.mo_coeff[i, idx, cols]
            self.mo_coeff[i] *= unit_phases(vals).conj()[np.newaxis, :]

    def _build_potential(self, dm: np.ndarray) -> lib.NPArrayWithTag:
        """
        this function builds the effective potential at a density; unrestricted
        Hartree-Fock builds it from the Coulomb and exchange matrices of each spin and
        tags these onto the result, since its Coulomb and exact-exchange potentials
        need them separately
        """
        if isinstance(self, scf.hf.KohnShamDFT):
            return super()._build_potential(dm)
        vj, vk = self._scf.get_jk(self._scf.mol, dm)

        return lib.tag_array(vj[0] + vj[1] - vk, vj=vj, vk=vk)

    def _spin_coulomb(self, dm: np.ndarray, veff: lib.NPArrayWithTag) -> np.ndarray:
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

    def _get_indices(self) -> None:
        """
        this function sets up the masks of the occupied-virtual pairs over the
        orbitals of either spin, of the parameters among them, which symmetry
        restricts, and of their virtual-occupied blocks
        """
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
            # labeled as the restricted orbitals are
            orbsyma, orbsymb = uhf_symm.get_orbsym(self._scf.mol, self.mo_coeff)
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

    def _exp_mat(self, matrix: List[np.ndarray]) -> List[np.ndarray]:
        """
        this function returns the exponentials of the anti-Hermitian matrices of both
        spins
        """
        return [ciah.expmat(matrix[0]), ciah.expmat(matrix[1])]

    def _unpack(self, rotation: np.ndarray) -> List[np.ndarray]:
        """
        this function unpacks the rotations into an anti-Hermitian matrix over the
        orbitals of each spin
        """
        n_param_a = np.count_nonzero(self.rot_matrix_mask[0])
        matrices = [
            np.zeros(mask.shape, dtype=rotation.dtype) for mask in self.rot_matrix_mask
        ]
        matrices[0][self.rot_matrix_mask[0]] = rotation[:n_param_a]
        matrices[1][self.rot_matrix_mask[1]] = rotation[n_param_a:]
        return [matrix - matrix.conj().T for matrix in matrices]

    def _unpack_vo(self, kappa: np.ndarray) -> List[np.ndarray]:
        """
        this function unpacks parameters into the virtual-occupied blocks of the
        rotation matrices of both spins
        """
        n_param_a = np.count_nonzero(self.rot_matrix_mask[0])
        matrix = [
            np.zeros((self.n_virt[0], self.n_occ[0]), dtype=np.float64),
            np.zeros((self.n_virt[1], self.n_occ[1]), dtype=np.float64),
        ]
        matrix[0][self.rot_matrix_mask_vo[0]] = kappa[:n_param_a]
        matrix[1][self.rot_matrix_mask_vo[1]] = kappa[n_param_a:]
        return matrix

    def _pack_vo(self, matrix: List[np.ndarray]) -> np.ndarray:
        """
        this function packs the virtual-occupied blocks of the rotation matrices of
        both spins into parameters
        """
        return np.concatenate((matrix[0].ravel(), matrix[1].ravel()))[self.kappa_mask]

    def _evaluate_dm(
        self, dm: np.ndarray, fock: Optional[np.ndarray], get_response: bool
    ) -> Tuple[float, Optional[Callable[[np.ndarray, np.ndarray], None]]]:
        """
        this function evaluates the energy at the density matrices of both spins for
        the MO and OAO extensions and, if asked for, the Fock matrices and the response
        function
        """
        # anything the caller did not ask for is not built
        vhf = self._potential(dm)
        if fock is not None:
            fock[:, :, :] = self._scf.get_fock(self.h1e, self.s1e, vhf, dm)
        response = self._response_function(dm, 1.0) if get_response else None

        return self._scf.energy_tot(dm, self.h1e, vhf), response

    def _evaluate_dm_density_qn(
        self,
        dm: np.ndarray,
        fock: Optional[np.ndarray],
        v_coulomb: Optional[np.ndarray],
        v_exchange: Optional[np.ndarray],
        v_nonlinear: Optional[np.ndarray],
    ) -> float:
        """
        this function evaluates the energy at the density matrices of both spins for
        the density QN extension and, if asked for, the Fock matrices and the Coulomb,
        exact-exchange and non-linear (exchange-correlation) parts of the effective
        potentials
        """
        # anything the caller did not ask for is not built
        veff = self._potential(dm)
        ks = isinstance(self, scf.hf.KohnShamDFT)

        # exact-exchange potential of each spin, the exchange matrix tagged onto the
        # potential, already weighted by the exact-exchange fractions of the
        # functional, which vanishes without exact exchange
        vk = getattr(veff, "vk", None)
        exchange = 0.0 if vk is None else np.asarray(vk)
        if v_exchange is not None:
            v_exchange[:, :, :] = exchange

        # Coulomb potential of the density matrix of each spin, which Hartree-Fock tags
        # onto its potential, while Kohn-Sham tags only that of the total density and
        # needs a second pass
        if v_coulomb is not None or (ks and v_nonlinear is not None):
            vj = self._spin_coulomb(dm, veff) if ks else veff.vj
            if v_coulomb is not None:
                v_coulomb[:, :, :] = vj

        # non-linear (exchange-correlation) part, the potential minus the linear part,
        # the Coulomb potential of both spins minus the exact-exchange potential, which
        # Hartree-Fock does not have
        if v_nonlinear is not None:
            if ks:
                v_nonlinear[:, :, :] = np.asarray(veff) - (vj[0] + vj[1] - exchange)
            else:
                v_nonlinear[:, :, :] = 0.0

        # construct Fock matrix
        if fock is not None:
            fock[:, :, :] = self._scf.get_fock(self.h1e, self.s1e, veff, dm)

        return self._scf.energy_tot(dm, self.h1e, veff)

    def _change_reference(
        self,
        new_ref: np.ndarray,
        kappa: np.ndarray,
        local_grad: np.ndarray,
        grad: np.ndarray,
    ) -> None:
        """
        this function performs a gauge transformation for the S-GEK extension which
        requires the history of orbital rotation parameters to have vanishing
        occupied-occupied and virtual-virtual blocks such that the GEK metric is
        consistent
        """
        # get reference rotation matrix
        ref_rot_mat = self._exp_mat(self._unpack(new_ref))

        for i in range(kappa.shape[0]):
            # get matrix to be rotated
            rot_mat = self._exp_mat(self._unpack(kappa[i, :]))

            kappa_2d = []
            local_grad_2d = self._unpack_vo(local_grad[i, :])
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
                # decide whether the occupied or virtual orbitals define the rank of the
                # parameter matrix
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
            kappa[i, :] = self._pack_vo(kappa_2d)
            local_grad[i, :] = self._pack_vo(local_grad_2d)
            grad[i, :] = self._pack_vo(grad_2d)

        return

    def _transport(self, geodesic: np.ndarray, tangent_vector: np.ndarray) -> None:
        """
        this function transports a tangent vector along a geodesic for the rotation QN
        extension, which leaves its parameters unchanged, as RHFOTR._transport
        explains for the rotations of each spin
        """
        return

    def _get_orth_mo_coeff(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns the occupations and the natural orbitals of the density
        matrix of each spin, which define the orbitals after the OAO extension
        optimized the density matrices
        """
        eigvals_s, eigvecs_s = np.linalg.eigh(self.s1e)
        s_sqrt = eigvecs_s @ np.diag(np.sqrt(eigvals_s)) @ eigvecs_s.T
        s_inv_sqrt = eigvecs_s @ np.diag(1.0 / np.sqrt(eigvals_s)) @ eigvecs_s.T
        dm_orth = s_sqrt @ self.dm @ s_sqrt
        mo_occ, mo_coeff_oao = np.linalg.eigh(dm_orth)

        # PySCF orders the orbitals with the occupied ones first, so the natural
        # orbitals are taken in decreasing occupation, which is rounded to a
        # nonnegative integer
        mo_occ = np.abs(np.rint(mo_occ[..., ::-1]))
        mo_coeff = s_inv_sqrt @ mo_coeff_oao[..., ::-1]
        return mo_occ, mo_coeff


class GHFOTR(SecondOrderOTR, newton_ah._SecondOrderGHF):
    """
    this class is the OTR driver for generalized mean-field objects, whose rotations
    are those between occupied and virtual spin orbitals that symmetry allows
    """

    _fix_phase = RHFOTR._fix_phase
    _exp_mat = RHFOTR._exp_mat
    _unpack = RHFOTR._unpack

    def _get_indices(self) -> None:
        """
        this function sets up the masks of the occupied-virtual pairs over the spin
        orbitals and of the parameters among them, which symmetry restricts
        """
        occ_idx = self.mo_occ > 0
        self.rot_matrix_mask = ~occ_idx[:, None] & occ_idx
        self.kappa_mask = self.rot_matrix_mask[self.rot_matrix_mask]
        if self._scf.mol.symmetry:
            # labeled as PySCF's Newton solver labels the spin orbitals
            orbsym = ghf_symm.get_orbsym(self._scf.mol, self.mo_coeff)
            sym_allow = orbsym[:, None] == orbsym
            self.kappa_mask = sym_allow[self.rot_matrix_mask]
            self.rot_matrix_mask[self.rot_matrix_mask] = self.kappa_mask


class X2CRHFOTR(SecondOrderOTR, x2c_newton_ah.SecondOrderX2CRHF):
    """
    this class is the OTR driver for Kramers-restricted spinor X2C, which keeps every
    odd spinor the time reverse of the even one before it, which the antiunitary time
    reversal maintains for rotations that turn the even spinors by a matrix and their
    partners by its complex conjugate. For every pair of a virtual and an occupied
    Kramers pair, the rotations are A between the even spinors and C of the even
    occupied spinor into the odd virtual one, whose time reverse turns the odd occupied
    spinor into the even virtual one by -conj(C); PySCF's own second-order solver turns
    both spinors of a pair alike and omits C, which keeps the pairs only for real
    rotations
    """

    _exp_mat = RHFOTR._exp_mat

    def _fix_phase(self) -> None:
        """
        this function fixes the phases of the even spinors as for restricted orbitals
        and gives the partner of every even spinor the conjugate phase, which keeps it
        its time reverse
        """
        even = self.mo_coeff[:, ::2]
        abs_even = np.abs(even)
        mask = np.isclose(abs_even, np.max(abs_even, axis=0), atol=1e-12)
        idx = np.argmax(mask, axis=0)
        phases = unit_phases(even[idx, np.arange(even.shape[1])])
        self.mo_coeff[:, ::2] *= phases.conj()[np.newaxis, :]
        self.mo_coeff[:, 1::2] *= phases[np.newaxis, :]

    def _get_indices(self) -> None:
        """
        this function sets up the mask of the combinations of a virtual and an
        occupied Kramers pair, each of which has the rotations A and C
        """
        occ_pairs = self.mo_occ[::2] > 0
        self.rot_matrix_mask = ~occ_pairs[:, None] & occ_pairs
        self.kappa_mask = np.ones(
            2 * int(np.count_nonzero(self.rot_matrix_mask)), dtype=bool
        )

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

    def _unpack(self, rotation: np.ndarray) -> np.ndarray:
        """
        this function unpacks the rotations into an anti-Hermitian matrix over the
        spinors
        """
        occidx = self.mo_occ > 0
        matrix = np.zeros(2 * (self.mo_occ.size,), dtype=rotation.dtype)
        matrix[np.ix_(~occidx, occidx)] = self._vo_block(rotation)
        return matrix - matrix.conj().T

    def _embed(self, rotation: np.ndarray) -> np.ndarray:
        """
        this function returns the vector over the virtual-occupied pairs of spinors of
        PySCF's derivatives that the rotations A and C correspond to
        """
        return self._vo_block(rotation).ravel()

    def _restrict(self, full: np.ndarray) -> np.ndarray:
        """
        this function returns the derivatives with respect to the rotations A and C
        from those over the virtual-occupied pairs of spinors, the adjoint of _embed
        """
        n_occ = int(np.count_nonzero(self.mo_occ > 0))
        block = full.reshape(-1, n_occ)
        a = block[::2, ::2] + block[1::2, 1::2].conj()
        c = block[1::2, ::2] - block[::2, 1::2].conj()
        return np.concatenate((a.ravel(), c.ravel()))

    def _restrict_diagonal(self, h_diag: np.ndarray) -> np.ndarray:
        """
        this function returns the Hessian diagonal of the rotations A and C from that
        over the virtual-occupied pairs of spinors
        """
        n_occ = int(np.count_nonzero(self.mo_occ > 0))
        block = h_diag.reshape(-1, n_occ)
        a = block[::2, ::2] + block[1::2, 1::2]
        c = block[1::2, ::2] + block[::2, 1::2]
        return np.concatenate((a.ravel(), c.ravel()))

    def _canonicalize(self, fock: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns the energies and the canonical orbitals within the
        occupied and within the virtual Kramers pairs from the quaternion eigensolver
        of the PySCF object, which returns them as pairs, since PySCF's generic
        canonicalization mixes the degenerate partners of a pair
        """
        mo_energy = np.empty(self.mo_occ.size)
        mo_coeff = np.empty_like(self.mo_coeff)
        for idx in (self.mo_occ > 0, self.mo_occ == 0):
            orbs = self.mo_coeff[:, idx]
            # the eigensolver takes the even spinors before their partners
            basis = np.hstack((orbs[:, ::2], orbs[:, 1::2]))
            mo_energy[idx], mo_coeff[:, idx] = self._scf._eigh(fock, self.s1e, x=basis)
        return mo_energy, mo_coeff


class X2CUHFOTR(SecondOrderOTR, x2c_newton_ah.SecondOrderX2CUHF):
    """
    this class is the OTR driver for unrestricted spinor X2C, whose rotations are
    those between occupied and virtual spinors
    """

    _fix_phase = RHFOTR._fix_phase
    _exp_mat = RHFOTR._exp_mat
    _unpack = RHFOTR._unpack

    def _get_indices(self) -> None:
        """
        this function sets up the mask of the occupied-virtual pairs over the spinors,
        which point-group symmetry can not restrict, since PySCF has no symmetry labels
        for spinors
        """
        occ_idx = self.mo_occ > 0
        self.rot_matrix_mask = ~occ_idx[:, None] & occ_idx
        self.kappa_mask = self.rot_matrix_mask[self.rot_matrix_mask]


def _mf_to_otr(mf: scf.hf.SCF) -> SecondOrderOTR:
    """
    this function converts a mean-field object into an object of its driver class,
    unless the class is not supported, as the four-component ones and those with
    k-points
    """
    # the spinor classes of X2C carry the names of the nonrelativistic classes, so they
    # are told apart by their type first, and an ROHF class also carries the name of
    # RHF, from which it derives
    cls: Type[SecondOrderOTR]
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
    else:
        raise _unsupported(mf)

    mf_otr = lib.set_class(cls(mf), (cls, mf.__class__), cls.__name__)
    mf_otr._record_inherited_settings()

    return mf_otr


class CASSCFOTR(OTR, newton_casscf.CASSCF):
    """
    this class is the OTR driver for CASSCF, whose parameters are the orbital
    rotations PySCF's second-order CASSCF selects and the CI coefficients of all
    states that the symmetry of their CI solvers allows
    """

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
        self._record_inherited_settings()

    @staticmethod
    def _ci_vectors(ci: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
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
        vectors = [np.ravel(c) for c in self._ci_vectors(ci)]
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

    def _displaced_ci(self, x: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
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

    def _eigenstates(
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

    def _func(self, x: np.ndarray) -> float:
        """
        this function returns the energy at a step from the current orbitals and CI
        vectors, which is infinite for an orbital rotation beyond pi / 4, whereas the
        CI part of a step is a displacement of the CI vectors rather than a rotation
        angle
        """
        if self._rotates_too_far(x[: self.n_param_orb]):
            return np.inf
        u = ciah.expmat(self.unpack_uniq_var(x[: self.n_param_orb]))

        rot_mo_coeff = self.rotate_mo(self.mo_coeff, u)
        eris = self.ao2mo(rot_mo_coeff)

        return self._eigenstates(rot_mo_coeff, self._displaced_ci(x), eris)[0]

    def _update_orbs(
        self, x: np.ndarray, grad: np.ndarray, h_diag: np.ndarray
    ) -> Tuple[float, OTRHessX]:
        """
        this function moves the orbitals and CI vectors by a step and returns the
        energy, its gradient, Hessian diagonal and Hessian linear transformation there
        """
        self._moves(x)
        u = ciah.expmat(self.unpack_uniq_var(x[: self.n_param_orb]))
        self.mo_coeff = self.rotate_mo(self.mo_coeff, u)
        eris = self.ao2mo(self.mo_coeff)

        # the states move on as the eigenstates within the space the displaced CI
        # vectors span, at which gen_g_hop's derivatives are those of the energy
        energy, states = self._eigenstates(self.mo_coeff, self._displaced_ci(x), eris)
        self.ci = states[0] if isinstance(self.ci, np.ndarray) else states

        new_grad, _, hess_x, new_h_diag = newton_casscf.gen_g_hop(
            self, self.mo_coeff, self.ci, eris
        )

        # gen_g_hop extends over all CI coefficients and takes the CI part of the
        # Hessian linear transformation of every state orthogonal only to that state,
        # whereas the steps are orthogonal to all states of its CI solver; its CI
        # gradient is orthogonal to all of them already, since the states are
        # eigenstates within the space they span
        grad[:] = self._restrict(new_grad)
        h_diag[:] = self._restrict(new_h_diag)

        def raw_hess_x(x: np.ndarray, hx: np.ndarray) -> None:
            x_full = self._orthogonal_ci(self._embed(x))
            hx[:] = self._restrict(self._orthogonal_ci(hess_x(x_full)))

        self._raw_hess_x = raw_hess_x

        return energy, self._count_hess_x

    def kernel(
        self,
        mo_coeff: Optional[np.ndarray] = None,
        ci0: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        callback: Optional[Callable[..., None]] = None,
        _kern: Any = None,
    ) -> Tuple[
        Optional[float],
        Optional[float],
        Union[np.ndarray, List[np.ndarray]],
        np.ndarray,
        Optional[np.ndarray],
    ]:
        """
        this function optimizes the orbitals and CI vectors passed in or those of this
        object and returns the values PySCF's CASSCF kernel returns
        """
        # discard anything computed by a previous kernel
        self._reset()

        # the SCF object is run first if there are no orbitals yet, as PySCF's kernel
        # does
        if mo_coeff is None:
            if self.mo_coeff is None and self._scf.mol.nelectron > 0:
                self._scf.run()
                self.mo_coeff = self._scf.mo_coeff
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if callback is None:
            callback = self.callback
        self._warn_on_callback(callback)

        if ci0 is None:
            ci0 = self.ci

        # warn about attributes no class registers, such as misspelled settings, and
        # check the active space, as PySCF's kernel does
        self.check_sanity()

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

        # write the result to the chkfile as PySCF's kernel does, from the attributes
        # that hold the values it passes and the active space density matrix
        casdm1 = self.fcisolver.make_rdm1(self.ci, self.ncas, self.nelecas)
        self.dump_chk({**vars(self), "casdm1": casdm1, "fcivec": self.ci})

        self._finalize()

        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy


def _casscf_to_otr(casscf: mc1step.CASSCF) -> CASSCFOTR:
    """
    this function converts a CASSCF object into an object of its driver class
    """
    # the driver builds on PySCF's second-order class, without a state average, which
    # has to precede the class it averages and is therefore applied again below
    if not isinstance(casscf, newton_casscf.CASSCF):
        casscf = casscf.newton()
    pyscf_class = (
        casscf._base_class
        if isinstance(casscf, addons.StateAverageMCSCFSolver)
        else casscf.__class__
    )

    casscf_otr = lib.set_class(
        CASSCFOTR(casscf._scf, casscf.ncas, casscf.nelecas),
        (CASSCFOTR, pyscf_class),
        CASSCFOTR.__name__,
    )
    casscf_otr.__dict__.update(casscf.__dict__)
    casscf_otr._record_inherited_settings()

    # the state average is applied with the symmetry of the wave functions, as PySCF's
    # newton() applies it
    if isinstance(casscf, addons.StateAverageMCSCFSolver):
        wfnsym = getattr(casscf, "wfnsym", None)
        casscf_otr = casscf_otr.state_average_(casscf.weights, wfnsym)

    return casscf_otr


def to_otr(obj: Any) -> Any:
    """
    this function converts a PySCF mean-field, localizer or CASSCF object into its
    OpenTrustRegion counterpart, which takes over its attributes and remains usable as
    the PySCF object
    """
    if isinstance(obj, OTR):
        return obj

    if isinstance(obj, scf.hf.SCF):
        return _mf_to_otr(obj)
    if type(obj) in localizer_drivers:
        return _localizer_to_otr(obj)
    if isinstance(obj, mc1step.CASSCF):
        return _casscf_to_otr(obj)
    raise _unsupported(obj)


def _unsupported(obj: Any) -> TypeError:
    """
    this function returns the error to_otr raises for an object it does not support
    """
    return TypeError(
        "to_otr supports RHF, ROHF, UHF and GHF and their Kohn-Sham counterparts, also "
        "at the Gamma point of periodic systems, spinor X2C, the Boys, Pipek-Mezey and "
        "Edmiston-Ruedenberg localizers and CASSCF, but not "
        f"{type(obj).__module__}.{type(obj).__name__}."
    )
