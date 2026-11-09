# Copyright (C) 2025- Jonas Greiner
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Tests for the PySCF interface to OpenTrustRegion
"""

import gc
import io
import types
import unittest
import weakref
from contextlib import contextmanager, redirect_stderr
from unittest import mock
from itertools import product
from typing import Any, Callable, Generator, List, Optional, Tuple

import numpy as np
import scipy.linalg

from pyscf import gto, scf, dft, fci, lo, mcscf, lib
from pyscf.pbc import gto as pbcgto, scf as pbcscf
from pyscf.scf import dhf
from pyscf.tools import mo_mapping
from pyscf.x2c import x2c

from pyscf.opentrustregion import to_otr
from pyscf.opentrustregion.opentrustregion_interface import (
    RHFOTR,
    ROHFOTR,
    UHFOTR,
    GHFOTR,
    X2CRHFOTR,
    X2CUHFOTR,
    BoysOTR,
    PipekMezeyOTR,
    EdmistonRuedenbergOTR,
    CASSCFOTR,
    OTR,
    SecondOrderOTR,
    transport_rotation,
    assign_extension_settings,
    assign_solver_settings,
    assign_stability_settings,
    mo_setting_fields,
    non_setting_fields,
    rotation_qn_methods,
    setting_is_assignable,
    setting_was_set_by_user,
    solver_setting_fields,
    stability_setting_aliases,
    stability_setting_fields,
)
from pyopentrustregion import SolverSettings, StabilitySettings
from pyopentrustregion.extensions.mo import MOSettings

# fixtures shared by every test, built once in setUpModule
mol: gto.Mole
mol_sym: gto.Mole
mol_open: gto.Mole
mol_open_sym: gto.Mole
mf: scf.hf.RHF
mf_sym: scf.hf.RHF
occ_orbs: np.ndarray


def setUpModule() -> None:
    global mol, mol_sym, mol_open, mol_open_sym, mf, mf_sym, occ_orbs
    mol = gto.M(
        atom="""
            O  0.000  0.000  0.000
            H  0.000 -0.757  0.587
            H  0.000  0.757  0.587
        """,
        basis="6-31g",
        verbose=0,
        output="/dev/null",
    )
    mol_sym = mol.copy()
    mol_sym.symmetry = True
    mol_sym.build(False, False)
    mol_open = mol.copy()
    mol_open.spin = 2
    mol_open.build(False, False)
    mol_open_sym = mol_open.copy()
    mol_open_sym.symmetry = True
    mol_open_sym.build(False, False)

    mf = scf.RHF(mol).run()
    mf_sym = scf.RHF(mol_sym).run()
    occ_orbs = mf.mo_coeff[:, : mol.nelec[0]]


def tearDownModule() -> None:
    global mol, mol_sym, mol_open, mol_open_sym, mf, mf_sym, occ_orbs
    mol.stdout.close()
    del mol, mol_sym, mol_open, mol_open_sym, mf, mf_sym, occ_orbs


def grad_and_h_diag(obj: OTR) -> Tuple[np.ndarray, np.ndarray]:
    """
    this function allocates the gradient and Hessian diagonal arrays that an orbital
    update writes into
    """
    return (
        np.empty(obj.n_param, dtype=np.float64),
        np.empty(obj.n_param, dtype=np.float64),
    )


def gradient_rms(obj: OTR) -> float:
    """
    this function returns the root mean square gradient at the point an object
    currently sits at, which is the quantity the solver drives to its convergence
    threshold
    """
    grad = np.empty(obj.n_param, dtype=np.float64)
    h_diag = np.empty(obj.n_param, dtype=np.float64)
    obj._update_orbs(np.zeros(obj.n_param, dtype=np.float64), grad, h_diag)
    return np.linalg.norm(grad) / np.sqrt(obj.n_param)


def complex_guess(mf: scf.hf.SCF) -> np.ndarray:
    """
    this function returns the initial guess of a mean-field object with a small
    imaginary part added, which keeps the density matrix Hermitian but makes the
    orbitals complex
    """
    dm = mf.get_init_guess()
    rng = np.random.default_rng(1)
    imag = 0.01 * rng.standard_normal(dm.shape[-2:])
    imag = imag - imag.T
    return dm + 1j * (imag if dm.ndim == 2 else np.array([imag, imag]))


def kramers_pairing_error(molecule: gto.Mole, mo_coeff: np.ndarray) -> float:
    """
    this function returns how far the odd spinors deviate from the time reverses of the
    even ones before them, which Kramers-restricted orbitals are, with the time
    reversal acting on the spinor basis as PySCF maps it
    """
    tr_map = np.asarray(molecule.time_reversal_map())
    signs = np.where(tr_map < 0, -1, 1)
    reversed_even = np.zeros_like(mo_coeff[:, ::2])
    reversed_even[abs(tr_map) - 1] = signs[:, None] * mo_coeff[:, ::2].conj()
    return float(abs(mo_coeff[:, 1::2] - reversed_even).max())


def kramers_rotation(n_occ: int, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    this function returns the unitary rotation of Kramers-paired spinors, the occupied
    before the virtual ones, by the rotations a between the even spinors, their
    conjugates between the odd ones, c of an even occupied spinor into an odd virtual
    one and -conj(c) of an odd occupied spinor into an even virtual one
    """
    n_vir = 2 * a.shape[0]
    block = np.empty((n_vir, n_occ), dtype=np.complex128)
    block[::2, ::2] = a
    block[1::2, 1::2] = a.conj()
    block[1::2, ::2] = c
    block[::2, 1::2] = -c.conj()
    generator = np.zeros(2 * (n_occ + n_vir,), dtype=np.complex128)
    generator[n_occ:, :n_occ] = block
    return scipy.linalg.expm(generator - generator.conj().T)


def singlet_and_triplet_solvers() -> List[Any]:
    """
    this function returns CI solvers for the lowest singlet and the lowest triplet
    state, over which a state average mixes states of different spin
    """
    solvers = [fci.direct_spin1.FCI(mol), fci.direct_spin1.FCI(mol)]
    solvers[1].spin = 2
    return solvers


def solvers_of_different_symmetry() -> List[Any]:
    """
    this function returns symmetry-adapted CI solvers for the lowest state of A1 and
    the lowest state of B1 symmetry, over which a state average mixes states of
    different point-group symmetry
    """
    solvers = [fci.direct_spin1_symm.FCI(mol_sym), fci.direct_spin1_symm.FCI(mol_sym)]
    solvers[0].wfnsym = "A1"
    solvers[1].wfnsym = "B1"
    return solvers


def check_derivatives_match_finite_differences(
    test: unittest.TestCase, obj: Any, curvature: bool = True, **kernel_args: Any
) -> None:
    """
    this function checks that the energy, the gradient and, unless the Hessian is
    approximate, the Hessian linear transformation an orbital update reports are the
    objective the solver is handed at the point the update moves to and its
    derivatives, which a mismatched factor on either derivative would break
    """
    obj.kernel(**kernel_args)
    grad, h_diag = grad_and_h_diag(obj)

    # step away from the stationary point, where the gradient vanishes and would
    # compare equal whatever factor it carried
    np.random.seed(2)
    displacement = 0.05 * (np.random.random(obj.n_param) - 0.5)
    func, hess_x = obj._update_orbs(displacement, grad, h_diag)
    test.assertAlmostEqual(func, obj._func(np.zeros(obj.n_param)), 10)

    x = np.random.random(obj.n_param) - 0.5
    x /= np.linalg.norm(x)
    hx = np.empty(obj.n_param, dtype=np.float64)
    hess_x(x, hx)

    step = 1e-5
    derivative = (obj._func(step * x) - obj._func(-step * x)) / (2 * step)
    test.assertLess(abs(grad @ x - derivative), 1e-5 * abs(derivative))
    if not curvature:
        return

    # the difference formulas are limited by round off in the objective, so both are
    # compared relative to the value they approximate
    step = 1e-4
    second_derivative = (
        obj._func(step * x)
        - 2 * obj._func(np.zeros(obj.n_param))
        + obj._func(-step * x)
    ) / step**2
    test.assertLess(abs(x @ hx - second_derivative), 1e-4 * abs(second_derivative))


def check_large_rotation_is_rejected(test: unittest.TestCase, obj: Any) -> None:
    """
    this function checks that the objective the solver is handed is infinite for a
    step which rotates a pair of orbitals by more than pi / 4, so that the solver
    rejects it, and finite for a rotation just below
    """
    obj.kernel()
    step = np.zeros(obj.n_param, dtype=np.float64)
    step[0] = 1.01 * np.pi / 4
    test.assertEqual(obj._func(step), np.inf)
    step[0] = 0.99 * np.pi / 4
    test.assertTrue(np.isfinite(obj._func(step)))


def hydrogen_cell(basis: str) -> pbcgto.Cell:
    """
    this function returns a periodic cell of a hydrogen molecule in a given basis
    """
    return pbcgto.M(
        atom="H 0 0 0; H 0 0 0.74",
        a=np.eye(3) * 4.0,
        basis=basis,
        pseudo="gth-pade",
        verbose=0,
    )


def run_without_adopted_routines(obj: SecondOrderOTR, **kernel_args: Any) -> None:
    """
    this function runs the kernel of a mean-field object and drops the routines it
    adopted from the extension factories, which work on the orbitals before their
    canonicalization, so that the orbital update and the objective function of the
    object work on its orbitals, which the kernel left canonical
    """
    obj.kernel(**kernel_args)
    obj._drop_adopted_routines()


def move_away_from_the_minimum(obj: SecondOrderOTR, **kernel_args: Any) -> np.ndarray:
    """
    this function runs the kernel of a mean-field object and moves it away from the
    minimum by an orbital update of its own, whose Hessian diagonal it returns, so that
    the orbitals and the Fock matrix are those of a generic point
    """
    run_without_adopted_routines(obj, **kernel_args)
    grad, h_diag = grad_and_h_diag(obj)
    rng = np.random.default_rng(2)
    obj._update_orbs(0.05 * (rng.random(obj.n_param) - 0.5), grad, h_diag)
    return h_diag


def dense_operator(
    apply: Callable[[np.ndarray, np.ndarray], None], n_param: int
) -> np.ndarray:
    """
    this function returns the matrix of a linear map applied in place, as the callback
    functions of the solver apply it, from its action on the unit vectors
    """
    matrix = np.empty((n_param, n_param))
    for i, unit_vector in enumerate(np.eye(n_param)):
        output = np.empty(n_param)
        apply(unit_vector, output)
        matrix[:, i] = output
    return matrix


def reference_transport(
    unpack: Callable[[np.ndarray], np.ndarray],
    pack: Callable[[np.ndarray], np.ndarray],
    non_redundant: np.ndarray,
    geodesic: np.ndarray,
    tangent_vector: np.ndarray,
) -> np.ndarray:
    """
    this function returns the parallel transport of a tangent vector along the geodesic
    of a rotation, the solution exp(L) a of da/dt = L a with L a = -1/2 P [K, A] for
    the matrices K and A that the geodesic and a unpack to and the projection P onto
    the non-redundant rotations, from the exponential of the matrix of L
    """
    generator = unpack(geodesic)

    def generator_action(a: np.ndarray, output: np.ndarray) -> None:
        matrix = unpack(a)
        output[:] = pack(
            np.where(non_redundant, -0.5 * (generator @ matrix - matrix @ generator), 0)
        )

    operator = dense_operator(generator_action, tangent_vector.size)
    return scipy.linalg.expm(operator) @ tangent_vector


def check_trial_gradient(test: unittest.TestCase, obj: Any) -> None:
    """
    this function checks that the gradient rotation QN receives at a trial point,
    right after the objective function evaluated it, is the gradient the orbital
    update to that point returns
    """
    rng = np.random.default_rng(4)
    kappa = 0.05 * (rng.random(obj.n_param) - 0.5)
    obj._func(kappa)
    trial_gradient = obj._trial_gradient(kappa)
    grad, h_diag = grad_and_h_diag(obj)
    obj._update_orbs(kappa, grad, h_diag)
    np.testing.assert_allclose(trial_gradient, grad, rtol=0, atol=1e-10)


def check_no_referrers(test: unittest.TestCase, obj: Any) -> None:
    """
    this function checks that nothing but the frames holding an object refers to it
    after its kernel, since the routines it adopted from the extension factories, the
    functions these call and its settings refer to it weakly, so that it is in no
    reference cycle and freed without the garbage collector once these frames are gone
    """
    referrers = gc.get_referrers(obj)
    test.assertEqual(
        [
            referrer
            for referrer in referrers
            if not isinstance(referrer, types.FrameType)
        ],
        [],
    )


@contextmanager
def capture_warnings(obj: Any) -> Generator[io.StringIO, None, None]:
    """
    this function routes the PySCF log of an object into a buffer, at the verbosity
    that still includes warnings, and discards the copy of every warning PySCF writes
    to the standard error while the log is not the standard output
    """
    log = io.StringIO()
    obj.stdout = log
    obj.verbose = lib.logger.WARN
    with redirect_stderr(io.StringIO()):
        yield log


def check_stability_shape(test: unittest.TestCase, obj: OTR) -> Tuple[bool, np.ndarray]:
    """
    this function runs the stability check of an object, checks that it returns a
    boolean and a direction over the parameters and returns both for further checks
    """
    stable, direction = obj.stability_check()
    test.assertIsInstance(stable, bool)
    test.assertIsInstance(direction, np.ndarray)
    test.assertEqual(direction.shape, (obj.n_param,))
    return stable, direction


class DummySettingsHolder(OTR):
    """
    this class stands in for a converted object carrying settings, so that the
    settings assignment can be tested without constructing a PySCF object
    """

    # the settings a test assigns are arbitrary by design, so they cannot be declared
    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def inherit(self, **kwargs: Any) -> "DummySettingsHolder":
        """
        this function copies settings onto the object and records those whose names
        PySCF also uses as inherited from PySCF, the way a conversion function does
        before the user gets to see the object
        """
        self.__dict__.update(kwargs)
        self._record_inherited_settings()
        return self


class SettingsTests(unittest.TestCase):
    """
    tests for the translation of object attributes into solver and stability settings
    """

    def test_setting_fields(self) -> None:
        # non-assignable settings
        for fields in (solver_setting_fields, stability_setting_fields):
            self.assertNotIn("stability_settings", fields)
            for field in non_setting_fields:
                self.assertNotIn(field, fields)

        # a setting the solver and the stability check share is read from a stability_
        # attribute for the stability check or from its plain name for both, so a new
        # shared setting upstream needs a decision here, whereas any other new setting
        # becomes assignable on its own
        shared_on_purpose = {
            "precond",
            "project",
            "hess_symm",
            "seed",
            "verbose",
        }
        self.assertEqual(
            set(solver_setting_fields) & set(stability_setting_fields),
            set(stability_setting_aliases) | shared_on_purpose,
        )
        for setting, alias in stability_setting_aliases.items():
            self.assertEqual(alias, "stability_" + setting)

    def test_plain_setting_is_copied(self) -> None:
        obj = DummySettingsHolder(n_macro=17, start_trust_radius=0.25)
        settings = SolverSettings()
        assign_solver_settings(obj, settings)
        self.assertEqual(settings.n_macro, 17)
        self.assertAlmostEqual(settings.start_trust_radius, 0.25, 12)

    def test_unset_setting_keeps_solver_default(self) -> None:
        default = SolverSettings()
        settings = SolverSettings()
        assign_solver_settings(DummySettingsHolder(), settings)
        self.assertEqual(settings.n_macro, default.n_macro)
        self.assertAlmostEqual(settings.conv_tol, default.conv_tol, 12)

    def test_inherited_conv_tol_does_not_override_solver_default(self) -> None:
        # PySCF sets conv_tol for its own solver; a value merely inherited from the
        # PySCF object must not silently become the trust region convergence threshold
        obj = DummySettingsHolder().inherit(conv_tol=1e-9)
        settings = SolverSettings()
        assign_solver_settings(obj, settings)
        self.assertAlmostEqual(settings.conv_tol, SolverSettings().conv_tol, 12)

    def test_assigned_conv_tol_overrides_solver_default(self) -> None:
        # also a value equal to the inherited one counts as assigned
        for value in (1e-7, 1e-9):
            with self.subTest(value=value):
                obj = DummySettingsHolder().inherit(conv_tol=1e-9)
                obj.conv_tol = value
                settings = SolverSettings()
                assign_solver_settings(obj, settings)
                self.assertAlmostEqual(settings.conv_tol, value, 12)

    def test_setting_was_set_by_user(self) -> None:
        obj = DummySettingsHolder().inherit(conv_tol=1e-9)
        self.assertFalse(setting_was_set_by_user(obj, "conv_tol"))
        obj.conv_tol = 1e-7
        self.assertTrue(setting_was_set_by_user(obj, "conv_tol"))

        # an attribute that was never recorded as inherited was necessarily assigned
        self.assertFalse(setting_was_set_by_user(obj, "n_macro"))
        obj.n_macro = 5
        self.assertTrue(setting_was_set_by_user(obj, "n_macro"))

    def test_boolean_conv_check_is_dropped(self) -> None:
        # PySCF uses conv_check as a boolean flag whereas the solver expects a
        # convergence check function, so the boolean must not be passed on
        obj = DummySettingsHolder(conv_check=True)
        self.assertFalse(setting_is_assignable(obj, "conv_check"))

        settings = SolverSettings()
        assign_solver_settings(obj, settings)
        self.assertEqual(settings.conv_check, SolverSettings().conv_check)

        # the stability check takes its convergence check function only from
        # stability_conv_check, so the boolean cannot reach it either
        self.assertEqual(
            settings.stability_settings.conv_check, StabilitySettings().conv_check
        )

    def test_callable_conv_check_is_kept(self) -> None:
        def conv_check(residual: np.ndarray, eigval: float) -> bool:
            return False

        obj = DummySettingsHolder(conv_check=conv_check)
        self.assertTrue(setting_is_assignable(obj, "conv_check"))
        settings = SolverSettings()
        assign_solver_settings(obj, settings)
        self.assertIs(settings.conv_check, conv_check)

    def test_stability_settings_use_prefixed_attributes(self) -> None:
        # a setting name shared by the solver and the stability check is taken from a
        # stability specific attribute, so a solver setting cannot leak into the check
        obj = DummySettingsHolder(
            conv_tol=1e-7,
            stability_conv_tol=1e-4,
            n_random_trial_vectors=7,
            stability_n_random_trial_vectors=3,
        )

        settings = SolverSettings()
        assign_solver_settings(obj, settings)
        self.assertAlmostEqual(settings.conv_tol, 1e-7, 12)
        self.assertEqual(settings.n_random_trial_vectors, 7)
        self.assertAlmostEqual(settings.stability_settings.conv_tol, 1e-4, 12)
        self.assertEqual(settings.stability_settings.n_random_trial_vectors, 3)

    def test_solver_setting_does_not_leak_into_stability_check(self) -> None:
        # a convergence check of the solver, which tests the solver's progress, must
        # not decide when the stability check stops either
        def conv_check() -> bool:
            return True

        obj = DummySettingsHolder(
            conv_tol=1e-7, conv_check=conv_check, n_extra_trial_vectors=3
        )
        settings = SolverSettings()
        assign_solver_settings(obj, settings)
        self.assertAlmostEqual(
            settings.stability_settings.conv_tol, StabilitySettings().conv_tol, 12
        )
        self.assertEqual(
            settings.stability_settings.conv_check, StabilitySettings().conv_check
        )
        self.assertEqual(
            settings.stability_settings.n_extra_trial_vectors,
            StabilitySettings().n_extra_trial_vectors,
        )

    def test_stability_settings_assigned_on_their_own(self) -> None:
        # stability_check builds its own settings object rather than filling in the one
        # nested in the solver settings, so the prefixed attributes have to be honoured
        # on that path too
        obj = DummySettingsHolder(conv_tol=1e-7, stability_conv_tol=1e-4, n_iter=17)
        settings = StabilitySettings()
        assign_stability_settings(obj, settings)
        self.assertAlmostEqual(settings.conv_tol, 1e-4, 12)
        self.assertEqual(settings.n_iter, 17)

    def test_messages_go_to_the_pyscf_log(self) -> None:
        # the solver, its stability check, the stability check on its own and the
        # extensions write their messages to the PySCF log of the object
        log = io.StringIO()
        obj = DummySettingsHolder(stdout=log, verbose=lib.logger.WARN)
        settings = SolverSettings()
        assign_solver_settings(obj, settings)
        stability_settings = StabilitySettings()
        assign_stability_settings(obj, stability_settings)
        mo_settings = MOSettings()
        assign_extension_settings(obj, mo_settings, mo_setting_fields)
        for logger in (
            settings.logger,
            settings.stability_settings.logger,
            stability_settings.logger,
            mo_settings.logger,
        ):
            logger("message")
        self.assertEqual(log.getvalue(), 4 * "message\n")

    def test_settings_are_recognized_keys(self) -> None:
        # PySCF warns about attributes it does not recognize, so every assignable
        # setting has to be registered on the class; PySCF collects the keys of every
        # class in the method resolution order, so a driver can register its own on top
        # of those of its base classes
        otr = to_otr(scf.RHF(mol))
        keys = set().union(*(getattr(cls, "_keys", ()) for cls in type(otr).__mro__))
        for setting in solver_setting_fields + list(stability_setting_aliases.values()):
            self.assertIn(setting, keys)

    def test_only_unknown_attributes_are_reported(self) -> None:
        # every kernel reports attributes that none of the classes of the object
        # registers, such as a misspelled setting, as PySCF's kernels do, so the
        # attributes the kernel sets itself have to be registered; PySCF joins the keys
        # of all these classes and skips class attributes and names starting with an
        # underscore
        for obj in (
            to_otr(scf.RHF(mol)),
            to_otr(scf.ROHF(mol_open)),
            to_otr(scf.UHF(mol_open)),
            to_otr(x2c.UHF(mol)),
            BoysOTR(mol, occ_orbs),
            to_otr(mcscf.CASSCF(mf, 4, 4)),
        ):
            with self.subTest(cls=type(obj).__name__):
                obj.kernel()
                keys = set().union(
                    *(getattr(cls, "_keys", ()) for cls in type(obj).__mro__)
                )
                public = {name for name in vars(obj) if not name.startswith("_")}
                self.assertFalse(public - keys - set(dir(type(obj))))

                with capture_warnings(obj) as log:
                    obj.n_marco = 1
                    obj.kernel()
                    self.assertIn("n_marco", log.getvalue())

    def test_kernel_returns_without_convergence(self) -> None:
        # like PySCF's own kernels, a kernel that runs out of macro iterations returns
        # and reports that it did not converge, which the solver also writes to the
        # PySCF log, whereas other errors are raised
        def failing_conv_check() -> bool:
            raise ValueError

        for obj in (
            to_otr(scf.RHF(mol)),
            BoysOTR(mol, occ_orbs),
            to_otr(mcscf.CASSCF(mf, 4, 4)),
        ):
            with self.subTest(cls=type(obj).__name__), capture_warnings(obj) as log:
                obj.n_macro = 1
                obj.kernel()
                self.assertFalse(obj.converged)
                self.assertIn("has not converged", log.getvalue())
                obj.conv_check = failing_conv_check
                self.assertRaises(RuntimeError, obj.kernel)


class ConversionTests(unittest.TestCase):
    """
    tests for the conversion of PySCF objects into their OpenTrustRegion counterparts
    """

    def test_to_otr_selects_the_matching_class(self) -> None:
        # the converted object must remain usable as the PySCF object it came from,
        # since PySCF routines are typed against those classes, takes over its
        # attributes and is returned as it is when converted again
        for obj, cls in (
            (scf.RHF(mol), RHFOTR),
            (scf.ROHF(mol_open), ROHFOTR),
            (scf.UHF(mol_open), UHFOTR),
            (scf.GHF(mol_open), GHFOTR),
            (dft.RKS(mol, xc="b3lyp"), RHFOTR),
            (dft.UKS(mol_open, xc="b3lyp"), UHFOTR),
            (dft.GKS(mol_open, xc="b3lyp"), GHFOTR),
            (x2c.UHF(mol), X2CUHFOTR),
            (lo.Boys(mol, occ_orbs), BoysOTR),
            (lo.PipekMezey(mol, occ_orbs, pop_method="mulliken"), PipekMezeyOTR),
            (lo.EdmistonRuedenberg(mol, occ_orbs), EdmistonRuedenbergOTR),
            (mcscf.CASSCF(mf, 4, 4), CASSCFOTR),
        ):
            with self.subTest(obj=obj.__class__.__name__):
                otr = to_otr(obj)
                self.assertIsInstance(otr, cls)
                self.assertIsInstance(otr, type(obj))
                for name, value in vars(obj).items():
                    self.assertIs(vars(otr)[name], value)
                self.assertIs(to_otr(otr), otr)

    def test_to_otr_rejects_unsupported_objects(self) -> None:
        # neither four-component calculations nor those at k-points are supported, nor
        # the complex Pipek-Mezey localizer, CASCI, which optimizes no orbitals,
        # unrestricted CASSCF, for which PySCF has no second-order solver, or other
        # objects
        cell = pbcgto.M(
            atom="H 0 0 0; H 0 0 0.74",
            a=np.eye(3) * 4.0,
            basis="gth-szv",
            pseudo="gth-pade",
            verbose=0,
        )
        for obj in (
            scf.DHF(mol),
            pbcscf.KRHF(cell, cell.make_kpts([2, 1, 1])),
            lo.PipekMezeyComplex(mol, occ_orbs),
            mcscf.CASCI(mf, 4, 4),
            mcscf.UCASSCF(scf.UHF(mol_open), 4, 4),
            mol,
        ):
            with self.subTest(obj=obj.__class__.__name__):
                self.assertRaises(TypeError, to_otr, obj)


class SCFTests(unittest.TestCase):
    """
    tests for the Hartree-Fock and Kohn-Sham drivers: the converged energies, the
    derivatives and the stability check
    """

    def _check(
        self,
        mf_in: scf.hf.SCF,
        ref_mf: scf.hf.SCF,
        dm0: Optional[np.ndarray] = None,
        **settings: Any,
    ) -> SecondOrderOTR:
        ref_newton = ref_mf.newton()
        ref = ref_newton.kernel()
        otr = to_otr(mf_in)
        for name, value in settings.items():
            setattr(otr, name, value)
        e_tot = otr.kernel(dm0=dm0)
        check_no_referrers(self, otr)

        self.assertTrue(otr.converged)
        self.assertAlmostEqual(e_tot, ref, 8)
        self.assertAlmostEqual(otr.e_tot, ref, 8)
        self.assertEqual(lib.chkfile.load(otr.chkfile, "scf/e_tot"), e_tot)

        # the orbitals are ordered as PySCF orders them, by energy with the occupied
        # ones first, which the methods building on them rely on
        self.assertFalse(np.signbit(otr.mo_occ).any())
        np.testing.assert_array_equal(otr.mo_occ, ref_newton.mo_occ)
        np.testing.assert_allclose(otr.mo_energy, ref_newton.mo_energy, atol=1e-4)

        # the orbitals must remain a valid, orthonormal set in the AO basis
        ovlp = otr.get_ovlp()
        mo_coeff = np.asarray(otr.mo_coeff)
        if mo_coeff.ndim == 2:
            mo_coeff = mo_coeff[None]
        for coeff in mo_coeff:
            self.assertAlmostEqual(
                abs(coeff.conj().T @ ovlp @ coeff - np.eye(coeff.shape[1])).max(),
                0.0,
                8,
            )

        # check if the gradient is converged
        self.assertLess(gradient_rms(otr), SolverSettings().conv_tol)
        return otr

    def test_stability_is_a_setting(self) -> None:
        # stability is a boolean solver setting, so the PySCF method of that name has
        # to give way; the OpenTrustRegion equivalent is stability_check
        otr = to_otr(scf.RHF(mol))
        self.assertFalse(callable(otr.stability))
        self.assertTrue(callable(otr.stability_check))

    def test_rhf(self) -> None:
        self._check(scf.RHF(mol), scf.RHF(mol))

    def test_rhf_in_the_oao_basis(self) -> None:
        self._check(scf.RHF(mol), scf.RHF(mol), oao=True)

    def test_rhf_with_symmetry(self) -> None:
        self._check(scf.RHF(mol_sym), scf.RHF(mol_sym))

    def test_rohf(self) -> None:
        self._check(scf.ROHF(mol_open), scf.ROHF(mol_open))

    def test_uhf(self) -> None:
        self._check(scf.UHF(mol_open), scf.UHF(mol_open))

    def test_uhf_in_the_oao_basis(self) -> None:
        self._check(scf.UHF(mol_open), scf.UHF(mol_open), oao=True)

    def test_uhf_with_symmetry(self) -> None:
        self._check(scf.UHF(mol_open_sym), scf.UHF(mol_open_sym))

    def test_ghf(self) -> None:
        self._check(scf.GHF(mol_open), scf.GHF(mol_open))

    def test_ghf_with_symmetry(self) -> None:
        self._check(scf.GHF(mol_open_sym), scf.GHF(mol_open_sym))

    def test_rks(self) -> None:
        self._check(dft.RKS(mol, xc="b3lyp"), dft.RKS(mol, xc="b3lyp"))

    def test_uks(self) -> None:
        self._check(dft.UKS(mol_open, xc="b3lyp"), dft.UKS(mol_open, xc="b3lyp"))

    def test_gks(self) -> None:
        self._check(dft.GKS(mol_open, xc="b3lyp"), dft.GKS(mol_open, xc="b3lyp"))

    def test_complex_orbitals_return_to_the_real_minimum(self) -> None:
        # from complex orbitals, the solver optimizes the real and imaginary parts of
        # the rotations and returns to the real minimum
        for cls, molecule in (
            (scf.RHF, mol),
            (scf.RHF, mol_sym),
            (scf.ROHF, mol_open),
            (scf.UHF, mol_open),
            (scf.GHF, mol),
        ):
            with self.subTest(cls=cls.__name__, symmetry=molecule.symmetry):
                otr = self._check(
                    cls(molecule), cls(molecule), dm0=complex_guess(cls(molecule))
                )
                self.assertTrue(otr.complex_orbitals)

    @unittest.skipIf(dhf.zquatev is None, "Kramers-restricted X2C requires zquatev")
    def test_kramers_restricted_x2c(self) -> None:
        # the reference is PySCF's SCF rather than its second-order solver, which turns
        # both spinors of a Kramers pair alike; the parameters are the rotations A and
        # C of every combination of a virtual and an occupied Kramers pair, and the
        # canonical orbitals the kernel returns remain Kramers pairs, also with
        # rotation QN, whose static part of the Hessian turns both spinors of a pair
        ref = x2c.RHF(mol)
        ref.kernel()
        for qn_method in (None,) + rotation_qn_methods:
            with self.subTest(qn_method=qn_method):
                otr = to_otr(x2c.RHF(mol))
                otr.qn_method = qn_method
                self.assertIsInstance(otr, X2CRHFOTR)
                self.assertAlmostEqual(otr.kernel(), ref.e_tot, 8)
                self.assertLess(gradient_rms(otr), SolverSettings().conv_tol)
                self.assertLess(kramers_pairing_error(mol, otr.mo_coeff), 1e-10)
                n_occ_pairs = mol.nelectron // 2
                n_vir_pairs = otr.mo_coeff.shape[1] // 2 - n_occ_pairs
                self.assertEqual(otr.n_param, 4 * n_vir_pairs * n_occ_pairs)
                stable, _ = check_stability_shape(self, otr)
                self.assertTrue(stable)

    @unittest.skipIf(dhf.zquatev is None, "Kramers-restricted X2C requires zquatev")
    def test_kramers_restricted_x2c_derivatives(self) -> None:
        # the step the derivatives are checked at keeps every odd spinor the time
        # reverse of the even one before it, as only the conjugate rotation of the
        # partners does for complex rotations
        otr = to_otr(x2c.RHF(mol))
        check_derivatives_match_finite_differences(self, otr)
        self.assertLess(kramers_pairing_error(mol, otr.mo_coeff), 1e-10)

    @unittest.skipIf(dhf.zquatev is None, "Kramers-restricted X2C requires zquatev")
    def test_kramers_restricted_x2c_returns_from_a_perturbation(self) -> None:
        # from the minimum perturbed by complex rotations of both kinds, which a solver
        # lacking either could not undo, the solver returns to the minimum, with the
        # orbitals Kramers pairs throughout
        ref = x2c.RHF(mol)
        ref.kernel()
        n_occ = mol.nelectron
        n_vir_pairs = (ref.mo_coeff.shape[1] - n_occ) // 2
        rng = np.random.default_rng(3)
        a, c = (
            0.05
            * (
                rng.standard_normal((n_vir_pairs, n_occ // 2))
                + 1j * rng.standard_normal((n_vir_pairs, n_occ // 2))
            )
            for _ in range(2)
        )
        perturbed = ref.mo_coeff @ kramers_rotation(n_occ, a, c)
        otr = to_otr(x2c.RHF(mol))
        e_tot = otr.kernel(mo_coeff=perturbed, mo_occ=ref.mo_occ)
        self.assertAlmostEqual(e_tot, ref.e_tot, 8)
        self.assertLess(kramers_pairing_error(mol, otr.mo_coeff), 1e-10)

    def test_x2c(self) -> None:
        otr = self._check(x2c.UHF(mol), x2c.UHF(mol))
        self.assertTrue(otr.complex_orbitals)
        stable, _ = check_stability_shape(self, otr)
        self.assertTrue(stable)

    def test_spin_orbit_x2c_in_the_ghf_basis(self) -> None:
        otr = self._check(scf.GHF(mol).x2c(), scf.GHF(mol).x2c())
        self.assertTrue(otr.complex_orbitals)

    def test_extensions_reject_orbitals_they_do_not_support(self) -> None:
        # OAO, density QN and S-GEK work on the real orbitals of RHF, ROHF and UHF, so
        # they are rejected for GHF, X2C and complex orbitals, OAO and density QN also
        # for ROHF, while rotation QN, which wraps the orbital update of every driver,
        # is only rejected in the OAO basis, since it takes the static part of the
        # Hessian of the MO basis
        for name, mf_in, settings, kernel_args in (
            ("GHF", scf.GHF(mol_open), {"oao": True}, {}),
            ("ROHF", scf.ROHF(mol_open), {"qn_method": "arh"}, {}),
            ("X2C", x2c.UHF(mol), {"s_gek": True}, {}),
            (
                "complex",
                scf.RHF(mol),
                {"qn_method": "arh"},
                {"dm0": complex_guess(scf.RHF(mol))},
            ),
            ("OAO", scf.RHF(mol), {"qn_method": "sr1", "oao": True}, {}),
        ):
            with self.subTest(case=name):
                otr = to_otr(mf_in)
                for setting, value in settings.items():
                    setattr(otr, setting, value)
                self.assertRaises(NotImplementedError, otr.kernel, **kernel_args)

    def test_rotation_qn(self) -> None:
        # rotation QN wraps the orbital update of every mean-field driver, starting
        # from the static part of the Hessian, with the gradient at every trial point
        # and, for ROHF, the parallel transport of its flag manifold, and reaches the
        # minimum with either updating scheme, also with the Hessian diagonal and the
        # identity transport the two options select instead
        for name, make, kernel_args in (
            ("RHF", lambda: scf.RHF(mol), {}),
            ("RHF symmetry", lambda: scf.RHF(mol_sym), {}),
            ("ROHF", lambda: scf.ROHF(mol_open), {}),
            ("ROHF symmetry", lambda: scf.ROHF(mol_open_sym), {}),
            ("UHF", lambda: scf.UHF(mol_open), {}),
            ("GHF", lambda: scf.GHF(mol_open), {}),
            ("complex RHF", lambda: scf.RHF(mol), {"dm0": complex_guess(scf.RHF(mol))}),
            (
                "complex ROHF",
                lambda: scf.ROHF(mol_open),
                {"dm0": complex_guess(scf.ROHF(mol_open))},
            ),
            ("X2C", lambda: x2c.UHF(mol), {}),
            ("GHF X2C", lambda: scf.GHF(mol).x2c(), {}),
        ):
            for qn_method in rotation_qn_methods:
                with self.subTest(case=name, qn_method=qn_method):
                    self._check(make(), make(), qn_method=qn_method, **kernel_args)
        with self.subTest(case="options"):
            self._check(
                scf.ROHF(mol_open),
                scf.ROHF(mol_open),
                qn_method="sr1",
                static_hess=False,
                parallel_transport=False,
            )

    def test_rotation_qn_callbacks(self) -> None:
        # the mean-field drivers supply the static part of the Hessian and its inverse
        # unless static_hess is unset, the objective function with the
        # gradient at a trial point, and for ROHF the parallel transport unless
        # parallel_transport is unset
        for mf_in, transport in ((scf.RHF(mol), False), (scf.ROHF(mol_open), True)):
            for static_hess, parallel_transport in product((True, False), repeat=2):
                with self.subTest(
                    cls=mf_in.__class__.__name__,
                    static_hess=static_hess,
                    parallel_transport=parallel_transport,
                ):
                    otr = to_otr(mf_in)
                    otr.static_hess = static_hess
                    otr.parallel_transport = parallel_transport
                    callbacks = otr._rotation_qn_callbacks(otr._func)
                    expected = ["obj_func_grad"]
                    if static_hess:
                        expected = ["init_hess", "init_precond"] + expected
                    if transport and parallel_transport:
                        expected.append("transport")
                    self.assertEqual(list(callbacks), expected)

    def test_static_hessian(self) -> None:
        # the static part of the Hessian, which neglects the response of the potential
        # and for ROHF the terms of the gradient, is symmetric and has PySCF's Hessian
        # diagonal, which neglects these too, as its diagonal, and its preconditioner
        # inverts it shifted by a level shift, for ROHF without the coupling of the
        # virtual-closed rotations to the virtual-open and open-closed ones, whose
        # blocks it inverts exactly
        for name, mf_in, kernel_args in (
            ("RHF", scf.RHF(mol), {}),
            ("RHF symmetry", scf.RHF(mol_sym), {}),
            ("ROHF", scf.ROHF(mol_open), {}),
            ("ROHF symmetry", scf.ROHF(mol_open_sym), {}),
            ("UHF", scf.UHF(mol_open), {}),
            ("GHF", scf.GHF(mol_open), {}),
            ("complex RHF", scf.RHF(mol), {"dm0": complex_guess(scf.RHF(mol))}),
            (
                "complex ROHF",
                scf.ROHF(mol_open),
                {"dm0": complex_guess(scf.ROHF(mol_open))},
            ),
            ("X2C", x2c.UHF(mol), {}),
            ("periodic UHF", pbcscf.UHF(hydrogen_cell("gth-dzv")), {}),
        ):
            with self.subTest(case=name):
                otr = to_otr(mf_in)
                h_diag = move_away_from_the_minimum(otr, **kernel_args)
                hessian = dense_operator(otr._static_hess_x, otr.n_param)
                np.testing.assert_allclose(hessian, hessian.T, rtol=0, atol=1e-12)
                np.testing.assert_allclose(np.diag(hessian), h_diag, rtol=0, atol=1e-10)

                inverted = hessian
                if isinstance(otr, ROHFOTR):
                    rows, cols = np.nonzero(otr.rot_matrix_mask)
                    kinds = np.tile(
                        otr.open_idx[rows] + 2 * otr.open_idx[cols],
                        1 + otr.complex_orbitals,
                    )
                    inverted = np.where(kinds[:, None] == kinds, hessian, 0)
                mu = np.linalg.eigvalsh(inverted)[0] - 1.0
                precond = dense_operator(
                    lambda r, output: otr._static_precond(r, mu, output), otr.n_param
                )
                np.testing.assert_allclose(
                    precond @ (inverted - mu * np.eye(otr.n_param)),
                    np.eye(otr.n_param),
                    rtol=0,
                    atol=1e-10,
                )

    def test_trial_gradient(self) -> None:
        # the gradient at a trial point, which rotation QN takes from the effective
        # potential the objective function left cached, is that of the orbital update
        for name, mf_in, kernel_args in (
            ("RHF", scf.RHF(mol), {}),
            ("ROHF", scf.ROHF(mol_open), {}),
            ("UHF", scf.UHF(mol_open), {}),
            ("GHF", scf.GHF(mol_open), {}),
            ("complex RHF", scf.RHF(mol), {"dm0": complex_guess(scf.RHF(mol))}),
            ("X2C", x2c.UHF(mol), {}),
            ("RKS", dft.RKS(mol), {}),
            ("periodic UHF", pbcscf.UHF(hydrogen_cell("gth-dzv")), {}),
        ):
            with self.subTest(case=name):
                otr = to_otr(mf_in)
                run_without_adopted_routines(otr, **kernel_args)
                check_trial_gradient(self, otr)

    def test_rohf_transport(self) -> None:
        # the transport of ROHF is the parallel transport of the flag manifold of the
        # closed, open and virtual orbitals, also for complex rotations, whereas for
        # the rotations between occupied and virtual orbitals of RHF, whose commutators
        # are all redundant, it would be the identity
        for name, kernel_args in (
            ("real", {}),
            ("complex", {"dm0": complex_guess(scf.ROHF(mol_open))}),
        ):
            with self.subTest(case=name):
                otr = to_otr(scf.ROHF(mol_open))
                otr.kernel(**kernel_args)
                rng = np.random.default_rng(5)
                geodesic, tangent_vector = 0.3 * rng.standard_normal((2, otr.n_param))
                transported = tangent_vector.copy()
                otr._transport(geodesic, transported)
                np.testing.assert_allclose(
                    transported,
                    reference_transport(
                        lambda a: otr._unpack(otr._rotation(a)),
                        lambda matrix: otr._parameters(matrix[otr.rot_matrix_mask]),
                        otr.rot_matrix_mask | otr.rot_matrix_mask.T,
                        geodesic,
                        tangent_vector,
                    ),
                    rtol=0,
                    atol=1e-12,
                )
                self.assertGreater(np.abs(transported - tangent_vector).max(), 1e-3)

        rhf = to_otr(scf.RHF(mol))
        rhf.kernel()
        rng = np.random.default_rng(6)
        generator, tangent = (
            rhf._unpack(0.3 * rng.standard_normal(rhf.n_param)) for _ in range(2)
        )
        non_redundant = rhf.rot_matrix_mask | rhf.rot_matrix_mask.T
        np.testing.assert_allclose(
            transport_rotation(generator, tangent, non_redundant), tangent, atol=1e-15
        )

    def test_linear_dependencies(self) -> None:
        # PySCF removes the linear dependencies of a basis, here at a threshold raised
        # so that one function goes, which leaves fewer orbitals than atomic orbitals
        with mock.patch.object(scf.hf, "overlap_zero_eigenvalue_threshold", 0.1):
            for cls, molecule in (
                (scf.RHF, mol),
                (scf.ROHF, mol_open),
                (scf.UHF, mol_open),
                (scf.GHF, mol_open),
            ):
                with self.subTest(cls=cls.__name__):
                    otr = self._check(cls(molecule), cls(molecule))
                    n_orbitals = otr.mo_coeff.shape[-1]
                    self.assertLess(n_orbitals, (1 + otr.istype("GHF")) * molecule.nao)

    def test_solvent_model(self) -> None:
        self._check(scf.RHF(mol).ddCOSMO(), scf.RHF(mol).ddCOSMO())

    def test_periodic_gamma_point(self) -> None:
        cell = hydrogen_cell("gth-szv")
        self._check(pbcscf.RHF(cell), pbcscf.RHF(cell))

        # rotation QN wraps the orbital update of the periodic drivers too, here with
        # several rotations, along which its model is applied
        cell_dzv = hydrogen_cell("gth-dzv")
        otr = self._check(pbcscf.RHF(cell_dzv), pbcscf.RHF(cell_dzv), qn_method="sr1")
        self.assertGreater(otr.n_hess_x, 0)

        # PySCF evaluates the responses of periodic systems only at the orbitals of the
        # object, whereas density QN evaluates them at a density
        otr = to_otr(pbcscf.RHF(cell))
        otr.qn_method = "arh"
        self.assertRaises(NotImplementedError, otr.kernel)

    def test_scanners_follow_the_geometry(self) -> None:
        # a geometry optimization or a scan calls the kernel through a scanner, which
        # hands it the density matrix of the previous geometry
        mol_displaced = gto.M(
            atom="O 0 0 0; H 0 -0.78 0.60; H 0 0.78 0.60", basis="6-31g", verbose=0
        )
        ref = scf.RHF(mol).newton().as_scanner()
        otr = to_otr(scf.RHF(mol)).as_scanner()
        ref(mol)
        otr(mol)
        self.assertAlmostEqual(otr(mol_displaced), ref(mol_displaced), 8)

        grad_ref = scf.RHF(mol).nuc_grad_method().as_scanner()
        grad_otr = to_otr(scf.RHF(mol)).nuc_grad_method().as_scanner()
        grad_ref(mol)
        grad_otr(mol)
        e_ref, g_ref = grad_ref(mol_displaced)
        e_otr, g_otr = grad_otr(mol_displaced)
        self.assertAlmostEqual(e_otr, e_ref, 8)
        self.assertAlmostEqual(abs(g_otr - g_ref).max(), 0.0, 5)

    def test_symmetry_restricts_the_parameter_count(self) -> None:
        # symmetry forbidden rotations must be excluded from the parameter vector
        otr = to_otr(scf.RHF(mol))
        otr.kernel()
        otr_sym = to_otr(scf.RHF(mol_sym))
        otr_sym.kernel()
        self.assertLess(otr_sym.n_param, otr.n_param)

    def test_symmetry_restricts_an_object_that_is_not_symmetry_adapted(self) -> None:
        # an object of a class that is not symmetry-adapted on a molecule with
        # symmetry, as one built before the symmetry of its molecule was switched on,
        # is optimized in the rotations symmetry allows, as one of the symmetry-adapted
        # class is; PySCF's Newton solver stops short for such an object itself, so the
        # reference is the symmetry-adapted one
        for plain, adapted, molecule in (
            (scf.hf.RHF, scf.RHF, mol_sym),
            (scf.rohf.ROHF, scf.ROHF, mol_open_sym),
            (scf.uhf.UHF, scf.UHF, mol_open_sym),
        ):
            with self.subTest(cls=plain.__name__):
                otr = self._check(plain(molecule), adapted(molecule))
                otr_adapted = to_otr(adapted(molecule))
                otr_adapted.kernel()
                self.assertEqual(otr.n_param, otr_adapted.n_param)

    def test_symmetry_can_forbid_every_rotation(self) -> None:
        # in the minimal basis, the orbitals of the carbon atom other than 1s and 2s
        # each belong to an irrep of their own, so symmetry forbids every
        # occupied-virtual rotation of the triplet and the kernel keeps the orbitals it
        # starts from
        carbon = gto.M(atom="C", basis="sto-3g", spin=2, symmetry=True, verbose=0)
        for cls in (scf.ROHF, scf.UHF):
            with self.subTest(cls=cls.__name__):
                ref = cls(carbon).newton().kernel()
                otr = to_otr(cls(carbon))
                self.assertAlmostEqual(otr.kernel(), ref, 8)
                self.assertEqual(otr.n_param, 0)
                stable, _ = check_stability_shape(self, otr)
                self.assertTrue(stable)

    def test_counters_record_the_work_done(self) -> None:
        otr = to_otr(scf.RHF(mol))
        self.assertEqual(otr.n_update_orbs, 0)
        self.assertEqual(otr.n_hess_x, 0)
        otr.kernel()
        self.assertGreater(otr.n_update_orbs, 0)
        self.assertGreater(otr.n_hess_x, 0)

    def test_derivatives_match_finite_differences(self) -> None:
        # the gradient and Hessian PySCF provides for GHF are doubled as those of the
        # restricted classes are; PySCF's ROHF Hessian lacks the terms proportional to
        # the gradient, so only the gradient is compared for ROHF away from the
        # stationary point
        for mf_in, curvature in (
            (scf.RHF(mol), True),
            (scf.RHF(mol_sym), True),
            (scf.ROHF(mol_open), False),
            (scf.GHF(mol_open), True),
        ):
            with self.subTest(cls=mf_in.__class__.__name__):
                check_derivatives_match_finite_differences(
                    self, to_otr(mf_in), curvature=curvature
                )

    def test_complex_derivatives_match_finite_differences(self) -> None:
        # the real and imaginary parts of PySCF's gradient and Hessian are the
        # derivatives with respect to those of the rotations, doubled as for real
        # orbitals; for ROHF, whose Hessian is approximate, only the gradient is
        # compared
        for name, mf_in, kernel_args in (
            ("RHF", scf.RHF(mol), {"dm0": complex_guess(scf.RHF(mol))}),
            (
                "RHF symmetry",
                scf.RHF(mol_sym),
                {"dm0": complex_guess(scf.RHF(mol_sym))},
            ),
            ("ROHF", scf.ROHF(mol_open), {"dm0": complex_guess(scf.ROHF(mol_open))}),
            ("UHF", scf.UHF(mol_open), {"dm0": complex_guess(scf.UHF(mol_open))}),
            ("GHF", scf.GHF(mol_open), {"dm0": complex_guess(scf.GHF(mol_open))}),
            ("X2C", x2c.UHF(mol), {}),
            ("GHF X2C", scf.GHF(mol).x2c(), {}),
        ):
            with self.subTest(case=name):
                otr = to_otr(mf_in)
                check_derivatives_match_finite_differences(
                    self, otr, curvature=name != "ROHF", **kernel_args
                )
                self.assertTrue(otr.complex_orbitals)
                self.assertEqual(otr.n_param, 2 * np.count_nonzero(otr.kappa_mask))

    def test_large_rotation_is_rejected(self) -> None:
        check_large_rotation_is_rejected(self, to_otr(scf.RHF(mol)))

    def test_large_complex_rotation_is_rejected(self) -> None:
        # the limit applies to the modulus of a complex rotation, which exceeds pi / 4
        # with real and imaginary parts below it
        otr = to_otr(x2c.UHF(mol))
        otr.kernel()
        n_pairs = otr.n_param // 2
        step = np.zeros(otr.n_param, dtype=np.float64)
        step[0] = step[n_pairs] = 0.6
        self.assertEqual(otr._func(step), np.inf)
        step[0] = step[n_pairs] = 0.5
        self.assertTrue(np.isfinite(otr._func(step)))

    def test_unstable_solution_is_detected(self) -> None:
        # a stretched singlet has a lower energy broken symmetry UHF solution, so the
        # restricted solution is a saddle point in the unrestricted parameters
        mol_stretched = gto.M(atom="H 0 0 0; H 0 0 3.0", basis="6-31g", verbose=0)
        rhf = scf.RHF(mol_stretched).run()

        # the kernel adopts the routines of the MO or the OAO extension, whose
        # extension objects keep the orbitals of the kernel
        for oao in (False, True):
            with self.subTest(oao=oao):
                otr = to_otr(scf.UHF(mol_stretched))
                otr.oao = oao
                e_tot = otr.kernel()

                # the solver escapes the restricted solution on its own, so its own
                # result is a minimum
                self.assertLess(e_tot, rhf.e_tot - 1e-3)
                stable, _ = check_stability_shape(self, otr)
                self.assertTrue(stable)

                # a copy checks its orbitals with its own routines, since the
                # extension objects of the routines the kernel adopted refer to the
                # object whose kernel ran
                n_hess_x_stability = otr.n_hess_x_stability
                stable, _ = check_stability_shape(self, otr.copy())
                self.assertTrue(stable)
                self.assertEqual(otr.n_hess_x_stability, n_hess_x_stability)

                # placing the same object back at the restricted solution must make the
                # check report the instability, along with a normalized direction out
                # of the saddle; the check drops the routines the kernel adopted, whose
                # extension objects are freed with them without the garbage collector
                adopted = weakref.ref(otr._func)
                gc.disable()
                try:
                    otr.mo_coeff = np.array([rhf.mo_coeff, rhf.mo_coeff])
                    stable, direction = check_stability_shape(self, otr)
                    freed = adopted() is None
                finally:
                    gc.enable()
                self.assertTrue(freed)
                self.assertFalse(stable)
                self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, 8)

                # the check examined the restricted solution, and its direction is the
                # one of lowest curvature there
                grad, h_diag = grad_and_h_diag(otr)
                func, hess_x = otr._update_orbs(np.zeros(otr.n_param), grad, h_diag)
                self.assertAlmostEqual(func, rhf.e_tot, 9)
                hess = np.empty((otr.n_param, otr.n_param))
                for column, unit_vector in zip(hess.T, np.eye(otr.n_param)):
                    hess_x(unit_vector, column)
                self.assertAlmostEqual(
                    float(direction @ hess @ direction),
                    float(np.linalg.eigvalsh(hess)[0]),
                    6,
                )

        # a new object whose kernel never ran must report the instability at the
        # restricted solution too
        new_otr = to_otr(scf.UHF(mol_stretched))
        new_otr.mo_occ = otr.mo_occ
        new_otr.mo_coeff = np.array([rhf.mo_coeff, rhf.mo_coeff])
        stable, direction = check_stability_shape(self, new_otr)
        self.assertFalse(stable)
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, 8)

    def test_stability_check_reports_a_point_that_is_not_stationary(self) -> None:
        # the curvature tells a minimum from a saddle point only at a stationary point,
        # so the check warns at the orbitals of the core Hamiltonian guess but not at
        # the orbitals the kernel converged to
        otr = to_otr(scf.RHF(mol))
        mo_energy, otr.mo_coeff = otr.eig(otr.get_hcore(), otr.get_ovlp())
        otr.mo_occ = otr.get_occ(mo_energy, otr.mo_coeff)
        with capture_warnings(otr) as log:
            otr.stability_check()
            self.assertIn("not stationary", log.getvalue())

            log.seek(0)
            log.truncate()
            otr.kernel()
            otr.stability_check()
            self.assertNotIn("not stationary", log.getvalue())

    def test_stability_check_examines_the_current_occupations(self) -> None:
        # the check examines the density of the current occupations and their
        # occupied-virtual rotations, which with symmetry are those between orbitals of
        # the same irrep, also when the occupations are assigned after the kernel; ROHF
        # adopts no routines from the extension factories
        for cls, molecule in product((scf.RHF, scf.ROHF), (mol, mol_sym)):
            with self.subTest(cls=cls.__name__, symmetry=molecule.symmetry):
                otr = to_otr(cls(molecule))
                otr.kernel()

                # move the highest occupied pair to the lowest virtual orbital
                mo_occ = otr.mo_occ.copy()
                homo = np.flatnonzero(mo_occ > 0)[-1]
                lumo = np.flatnonzero(mo_occ == 0)[0]
                mo_occ[[homo, lumo]] = mo_occ[[lumo, homo]]
                otr.mo_occ = mo_occ
                check_stability_shape(self, otr)

                # the energy and the gradient are those of the current occupations
                grad, h_diag = grad_and_h_diag(otr)
                func, _ = otr._update_orbs(np.zeros(otr.n_param), grad, h_diag)

                ref = cls(molecule)
                dm = ref.make_rdm1(otr.mo_coeff, mo_occ)
                ref_grad = ref.get_grad(otr.mo_coeff, mo_occ, ref.get_fock(dm=dm))
                if molecule.symmetry:
                    occ = mo_occ > 0
                    orbsym = ref.get_orbsym(otr.mo_coeff)
                    ref_grad = ref_grad[(orbsym[~occ][:, None] == orbsym[occ]).ravel()]
                self.assertAlmostEqual(func, ref.energy_tot(dm), 9)
                np.testing.assert_allclose(grad, 2 * ref_grad, atol=1e-8)

    def test_energy_option_assigned_after_conversion_is_reported(self) -> None:
        # the PySCF object passed to to_otr evaluates the energy and the gradient, so
        # an option defining the energy that is assigned afterwards reaches at most the
        # Hessian, which the solver and the stability check warn about
        otr = to_otr(dft.UKS(mol_open, xc="b3lyp"))
        with capture_warnings(otr) as log:
            # a replaced grid for the nonlocal correlation, which b3lyp does not use,
            # leaves the optimization unaffected
            otr.nlcgrids = dft.gen_grid.Grids(mol_open)
            otr.kernel()
            self.assertIn("nlcgrids of this object differs", log.getvalue())

            log.seek(0)
            log.truncate()
            otr.xc = "pbe"
            otr.stability_check()
            self.assertIn("xc, nlcgrids of this object differ", log.getvalue())

    def test_energy_option_reaching_the_energy_is_not_reported(self) -> None:
        # an option assigned before the conversion, a functional name differing only in
        # case, a grid changed in place, which both objects share, and the grid PySCF
        # only uses for the Hessian all reach the energy or are meant for the Hessian
        uks = dft.UKS(mol_open)
        uks.xc = "b3lyp"
        otr = to_otr(uks)
        with capture_warnings(otr) as log:
            otr.xc = "B3LYP"
            otr.grids.level = 2
            # the secondary grid only exists in PySCF releases after 2.14
            if hasattr(otr, "set_second_grids"):
                otr.set_second_grids(2)
            otr.kernel()
            otr.stability_check()
            self.assertNotIn("to_otr", log.getvalue())


class LocalizerTests(unittest.TestCase):
    """
    tests for the orbital localizers, whose cost functions differ in whether they are
    minimized or maximized
    """

    # the sign that turns each PySCF cost function into a minimization objective
    localizers = (
        (BoysOTR, lo.Boys, 1.0),
        (PipekMezeyOTR, lo.PipekMezey, -1.0),
        (EdmistonRuedenbergOTR, lo.EdmistonRuedenberg, -1.0),
    )

    def _objective(self, cls: type, mo_coeff: np.ndarray, sign: float) -> float:
        return sign * cls(mol, mo_coeff).cost_function(np.eye(mo_coeff.shape[1]))

    def test_localizer_stability_is_a_setting(self) -> None:
        # the boolean solver setting shadows the stability() method the pyscf.lo class
        # provides, which only holds while the driver precedes it in the base list
        for cls in (BoysOTR, PipekMezeyOTR, EdmistonRuedenbergOTR):
            with self.subTest(cls=cls.__name__):
                loc = cls(mol, occ_orbs)
                self.assertFalse(callable(loc.stability))
                self.assertTrue(callable(loc.stability_check))

    def test_localization_improves_the_objective(self) -> None:
        # a flipped cost_sign would drive each cost function the wrong way, which no
        # comparison against a converged PySCF result on its own would reveal
        for cls, _, sign in self.localizers:
            with self.subTest(cls=cls.__name__):
                loc = cls(mol, occ_orbs)
                start = self._objective(cls, loc.mo_coeff, sign)
                mo_loc = loc.kernel()
                final = self._objective(cls, mo_loc, sign)
                self.assertLess(final, start)

    def test_localization_is_at_least_as_good_as_pyscf(self) -> None:
        for cls, ref_cls, sign in self.localizers:
            with self.subTest(cls=cls.__name__):
                ref = self._objective(cls, ref_cls(mol, occ_orbs).kernel(), sign)
                otr = self._objective(cls, cls(mol, occ_orbs).kernel(), sign)
                # both may land in different local minima, so only a clear regression
                # counts as a failure
                self.assertLessEqual(otr, ref + 1e-6)

    def test_localized_orbitals_span_the_input_space(self) -> None:
        # localization is a unitary rotation within the given subspace
        ovlp = mol.intor_symmetric("int1e_ovlp")
        for cls, _, _ in self.localizers:
            with self.subTest(cls=cls.__name__):
                mo_loc = cls(mol, occ_orbs).kernel()
                self.assertEqual(mo_loc.shape, occ_orbs.shape)
                self.assertAlmostEqual(
                    abs(mo_loc.T @ ovlp @ mo_loc - np.eye(mo_loc.shape[1])).max(),
                    0.0,
                    8,
                )
                # the projector onto the subspace is invariant under the rotation
                self.assertAlmostEqual(
                    abs(mo_loc @ mo_loc.T - occ_orbs @ occ_orbs.T).max(),
                    0.0,
                    8,
                )
                # each localized orbital is closest to the input orbital of the same
                # index, as PySCF sorts them
                overlap = occ_orbs.T @ ovlp @ mo_loc
                self.assertEqual(
                    mo_mapping.mo_1to1map(overlap), list(range(overlap.shape[1]))
                )

    def test_single_orbital_is_returned_unchanged(self) -> None:
        # without any rotation, the orbital also counts as stable
        one_orb = occ_orbs[:, :1]
        loc = BoysOTR(mol, one_orb)
        self.assertAlmostEqual(abs(loc.kernel() - one_orb).max(), 0.0, 12)
        stable, _ = check_stability_shape(self, loc)
        self.assertTrue(stable)

    def test_counters_record_the_work_done(self) -> None:
        loc = BoysOTR(mol, occ_orbs)
        self.assertEqual(loc.n_update_orbs, 0)
        self.assertEqual(loc.n_hess_x, 0)
        loc.kernel()
        self.assertGreater(loc.n_update_orbs, 0)
        self.assertGreater(loc.n_hess_x, 0)

    def test_derivatives_match_finite_differences(self) -> None:
        for cls, _, _ in self.localizers:
            with self.subTest(cls=cls.__name__):
                check_derivatives_match_finite_differences(self, cls(mol, occ_orbs))

    def test_large_rotation_is_rejected(self) -> None:
        check_large_rotation_is_rejected(self, BoysOTR(mol, occ_orbs))

    def test_localizer_stability_check(self) -> None:
        loc = BoysOTR(mol, occ_orbs)
        loc.kernel()
        check_stability_shape(self, loc)

    def test_rotation_qn(self) -> None:
        # rotation QN wraps the orbital update of every localizer, with the gradient at
        # every trial point and the parallel transport of the orthogonal group, and
        # reaches the optimum the exact Hessian reaches with either updating scheme,
        # also with the identity transport, whereas the density QN methods work only
        # on the densities of the mean-field drivers
        for cls, _, sign in self.localizers:
            exact = cls(mol, occ_orbs)
            exact.kernel()
            objective = self._objective(cls, exact.mo_coeff, sign)
            for qn_method, parallel_transport in product(
                rotation_qn_methods, (True, False)
            ):
                with self.subTest(
                    cls=cls.__name__,
                    qn_method=qn_method,
                    parallel_transport=parallel_transport,
                ):
                    loc = cls(mol, occ_orbs)
                    loc.qn_method = qn_method
                    loc.parallel_transport = parallel_transport
                    loc.kernel()
                    check_no_referrers(self, loc)
                    self.assertTrue(loc.converged)
                    self.assertAlmostEqual(
                        self._objective(cls, loc.mo_coeff, sign), objective, 8
                    )
            with self.subTest(cls=cls.__name__, qn_method="arh"):
                loc = cls(mol, occ_orbs)
                loc.qn_method = "arh"
                self.assertRaises(ValueError, loc.kernel)

    def test_rotation_qn_callbacks(self) -> None:
        # the localizers supply the objective function with the gradient at a trial
        # point and the parallel transport unless parallel_transport is
        # unset
        for parallel_transport in (True, False):
            with self.subTest(parallel_transport=parallel_transport):
                loc = BoysOTR(mol, occ_orbs)
                loc.parallel_transport = parallel_transport
                self.assertEqual(
                    list(loc._rotation_qn_callbacks(loc._func)),
                    ["obj_func_grad"] + parallel_transport * ["transport"],
                )

    def test_trial_gradient(self) -> None:
        # the gradient at a trial point is that of the orbital update, with the sign of
        # the minimization objective also for the localizers that maximize
        for cls, _, _ in self.localizers:
            with self.subTest(cls=cls.__name__):
                loc = cls(mol, occ_orbs)
                loc.kernel()
                check_trial_gradient(self, loc)

    def test_transport(self) -> None:
        # the transport is the parallel transport of the orthogonal group, none of
        # whose rotations are redundant
        loc = BoysOTR(mol, occ_orbs)
        loc.kernel()
        rng = np.random.default_rng(7)
        geodesic, tangent_vector = 0.3 * rng.standard_normal((2, loc.n_param))
        transported = tangent_vector.copy()
        loc._transport(geodesic, transported)
        n_orbitals = occ_orbs.shape[1]
        np.testing.assert_allclose(
            transported,
            reference_transport(
                loc.unpack_uniq_var,
                loc.pack_uniq_var,
                np.ones((n_orbitals, n_orbitals), dtype=bool),
                geodesic,
                tangent_vector,
            ),
            rtol=0,
            atol=1e-12,
        )
        self.assertGreater(np.abs(transported - tangent_vector).max(), 1e-3)


class CASSCFTests(unittest.TestCase):
    """
    tests for the CASSCF driver: the active space, the converged energy, the
    derivatives and the stability check
    """

    def test_conversion_preserves_the_active_space(self) -> None:
        mc = to_otr(mcscf.CASSCF(mf, 4, 4))
        self.assertEqual(mc.ncas, 4)
        self.assertEqual(mc.nelecas, (2, 2))

    def test_casscf_energy(self) -> None:
        # with symmetry, the parameters are the rotations between orbitals of the same
        # irreducible representation and the CI coefficients of the determinants of
        # the symmetry of the state
        n_param_orb, n_param_ci = {}, {}
        for mf_in in (mf, mf_sym):
            with self.subTest(symmetry=mf_in.mol.symmetry):
                ref = mcscf.CASSCF(mf_in, 4, 4).run(conv_tol=1e-10)
                mc = to_otr(mcscf.CASSCF(mf_in, 4, 4))
                e_tot, e_cas, ci, mo_coeff, mo_energy = mc.kernel()
                self.assertAlmostEqual(e_tot, ref.e_tot, 7)
                self.assertLess(gradient_rms(mc), SolverSettings().conv_tol)
                self.assertAlmostEqual(mc.e_tot, ref.e_tot, 7)
                self.assertEqual(lib.chkfile.load(mc.chkfile, "mcscf/e_tot"), e_tot)
                self.assertAlmostEqual(float(np.linalg.norm(np.ravel(ci))), 1.0, 9)
                n_param_orb[mf_in.mol.symmetry] = mc.n_param_orb
                n_param_ci[mf_in.mol.symmetry] = mc.n_param_ci
        self.assertLess(n_param_orb[True], n_param_orb[False])
        self.assertLess(n_param_ci[True], n_param_ci[False])

    def test_scanner_follows_the_geometry(self) -> None:
        # a geometry optimization or a scan calls the kernel through a scanner, which
        # takes the first value the kernel returns as the energy
        mol_displaced = gto.M(
            atom="O 0 0 0; H 0 -0.78 0.60; H 0 0.78 0.60", basis="6-31g", verbose=0
        )
        ref = mcscf.CASSCF(scf.RHF(mol), 4, 4).as_scanner()
        mc = to_otr(mcscf.CASSCF(scf.RHF(mol), 4, 4)).as_scanner()
        ref(mol)
        mc(mol)
        self.assertAlmostEqual(mc(mol_displaced), ref(mol_displaced), 7)

    def test_state_averaged_casscf_energy(self) -> None:
        # the states stay orthonormal and so do not collapse onto the ground state, for
        # equal and unequal weights, for states of several CI solvers, here of
        # different spin or of different point-group symmetry, and for either order of
        # the conversion and the state averaging; the energies of the individual
        # states, at which the averaged energy is not stationary, converge more slowly
        # than the averaged energy
        for name, mf_in, average in (
            ("equal weights", mf, lambda mc: mcscf.state_average_(mc, [0.5, 0.5])),
            ("unequal weights", mf, lambda mc: mcscf.state_average_(mc, [0.6, 0.4])),
            (
                "singlet and triplet",
                mf,
                lambda mc: mcscf.state_average_mix_(
                    mc, singlet_and_triplet_solvers(), [0.6, 0.4]
                ),
            ),
            (
                "states of different symmetry",
                mf_sym,
                lambda mc: mcscf.state_average_mix_(
                    mc, solvers_of_different_symmetry(), [0.6, 0.4]
                ),
            ),
        ):
            ref = average(mcscf.CASSCF(mf_in, 4, 4)).run(conv_tol=1e-10)
            for order, mc in (
                (
                    "conversion of the average",
                    to_otr(average(mcscf.CASSCF(mf_in, 4, 4))),
                ),
                (
                    "average of the conversion",
                    average(to_otr(mcscf.CASSCF(mf_in, 4, 4))),
                ),
            ):
                with self.subTest(average=name, order=order):
                    mc.kernel()
                    self.assertAlmostEqual(mc.e_tot, ref.e_tot, 7)
                    np.testing.assert_allclose(mc.e_states, ref.e_states, atol=1e-4)
                    self.assertLess(gradient_rms(mc), SolverSettings().conv_tol)
                    stable, _ = check_stability_shape(self, mc)
                    self.assertTrue(stable)

    def test_casscf_orbitals_stay_orthonormal(self) -> None:
        mc = to_otr(mcscf.CASSCF(mf, 4, 4))
        mc.kernel()
        ovlp = mol.intor_symmetric("int1e_ovlp")
        self.assertAlmostEqual(
            abs(
                mc.mo_coeff.T @ ovlp @ mc.mo_coeff - np.eye(mc.mo_coeff.shape[1])
            ).max(),
            0.0,
            8,
        )

    def test_counters_record_the_work_done(self) -> None:
        mc = to_otr(mcscf.CASSCF(mf, 4, 4))
        self.assertEqual(mc.n_update_orbs, 0)
        self.assertEqual(mc.n_hess_x, 0)
        mc.kernel()
        self.assertGreater(mc.n_update_orbs, 0)
        self.assertGreater(mc.n_hess_x, 0)

    def test_derivatives_match_finite_differences(self) -> None:
        # for unequal weights, the Hessian lacks the response of the eigenstates within
        # the space the states span, so only the gradient is compared, except for
        # states of several CI solvers, each the only state of its solver
        for name, mf_in, average, curvature in (
            ("single state", mf, lambda mc: mc, True),
            ("symmetry", mf_sym, lambda mc: mc, True),
            (
                "equal weights",
                mf,
                lambda mc: mcscf.state_average_(mc, [0.5, 0.5]),
                True,
            ),
            (
                "unequal weights",
                mf,
                lambda mc: mcscf.state_average_(mc, [0.6, 0.4]),
                False,
            ),
            (
                "singlet and triplet",
                mf,
                lambda mc: mcscf.state_average_mix_(
                    mc, singlet_and_triplet_solvers(), [0.6, 0.4]
                ),
                True,
            ),
            (
                "states of different symmetry",
                mf_sym,
                lambda mc: mcscf.state_average_mix_(
                    mc, solvers_of_different_symmetry(), [0.6, 0.4]
                ),
                True,
            ),
        ):
            with self.subTest(average=name):
                check_derivatives_match_finite_differences(
                    self,
                    average(to_otr(mcscf.CASSCF(mf_in, 4, 4))),
                    curvature=curvature,
                )

    def test_only_large_orbital_rotations_are_rejected(self) -> None:
        mc = to_otr(mcscf.CASSCF(mf, 4, 4))
        check_large_rotation_is_rejected(self, mc)

        # the CI part of a step displaces the CI vector rather than rotating orbitals
        step = np.zeros(mc.n_param, dtype=np.float64)
        step[mc.n_param_orb] = 1.0
        self.assertTrue(np.isfinite(mc._func(step)))

    def test_casscf_stability_check(self) -> None:
        mc = to_otr(mcscf.CASSCF(mf, 4, 4))
        mc.kernel()
        check_stability_shape(self, mc)

    def test_rotation_qn(self) -> None:
        # rotation QN wraps the orbital update of CASSCF, with the gradient at every
        # trial point, the parallel transport of the orbitals and the transport of the
        # states, which for a state average follows the states as they turn within the
        # space they span, and reaches the minimum with either updating scheme, also
        # with the identity transport of the orbitals
        for name, mf_in, average, parallel_transport in (
            ("single state", mf, lambda mc: mc, True),
            ("identity transport", mf, lambda mc: mc, False),
            ("symmetry", mf_sym, lambda mc: mc, True),
            (
                "equal weights",
                mf,
                lambda mc: mcscf.state_average_(mc, [0.5, 0.5]),
                True,
            ),
            (
                "unequal weights",
                mf,
                lambda mc: mcscf.state_average_(mc, [0.6, 0.4]),
                True,
            ),
        ):
            ref = average(mcscf.CASSCF(mf_in, 4, 4)).run(conv_tol=1e-10)
            for qn_method in rotation_qn_methods:
                with self.subTest(case=name, qn_method=qn_method):
                    mc = to_otr(average(mcscf.CASSCF(mf_in, 4, 4)))
                    mc.qn_method = qn_method
                    mc.parallel_transport = parallel_transport
                    mc.kernel()
                    check_no_referrers(self, mc)
                    self.assertTrue(mc.converged)
                    self.assertAlmostEqual(mc.e_tot, ref.e_tot, 6)
                    self.assertLess(gradient_rms(mc), SolverSettings().conv_tol)

    def test_trial_gradient(self) -> None:
        # the gradient at a trial point, which reuses the integrals the objective
        # function transformed, is that of the orbital update, also for a state
        # average, whose states turn within the space they span
        for name, average in (
            ("single state", lambda mc: mc),
            ("unequal weights", lambda mc: mcscf.state_average_(mc, [0.6, 0.4])),
        ):
            with self.subTest(case=name):
                mc = to_otr(average(mcscf.CASSCF(mf, 4, 4)))
                mc.kernel()
                check_trial_gradient(self, mc)

                # the objective function evaluates a step anew once the CI vectors
                # moved, also where the orbitals did not
                rng = np.random.default_rng(9)
                step = 0.05 * rng.standard_normal(mc.n_param)
                energy = mc._func(step)
                ci_step = np.zeros(mc.n_param)
                ci_step[mc.n_param_orb :] = 0.05 * rng.standard_normal(mc.n_param_ci)
                grad, h_diag = grad_and_h_diag(mc)
                mc._update_orbs(ci_step, grad, h_diag)
                self.assertNotEqual(mc._func(step), energy)

    def test_transport(self) -> None:
        # the transport of the orbital rotations is the parallel transport of the
        # manifold of the orbitals whose rotations PySCF does not select are
        # redundant, and that of the CI part lies in the tangent space of the states
        mc = to_otr(mcscf.state_average_(mcscf.CASSCF(mf, 4, 4), [0.6, 0.4]))
        mc.kernel()
        rng = np.random.default_rng(8)
        geodesic, tangent_vector = 0.3 * rng.standard_normal((2, mc.n_param))
        transported = tangent_vector.copy()
        mc._transport(geodesic, transported)
        n_param_orb = mc.n_param_orb
        uniq_var = mc.uniq_var_indices(
            mc.mo_coeff.shape[1], mc.ncore, mc.ncas, mc.frozen
        )
        np.testing.assert_allclose(
            transported[:n_param_orb],
            reference_transport(
                mc.unpack_uniq_var,
                mc.pack_uniq_var,
                uniq_var | uniq_var.T,
                geodesic[:n_param_orb],
                tangent_vector[:n_param_orb],
            ),
            rtol=0,
            atol=1e-12,
        )
        self.assertGreater(
            np.abs(transported[:n_param_orb] - tangent_vector[:n_param_orb]).max(),
            1e-3,
        )
        states = np.array(mc.ci)
        ci_part = mc._embed(transported)[n_param_orb:].reshape(states.shape)
        np.testing.assert_allclose(ci_part @ states.T, 0.0, rtol=0, atol=1e-12)

        # without the parallel transport, the orbital rotations are kept
        mc.parallel_transport = False
        kept = tangent_vector.copy()
        mc._transport(geodesic, kept)
        np.testing.assert_array_equal(kept[:n_param_orb], tangent_vector[:n_param_orb])
        mc.parallel_transport = True

        def check_turn(geodesic: np.ndarray, turn: np.ndarray) -> None:
            # the CI part turns as the states did and then lies in the tangent space
            # of the current states
            transported = tangent_vector.copy()
            mc._transport(geodesic, transported)
            states = np.array(mc.ci)
            expected = turn @ mc._embed(tangent_vector)[n_param_orb:].reshape(
                states.shape
            )
            expected -= (expected @ states.T) @ states
            np.testing.assert_allclose(
                mc._embed(transported)[n_param_orb:].reshape(states.shape),
                expected,
                rtol=0,
                atol=1e-12,
            )
            self.assertGreater(np.abs(turn - np.eye(len(states))).max(), 1e-3)

        # along the step of an orbital update, the states turned from the displaced CI
        # vectors D to the eigenstates E = R^T D within the space these span, so that
        # the CI part turns by R^T = E D^T, and back from a trial point, whose
        # eigenstates turned likewise, by R = D E^T
        step = 0.3 * rng.standard_normal(mc.n_param)
        displaced = np.array(mc._displaced_ci(step))
        grad, h_diag = grad_and_h_diag(mc)
        mc._update_orbs(step, grad, h_diag)
        check_turn(step, np.array(mc.ci) @ displaced.T)
        step = 0.3 * rng.standard_normal(mc.n_param)
        mc._func(step)
        check_turn(
            -step,
            np.array(mc._displaced_ci(step)) @ np.array(mc._trial_point(step)[2]).T,
        )
