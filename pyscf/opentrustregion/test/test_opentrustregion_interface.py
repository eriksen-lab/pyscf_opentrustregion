# Copyright (C) 2025- Jonas Greiner
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Tests for the PySCF interface to OpenTrustRegion
"""

import io
import unittest
from itertools import product
from typing import Any, Dict, Tuple

import numpy as np

from pyscf import gto, scf, dft, lo, mcscf, lib

from pyscf.opentrustregion import (
    RHFOTR,
    ROHFOTR,
    UHFOTR,
    BoysOTR,
    PipekMezeyOTR,
    EdmistonRuedenbergOTR,
    mf_to_otr,
    casscf_to_otr,
)
from pyscf.opentrustregion.opentrustregion_interface import (
    CASSCFOTR,
    OTR,
    assign_solver_settings,
    assign_stability_settings,
    non_setting_fields,
    setting_fields,
    setting_is_assignable,
    setting_was_set_by_user,
    solver_setting_fields,
    stability_setting_aliases,
    stability_setting_fields,
)
from pyopentrustregion import SolverSettings, StabilitySettings

# fixtures shared by every test, built once in setUpModule
mol: gto.Mole
mol_sym: gto.Mole
mol_open: gto.Mole
mf: scf.hf.RHF
occ_orbs: np.ndarray


def setUpModule() -> None:
    global mol, mol_sym, mol_open, mf, occ_orbs
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

    mf = scf.RHF(mol).run()
    occ_orbs = mf.mo_coeff[:, : mol.nelec[0]]


def tearDownModule() -> None:
    global mol, mol_sym, mol_open, mf, occ_orbs
    mol.stdout.close()
    del mol, mol_sym, mol_open, mf, occ_orbs


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


def check_derivatives_match_finite_differences(
    test: unittest.TestCase, obj: Any
) -> None:
    """
    this function checks that the gradient and the Hessian linear transformation an
    orbital update reports are the derivatives of the objective the solver is handed,
    which a mismatched factor on either would break
    """
    obj.kernel()
    grad, h_diag = grad_and_h_diag(obj)

    # step away from the stationary point, where the gradient vanishes and would
    # compare equal whatever factor it carried
    np.random.seed(2)
    displacement = 0.05 * (np.random.random(obj.n_param) - 0.5)
    _, hess_x = obj._update_orbs(displacement, grad, h_diag)

    x = np.random.random(obj.n_param) - 0.5
    x /= np.linalg.norm(x)
    hx = np.empty(obj.n_param, dtype=np.float64)
    hess_x(x, hx)

    step = 1e-5
    derivative = (obj._func(step * x) - obj._func(-step * x)) / (2 * step)
    test.assertLess(abs(grad @ x - derivative), 1e-5 * abs(derivative))

    # the difference formulas are limited by round off in the objective, so both are
    # compared relative to the value they approximate
    step = 1e-4
    curvature = (
        obj._func(step * x)
        - 2 * obj._func(np.zeros(obj.n_param))
        + obj._func(-step * x)
    ) / step**2
    test.assertLess(abs(x @ hx - curvature), 1e-4 * abs(curvature))


def capture_warnings(obj: Any) -> io.StringIO:
    """
    this function routes the PySCF log of an object into a buffer, at the verbosity that
    still includes warnings
    """
    log = io.StringIO()
    obj.stdout = log
    obj.verbose = lib.logger.WARN
    return log


def check_stability_shape(test: unittest.TestCase, obj: OTR) -> Tuple[bool, np.ndarray]:
    stable, direction = obj.stability_check()
    test.assertIsInstance(stable, bool)
    test.assertIsInstance(direction, np.ndarray)
    test.assertEqual(direction.shape, (obj.n_param,))
    return stable, direction


class DummySettingsHolder:
    """
    this class stands in for a PySCF object carrying settings, so that the settings
    assignment can be tested without constructing a solver object
    """

    # the settings a test assigns are arbitrary by design, so they cannot be declared
    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
        self._inherited_settings: Dict[str, Any] = {}

    def inherit(self, **kwargs: Any) -> "DummySettingsHolder":
        """
        this function assigns settings and records them as inherited from PySCF, the
        way a conversion function does before the user gets to see the object
        """
        self.__dict__.update(kwargs)
        self._inherited_settings.update(kwargs)
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

        # the settings the library exposes, pinned so that adding or removing one
        # upstream shows up here rather than silently changing what the interface
        # passes through and what the README documents
        self.assertEqual(
            solver_setting_fields,
            [
                "precond",
                "precond_pd",
                "project",
                "get_extra_trial_vectors",
                "conv_check",
                "stability_hess_x",
                "logger",
                "stability",
                "line_search",
                "refresh_hess",
                "grad_noise",
                "hess_symm",
                "conv_tol",
                "start_trust_radius",
                "global_red_factor",
                "local_red_factor",
                "n_random_trial_vectors",
                "n_extra_trial_vectors",
                "n_macro",
                "n_micro",
                "jacobi_davidson_start",
                "seed",
                "verbose",
                "subsystem_solver",
                "trust_region_shape",
            ],
        )
        self.assertEqual(
            stability_setting_fields,
            [
                "precond",
                "project",
                "approx_hess_x",
                "get_extra_trial_vectors",
                "conv_check",
                "logger",
                "hess_symm",
                "stop_on_instability",
                "conv_tol",
                "stability_thresh",
                "n_random_trial_vectors",
                "n_extra_trial_vectors",
                "n_iter",
                "jacobi_davidson_start",
                "seed",
                "verbose",
                "diag_solver",
            ],
        )

        # the field list must be derived from the ctypes structure rather than hard
        # coded, so that a new solver setting becomes assignable on its own
        from pyopentrustregion.python_interface import SolverSettingsC

        self.assertEqual(
            setting_fields(SolverSettingsC),
            [
                field[0]
                for field in SolverSettingsC._fields_
                if field[0] != "stability_settings"
                and field[0] not in non_setting_fields
            ],
        )

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
        obj = DummySettingsHolder().inherit(conv_tol=1e-9)
        obj.conv_tol = 1e-7
        settings = SolverSettings()
        assign_solver_settings(obj, settings)
        self.assertAlmostEqual(settings.conv_tol, 1e-7, 12)

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

    def test_every_alias_target_is_a_shared_field(self) -> None:
        # an alias only earns its existence for a name both settings objects use
        for setting, alias in stability_setting_aliases.items():
            self.assertIn(setting, stability_setting_fields)
            self.assertIn(setting, solver_setting_fields)
            self.assertEqual(alias, "stability_" + setting)

    def test_settings_are_recognized_keys(self) -> None:
        # PySCF warns about attributes it does not recognize, so every assignable
        # setting has to be registered on the class; PySCF collects the keys of every
        # class in the method resolution order, so a driver can register its own on top
        # of those of its base classes
        otr = mf_to_otr(scf.RHF(mol))
        keys = set().union(*(getattr(cls, "_keys", ()) for cls in type(otr).__mro__))
        for setting in solver_setting_fields + list(stability_setting_aliases.values()):
            self.assertIn(setting, keys)


class SCFTests(unittest.TestCase):
    """
    tests for the Hartree-Fock and Kohn-Sham drivers: the conversion, the converged
    energies, the cached orbital updates, the derivatives and the stability check
    """

    def _check(self, mf_in: scf.hf.SCF, ref_mf: scf.hf.SCF, **settings: Any) -> OTR:
        ref_newton = ref_mf.newton()
        ref = ref_newton.kernel()
        otr = mf_to_otr(mf_in)
        for name, value in settings.items():
            setattr(otr, name, value)
        e_tot = otr.kernel()
        self.assertAlmostEqual(e_tot, ref, 8)
        self.assertAlmostEqual(otr.e_tot, ref, 8)

        # the orbitals are ordered as PySCF orders them, by energy with the occupied
        # ones first, which the methods building on them rely on
        self.assertFalse(np.signbit(otr.mo_occ).any())
        np.testing.assert_array_equal(otr.mo_occ, ref_newton.mo_occ)
        np.testing.assert_allclose(otr.mo_energy, ref_newton.mo_energy, atol=1e-4)

        # the orbitals must remain a valid, orthonormal set in the AO basis
        ovlp = otr.mol.intor_symmetric("int1e_ovlp")
        mo_coeff = np.asarray(otr.mo_coeff)
        if mo_coeff.ndim == 2:
            mo_coeff = mo_coeff[None]
        for coeff in mo_coeff:
            self.assertAlmostEqual(
                abs(coeff.T @ ovlp @ coeff - np.eye(coeff.shape[1])).max(), 0.0, 8
            )

        # check if the gradient is converged
        self.assertLess(gradient_rms(otr), SolverSettings().conv_tol)
        return otr

    def test_mf_to_otr_selects_the_matching_class(self) -> None:
        for mf_in, cls in (
            (scf.RHF(mol), RHFOTR),
            (scf.ROHF(mol_open), ROHFOTR),
            (scf.UHF(mol_open), UHFOTR),
            (dft.RKS(mol, xc="b3lyp"), RHFOTR),
            (dft.UKS(mol_open, xc="b3lyp"), UHFOTR),
        ):
            with self.subTest(mf=mf_in.__class__.__name__):
                otr = mf_to_otr(mf_in)
                self.assertIsInstance(otr, cls)
                # the converted object must remain usable as the PySCF object it came
                # from, since PySCF routines are typed against those classes
                self.assertIsInstance(otr, scf.hf.SCF)

    def test_mf_to_otr_is_idempotent(self) -> None:
        otr = mf_to_otr(scf.RHF(mol))
        self.assertIs(mf_to_otr(otr), otr)

    def test_mf_to_otr_stability_is_a_setting(self) -> None:
        # stability is a boolean solver setting, so the PySCF method of that name has
        # to give way; the OpenTrustRegion equivalent is stability_check
        otr = mf_to_otr(scf.RHF(mol))
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
        mol_open_sym = mol_open.copy()
        mol_open_sym.symmetry = True
        mol_open_sym.build(False, False)
        self._check(scf.UHF(mol_open_sym), scf.UHF(mol_open_sym))

    def test_rks(self) -> None:
        self._check(dft.RKS(mol, xc="b3lyp"), dft.RKS(mol, xc="b3lyp"))

    def test_uks(self) -> None:
        self._check(dft.UKS(mol_open, xc="b3lyp"), dft.UKS(mol_open, xc="b3lyp"))

    def test_symmetry_restricts_the_parameter_count(self) -> None:
        # symmetry forbidden rotations must be excluded from the parameter vector
        otr = mf_to_otr(scf.RHF(mol))
        otr.kernel()
        otr_sym = mf_to_otr(scf.RHF(mol_sym))
        otr_sym.kernel()
        self.assertLess(otr_sym.n_param, otr.n_param)

    def test_kernel_from_a_given_density_matrix(self) -> None:
        ref = scf.RHF(mol).newton().kernel()
        otr = mf_to_otr(scf.RHF(mol))
        e_tot = otr.kernel(dm=mf.make_rdm1())
        self.assertAlmostEqual(e_tot, ref, 8)

    def test_repeated_zero_step_reuses_the_update(self) -> None:
        otr = mf_to_otr(scf.RHF(mol))
        otr.kernel()
        grad, h_diag = grad_and_h_diag(otr)
        zero = np.zeros(otr.n_param, dtype=np.float64)

        # the kernel canonicalizes at the end, so the first request has to recompute
        otr._update_orbs(zero, grad, h_diag)
        n_after_first = otr.n_update_orbs

        # a second request at the same point is served from the cache
        func, _ = otr._update_orbs(zero, grad, h_diag)
        self.assertEqual(otr.n_update_orbs, n_after_first)
        self.assertAlmostEqual(func, otr.e_tot, 9)

    def test_cache_is_invalidated_when_orbitals_move(self) -> None:
        otr = mf_to_otr(scf.RHF(mol))
        otr.kernel()

        # the routines the kernel adopted from the extension factories give way to the
        # cached ones of this object once a stability check examines orbitals that
        # replace the ones the kernel left
        otr.mo_coeff = otr.mo_coeff.copy()
        otr.mo_coeff[:, 0] *= -1.0
        otr.stability_check()
        grad, h_diag = grad_and_h_diag(otr)
        zero = np.zeros(otr.n_param, dtype=np.float64)
        cached = otr._last_update

        # a zero step alone must not be taken as proof that nothing has changed;
        # n_update_orbs only counts updates that move the optimization, so a
        # recomputation shows as a new cached result
        otr.mo_coeff = otr.mo_coeff.copy()
        otr.mo_coeff[:, 1] *= -1.0
        otr._update_orbs(zero, grad, h_diag)
        self.assertIsNot(otr._last_update, cached)

    def test_nonzero_step_always_recomputes(self) -> None:
        otr = mf_to_otr(scf.RHF(mol))
        otr.kernel()
        grad, h_diag = grad_and_h_diag(otr)
        step = np.zeros(otr.n_param, dtype=np.float64)
        step[0] = 1e-3

        otr._update_orbs(np.zeros(otr.n_param), grad, h_diag)
        n_before = otr.n_update_orbs
        otr._update_orbs(step, grad, h_diag)
        self.assertEqual(otr.n_update_orbs, n_before + 1)

    def test_counters_record_the_work_done(self) -> None:
        otr = mf_to_otr(scf.RHF(mol))
        self.assertEqual(otr.n_update_orbs, 0)
        self.assertEqual(otr.n_hess_x, 0)
        otr.kernel()
        self.assertGreater(otr.n_update_orbs, 0)
        self.assertGreater(otr.n_hess_x, 0)

    def test_derivatives_match_finite_differences(self) -> None:
        check_derivatives_match_finite_differences(self, mf_to_otr(scf.RHF(mol)))

    def test_converged_rhf_is_stable(self) -> None:
        otr = mf_to_otr(scf.RHF(mol))
        otr.kernel()
        stable, _ = check_stability_shape(self, otr)
        self.assertTrue(stable)

    def test_unstable_solution_is_detected(self) -> None:
        # a stretched singlet has a lower energy broken symmetry UHF solution, so the
        # restricted solution is a saddle point in the unrestricted parameters
        mol_stretched = gto.M(atom="H 0 0 0; H 0 0 3.0", basis="6-31g", verbose=0)
        rhf = scf.RHF(mol_stretched).run()

        # the kernel adopts the routines of the MO or the OAO extension, whose
        # extension objects keep the orbitals of the kernel
        for oao in (False, True):
            with self.subTest(oao=oao):
                otr = mf_to_otr(scf.UHF(mol_stretched))
                otr.oao = oao
                e_tot = otr.kernel()

                # the solver escapes the restricted solution on its own, so its own
                # result is a minimum
                self.assertLess(e_tot, rhf.e_tot - 1e-3)
                stable, _ = check_stability_shape(self, otr)
                self.assertTrue(stable)

                # placing the same object back at the restricted solution must make the
                # check report the instability, along with a normalized direction out
                # of the saddle
                otr.mo_coeff = np.array([rhf.mo_coeff, rhf.mo_coeff])
                stable, direction = check_stability_shape(self, otr)
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

    def test_stability_check_examines_the_current_occupations(self) -> None:
        # the check examines the density of the current occupations and their
        # occupied-virtual rotations, which with symmetry are those between orbitals of
        # the same irrep, also when the occupations are assigned after the kernel and
        # the update the kernel left is cached; ROHF adopts no routines from the
        # extension factories
        for cls, molecule in product((scf.RHF, scf.ROHF), (mol, mol_sym)):
            with self.subTest(cls=cls.__name__, symmetry=molecule.symmetry):
                otr = mf_to_otr(cls(molecule))
                otr.kernel()
                grad, h_diag = grad_and_h_diag(otr)
                otr._update_orbs(np.zeros(otr.n_param), grad, h_diag)

                # move the highest occupied pair to the lowest virtual orbital
                mo_occ = otr.mo_occ.copy()
                homo = np.flatnonzero(mo_occ > 0)[-1]
                lumo = np.flatnonzero(mo_occ == 0)[0]
                mo_occ[[homo, lumo]] = mo_occ[[lumo, homo]]
                otr.mo_occ = mo_occ
                check_stability_shape(self, otr)

                # the update the check used is reused; n_update_orbs only counts
                # updates that move the optimization, so reuse shows as an unchanged
                # cached result
                cached = otr._last_update
                grad, h_diag = grad_and_h_diag(otr)
                func, _ = otr._update_orbs(np.zeros(otr.n_param), grad, h_diag)
                self.assertIs(otr._last_update, cached)

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
        # the PySCF object passed to mf_to_otr evaluates the energy and the gradient, so
        # an option defining the energy that is assigned afterwards reaches at most the
        # Hessian, which the solver and the stability check warn about
        otr = mf_to_otr(dft.UKS(mol_open, xc="b3lyp"))
        log = capture_warnings(otr)

        # a replaced grid for the nonlocal correlation, which b3lyp does not use, leaves
        # the optimization unaffected
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
        otr = mf_to_otr(uks)
        log = capture_warnings(otr)
        otr.xc = "B3LYP"
        otr.grids.level = 2
        # the secondary grid only exists in PySCF releases after 2.14
        if hasattr(otr, "set_second_grids"):
            otr.set_second_grids(2)
        otr.kernel()
        otr.stability_check()
        self.assertNotIn("mf_to_otr", log.getvalue())


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
        for cls, ref_cls, sign in self.localizers:
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

    def test_single_orbital_is_returned_unchanged(self) -> None:
        one_orb = occ_orbs[:, :1]
        self.assertAlmostEqual(
            abs(BoysOTR(mol, one_orb).kernel() - one_orb).max(), 0.0, 12
        )

    def test_repeated_zero_step_reuses_the_update(self) -> None:
        loc = BoysOTR(mol, occ_orbs)
        loc.kernel()
        grad, h_diag = grad_and_h_diag(loc)
        zero = np.zeros(loc.n_param, dtype=np.float64)

        # n_update_orbs only counts updates that move the optimization, so reuse shows
        # as an unchanged cached result
        loc._update_orbs(zero, grad, h_diag)
        cached = loc._last_update
        loc._update_orbs(zero, grad, h_diag)
        self.assertIs(loc._last_update, cached)

    def test_counters_record_the_work_done(self) -> None:
        loc = BoysOTR(mol, occ_orbs)
        self.assertEqual(loc.n_update_orbs, 0)
        self.assertEqual(loc.n_hess_x, 0)
        loc.kernel()
        self.assertGreater(loc.n_update_orbs, 0)
        self.assertGreater(loc.n_hess_x, 0)

    def test_derivatives_match_finite_differences(self) -> None:
        check_derivatives_match_finite_differences(self, BoysOTR(mol, occ_orbs))

    def test_localizer_stability_check(self) -> None:
        loc = BoysOTR(mol, occ_orbs)
        loc.kernel()
        check_stability_shape(self, loc)


class CASSCFTests(unittest.TestCase):
    """
    tests for the CASSCF driver: the conversion, the converged energy, the cached
    updates, the derivatives and the stability check
    """

    def test_casscf_to_otr_preserves_the_active_space(self) -> None:
        mc = casscf_to_otr(mcscf.CASSCF(mf, 4, 4))
        self.assertIsInstance(mc, CASSCFOTR)
        self.assertEqual(mc.ncas, 4)
        self.assertEqual(mc.nelecas, (2, 2))

    def test_casscf_to_otr_is_idempotent(self) -> None:
        mc = casscf_to_otr(mcscf.CASSCF(mf, 4, 4))
        self.assertIs(casscf_to_otr(mc), mc)

    def test_casscf_to_otr_rejects_state_averaging(self) -> None:
        mc = mcscf.state_average_(mcscf.CASSCF(mf, 4, 4), [0.5, 0.5])
        self.assertRaises(RuntimeError, casscf_to_otr, mc)

    def test_casscf_energy(self) -> None:
        ref = mcscf.CASSCF(mf, 4, 4).run()
        mc = casscf_to_otr(mcscf.CASSCF(mf, 4, 4))
        converged, e_tot, e_cas, ci, mo_coeff, mo_energy = mc.kernel()
        self.assertAlmostEqual(e_tot, ref.e_tot, 7)
        self.assertLess(gradient_rms(mc), SolverSettings().conv_tol)
        self.assertAlmostEqual(mc.e_tot, ref.e_tot, 7)
        self.assertAlmostEqual(float(np.linalg.norm(np.ravel(ci))), 1.0, 9)

    def test_ci_vectors_covers_one_and_several_roots(self) -> None:
        # a single root keeps the CI vector as one array and several roots as a list
        # of them. Nothing else in the suite reaches the several root shape, since a
        # state averaged calculation is rejected outright
        single = np.array([0.6, 0.8])
        self.assertEqual(len(CASSCFOTR.ci_vectors(single)), 1)
        self.assertIs(CASSCFOTR.ci_vectors(single)[0], single)

        several = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        self.assertIs(CASSCFOTR.ci_vectors(several), several)

    def test_casscf_orbitals_stay_orthonormal(self) -> None:
        mc = casscf_to_otr(mcscf.CASSCF(mf, 4, 4))
        mc.kernel()
        ovlp = mol.intor_symmetric("int1e_ovlp")
        self.assertAlmostEqual(
            abs(
                mc.mo_coeff.T @ ovlp @ mc.mo_coeff - np.eye(mc.mo_coeff.shape[1])
            ).max(),
            0.0,
            8,
        )

    def test_cache_notices_a_changed_ci_vector(self) -> None:
        # for CASSCF the cached update describes a point in the orbitals and the CI
        # vector both, so moving only the CI vector has to invalidate it
        mc = casscf_to_otr(mcscf.CASSCF(mf, 4, 4))
        mc.kernel()
        grad, h_diag = grad_and_h_diag(mc)
        zero = np.zeros(mc.n_param, dtype=np.float64)

        # the kernel canonicalizes, so the first request recomputes and the second is
        # served from the cache; n_update_orbs only counts updates that move the
        # optimization, so a recomputation shows as a new cached result
        mc._update_orbs(zero, grad, h_diag)
        cached = mc._last_update
        mc._update_orbs(zero, grad, h_diag)
        self.assertIs(mc._last_update, cached)

        ci = np.array(mc.ci, copy=True)
        ci[0] = -ci[0]
        mc.ci = ci / np.linalg.norm(ci)
        mc._update_orbs(zero, grad, h_diag)
        self.assertIsNot(mc._last_update, cached)

    def test_counters_record_the_work_done(self) -> None:
        mc = casscf_to_otr(mcscf.CASSCF(mf, 4, 4))
        self.assertEqual(mc.n_update_orbs, 0)
        self.assertEqual(mc.n_hess_x, 0)
        mc.kernel()
        self.assertGreater(mc.n_update_orbs, 0)
        self.assertGreater(mc.n_hess_x, 0)

    def test_derivatives_match_finite_differences(self) -> None:
        check_derivatives_match_finite_differences(
            self, casscf_to_otr(mcscf.CASSCF(mf, 4, 4))
        )

    def test_casscf_stability_check(self) -> None:
        mc = casscf_to_otr(mcscf.CASSCF(mf, 4, 4))
        mc.kernel()
        check_stability_shape(self, mc)
