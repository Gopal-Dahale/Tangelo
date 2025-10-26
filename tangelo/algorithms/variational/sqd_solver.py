# Copyright SandboxAQ 2021-2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements the variational quantum eigensolver (SQD) algorithm to solve
electronic structure calculations.
"""

from contextlib import nullcontext
import warnings
import itertools
from typing import Optional, Union, List

from enum import Enum
import numpy as np

from tangelo.helpers.utils import HiddenPrints
from tangelo import SecondQuantizedMolecule
from tangelo.linq import get_backend, Circuit
from tangelo.linq.helpers.circuits.measurement_basis import measurement_basis_gates
from tangelo.toolboxes.operators import count_qubits, FermionOperator, QubitOperator
from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.toolboxes.qubit_mappings.statevector_mapping import get_mapped_vector, vector_to_circuit
from tangelo.toolboxes.post_processing.bootstrapping import get_resampled_frequencies
from tangelo.toolboxes.optimizers import rotosolve
from tangelo.algorithms.classical.ccsd_solver import CCSDSolver
import tangelo.toolboxes.ansatz_generator as agen

import ffsim
from qiskit import QuantumCircuit, QuantumRegister
from functools import partial
from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian, solve_sci_batch
from qiskit_addon_dice_solver import solve_sci_batch
import pyscf


class SQDSolver:
    r"""Solve the electronic structure problem for a molecular system by using
    the SQD algorithm.

    Users must first set the desired options of the SQDSolver object through the
    __init__ method, and call the "build" method to build the underlying objects
    (mean-field, hardware backend, ansatz...). They are then able to call any of
    the energy_estimation, simulate, or get_rdm methods. In particular, simulate
    runs the SQD algorithm, returning the optimal energy found by the classical
    optimizer.

    Attributes:
        molecule (SecondQuantizedMolecule) : the molecular system.
        qubit_mapping (str) : one of the supported qubit mapping identifiers.
        ansatz (Ansatze) : one of the supported ansatze.
        backend_options (dict): parameters to build the underlying compute backend (simulator, etc).
        simulate_options (dict): Options for fine-control of the simulator backend, including desired measurement results, etc.
        ansatz_options (dict): parameters for the given ansatz (see given ansatz
            file for details).
        up_then_down (bool): change basis ordering putting all spin up orbitals
            first, followed by all spin down. Default, False has alternating
                spin up/down ordering.
        verbose (bool): Flag for SQD verbosity.
        save_energies (bool): Flag for saving energy estimation values.
    """

    def __init__(self, opt_dict):

        default_backend_options = {"target": None, "n_shots": None, "noise_model": None}
        copt_dict = opt_dict.copy()

        self.molecule: Optional[SecondQuantizedMolecule] = copt_dict.pop("molecule", None)
        self.qubit_mapping: str = copt_dict.pop("qubit_mapping", "jw")
        self.ansatz_name: str = copt_dict.pop("ansatz", 'UCJ')
        self.backend_options: dict = copt_dict.pop("backend_options", default_backend_options)
        self.simulate_options: dict = copt_dict.pop("simulate_options", dict())
        self.ansatz_options: dict = copt_dict.pop("ansatz_options", dict())
        self.up_then_down: bool = copt_dict.pop("up_then_down", False)
        self.verbose: bool = copt_dict.pop("verbose", False)
        self.save_energies: bool = copt_dict.pop("save_energies", False)
        self.save_history: bool = copt_dict.pop("save_history", False)

        self.spin = self.molecule.spin
        self.uhf = self.molecule.uhf

        if len(copt_dict) > 0:
            raise KeyError(f"The following keywords are not supported in {self.__class__.__name__}: \n {copt_dict.keys()}")

        # Raise error/warnings if input is not as expected. Only a single input
        # must be provided to avoid conflicts.
        if not bool(self.molecule):
            raise ValueError(f"A molecule object must be provided when instantiating {self.__class__.__name__}.")

        default_backend_options.update(self.backend_options)
        self.backend_options = default_backend_options
        self.optimal_energy = None

        self.energies = list()

    def build(self):
        """Build the underlying objects required to run the SQD algorithm afterwards."""

        if self.ansatz_name == 'UCJ':
            self.ccsd_solver = CCSDSolver(self.molecule)
            _ = self.ccsd_solver.simulate()

            t1 = self.ccsd_solver.solver.cc_fragment.t1
            t2 = self.ccsd_solver.solver.cc_fragment.t2

            num_orbitals = self.molecule.n_active_mos
            n_electrons = self.molecule.n_active_electrons
            num_elec_a = (n_electrons + self.molecule.spin) // 2
            num_elec_b = (n_electrons - self.molecule.spin) // 2

            n_reps = self.ansatz_options.get("n_reps", 1)
            alpha_alpha_indices = [(p, p + 1) for p in range(num_orbitals - 1)]
            alpha_beta_indices = [(p, p) for p in range(0, num_orbitals, 4)]

            ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
                t2=t2,
                # t1=t1,
                n_reps=n_reps,
                interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
            )


            nelec = self.molecule.n_active_ab_electrons

            cas = pyscf.mcscf.CASCI(self.molecule.mean_field, num_orbitals, (num_elec_a, num_elec_b))
            mo = cas.sort_mo(self.molecule.active_mos, base=0)
            self.hcore, self.nuclear_repulsion_energy = cas.get_h1cas(mo)
            self.eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), num_orbitals)

            # create an empty quantum circuit
            qubits = QuantumRegister(2 * num_orbitals, name="q")
            self.ansatz = QuantumCircuit(qubits)

            # prepare Hartree-Fock state as the reference state and append it to the quantum circuit
            self.ansatz.append(ffsim.qiskit.PrepareHartreeFockJW(num_orbitals, nelec), qubits)

            # apply the UCJ operator to the reference state
            self.ansatz.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
            self.ansatz.measure_all()
        
        elif self.ansatz_name == 'Krylov':
            raise NotImplementedError("Krylov ansatz is not yet implemented in SQDSolver.")

        self.ansatz = self.backend_options.get("pass_manager", None).run(self.ansatz)

    def simulate(self):
        """Run the SQD algorithm, using the ansatz, classical optimizer, initial
        parameters and hardware backend built in the build method.
        """
        samples = self.sampling()
        self.optimal_energy = self.energy_estimation(samples)

        return self.optimal_energy

    def get_resources(self):
        """Estimate the resources required by SQD, with the current ansatz. This
        assumes "build" has been run, as it requires the ansatz circuit and the
        qubit Hamiltonian. Return information that pertains to the user, for the
        purpose of running an experiment on a classical simulator or a quantum
        device.
        """

        resources = dict()
        resources["circuit_width"] = self.ansatz.num_qubits
        resources["circuit_depth"] = self.ansatz.depth()
        return resources

    def sampling(self):

        sampler = self.backend_options.get("sampler", None)
        sampler.options.default_shots = self.backend_options.get("n_shots", 1024)

        job = sampler.run([self.ansatz])    
        primitive_result = job.result()
        pub_result = primitive_result[0]
        bit_array = pub_result.data.meas
        return bit_array

    def energy_estimation(self, samples):
        """Estimate energy using the given ansatz, qubit hamiltonian and compute
        backend. Keeps track of optimal energy and variational parameters along
        the way.

        Args:
             samples (numpy.array): Circuit measurement samples to use for energy

        Returns:
             float: energy computed by SQD 
        """
        
        # SQD options
        energy_tol = self.simulate_options.get("energy_tol", 1e-8)
        occupancies_tol = self.simulate_options.get("occupancies_tol", 1e-8)
        max_iterations = self.simulate_options.get("max_iterations", 10)

        # Eigenstate solver options
        num_batches = self.simulate_options.get("num_batches", 1)
        samples_per_batch = self.simulate_options.get("samples_per_batch", 300)
        symmetrize_spin = self.simulate_options.get("symmetrize_spin", True)
        carryover_threshold = self.simulate_options.get("carryover_threshold", 1e-8)
        max_cycle = self.simulate_options.get("max_cycle", 200)


        # Pass options to the built-in eigensolver. If you just want to use the defaults,
        # you can omit this step, in which case you would not specify the sci_solver argument
        # in the call to diagonalize_fermionic_hamiltonian below.
        # sci_solver = partial(solve_sci_batch, spin_sq=0.0, max_cycle=max_cycle)

        # List to capture intermediate results
        result_history = []

        def callback(results: list[SCIResult]):
            result_history.append(results)
            iteration = len(result_history)
            if self.verbose:
                print(f"Iteration {iteration}")
                for i, result in enumerate(results):
                    print(f"\tSubsample {i}")
                    print(f"\t\tEnergy: {result.energy + self.nuclear_repulsion_energy}")
                    print(f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}")


        rng = np.random.default_rng(24)
        result = diagonalize_fermionic_hamiltonian(
            self.hcore,
            self.eri,
            samples,
            samples_per_batch=samples_per_batch,
            norb=self.molecule.n_active_mos,
            nelec=self.molecule.n_active_ab_electrons,
            num_batches=num_batches,
            energy_tol=energy_tol,
            occupancies_tol=occupancies_tol,
            max_iterations=max_iterations,
            sci_solver=solve_sci_batch,
            symmetrize_spin=symmetrize_spin,
            carryover_threshold=carryover_threshold,
            callback=callback,
            seed=rng,
        )
        
        energy = result.energy + self.nuclear_repulsion_energy

        if self.verbose:
            print(f"\tEnergy = {energy:.7f} ")

        if self.save_energies:
            self.energies += [energy]
        
        if self.save_history:
            self.result_history = result_history
            self.final_result = result

        return energy


    def get_rdm(self):
        """Calculate the 1- and 2-particle reduced density matrices. The CCSD
        lambda equation will be solved for calculating the RDMs.

        Returns:
            numpy.array: One-particle RDM.
            numpy.array: Two-particle RDM.

        Raises:
            RuntimeError: If no simulation has been run.
        """
        from pyscf import lib
        from pyscf.cc.ccsd_rdm import _make_rdm1, _make_rdm2, _gamma1_intermediates, _gamma2_outcore
        from pyscf.cc.uccsd_rdm import (_make_rdm1 as _umake_rdm1, _make_rdm2 as _umake_rdm2,
                                        _gamma1_intermediates as _ugamma1_intermediates, _gamma2_outcore as _ugamma2_outcore)

        # Check if CCSD calculation is performed
        if self.ccsd_solver.solver.cc_fragment is None:
            raise RuntimeError("CCSDSolver: Cannot retrieve RDM. Please run the 'simulate' method first")

        # Solve the lambda equation and obtain the reduced density matrix from CC calculation
        t1 = self.ccsd_solver.solver.cc_fragment.t1
        t2 = self.ccsd_solver.solver.cc_fragment.t2
        l1, l2 = self.ccsd_solver.solver.cc_fragment.solve_lambda(t1, t2)

        if self.spin == 0 and not self.uhf:
            d1 = _gamma1_intermediates(self.ccsd_solver.solver.cc_fragment, t1, t2, l1, l2)
            f = lib.H5TmpFile()
            d2 = _gamma2_outcore(self.ccsd_solver.solver.cc_fragment, t1, t2, l1, l2, f, False)

            one_rdm = _make_rdm1(self.ccsd_solver.solver.cc_fragment, d1, with_frozen=False)
            two_rdm = _make_rdm2(self.ccsd_solver.solver.cc_fragment, d1, d2, with_dm1=True, with_frozen=False)
        else:
            d1 = _ugamma1_intermediates(self.ccsd_solver.solver.cc_fragment, t1, t2, l1, l2)
            f = lib.H5TmpFile()
            d2 = _ugamma2_outcore(self.ccsd_solver.solver.cc_fragment, t1, t2, l1, l2, f, False)

            one_rdm = _umake_rdm1(self.ccsd_solver.solver.cc_fragment, d1, with_frozen=False)
            two_rdm = _umake_rdm2(self.ccsd_solver.solver.cc_fragment, d1, d2, with_dm1=True, with_frozen=False)

            if not self.uhf:
                one_rdm = np.sum(one_rdm, axis=0)
                two_rdm = np.sum((two_rdm[0], 2*two_rdm[1], two_rdm[2]), axis=0)

        return one_rdm, two_rdm

    