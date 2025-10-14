'''
Validador para el Nivel de Renormalización Computacional.

Este módulo define el `RenormalizationValidator` que se encarga de verificar
la corrección, eficiencia y coherencia de las transformaciones de renormalización.
'''

from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
import time
import hashlib
import numpy as np
import random
import traceback
from ..renormalization.core import renormalize_csp, RenormalizationSolver, refine_solution
from ..paging.page_manager import PageManager
from ..renormalization.partition import VariablePartitioner
from ..core.csp_problem import CSP, Constraint, is_satisfiable, verify_solution, solve_subproblem_exhaustive, generate_nqueens, generate_random_csp
from ..core.simple_backtracking_solver import generate_solutions_backtracking
from .certificates import ValidationCertificate, create_certificate, CertificateRepository


class RenormalizationValidator:
    '''
    Valida la corrección de las operaciones de renormalización.
    
    Verifica que la renormalización preserva la satisfacibilidad y las soluciones,
    y que efectivamente reduce la complejidad del problema.
    '''
    
    def __init__(self, 
                 original_csp: CSP,
                 renormalized_csp: CSP,
                 partition: List[Set[str]],
                 k: int,
                 partition_strategy: str = 'metis',
                 page_manager: Optional[PageManager] = None):
        '''
        Inicializa el validador de renormalización.
        
        Args:
            original_csp: El CSP original antes de la renormalización.
            renormalized_csp: El CSP resultante después de la renormalización.
            partition: La partición de variables utilizada.
            k: El factor de renormalización (número de grupos).
            partition_strategy: Estrategia de particionamiento utilizada.
        '''
        self.original_csp = original_csp
        self.renormalized_csp = renormalized_csp
        self.partition = partition
        self.k = k
        self.partition_strategy = partition_strategy
        self.page_manager = page_manager
        
        self._metrics: Dict[str, Any] = {}

    def validate_satisfiability_preservation(self, num_samples: int = 5) -> bool:
        '''
        Valida que la satisfacibilidad se preserva.
        
        Compara la satisfacibilidad del CSP original y el renormalizado.
        '''
        sat_original = is_satisfiable(self.original_csp)
        sat_renormalized = False
        if self.renormalized_csp is not None:
            sat_renormalized = is_satisfiable(self.renormalized_csp)
        
        if sat_original == sat_renormalized:
            return True
        
        return False

    def validate_solution_preservation(self, num_solutions_to_test: int = 5) -> float:
        '''
        Valida que las soluciones se preservan.
        
        Genera soluciones del CSP renormalizado, las refina al original
        y verifica que sean válidas para el CSP original.
        '''
        if self.renormalized_csp is None:
            if not is_satisfiable(self.original_csp):
                return 1.0 # Original CSP is unsatisfiable, and renormalized is None, so it's correct
            else:
                return 0.0 # Original CSP is satisfiable, but renormalized is None, so it's incorrect
        
        # Solo generar soluciones si el CSP renormalizado existe
        renormalized_solutions = []
        if self.renormalized_csp is not None:
            renormalized_solutions = self._generate_renormalized_solutions(self.renormalized_csp, num_solutions_to_test)
        
        if not renormalized_solutions:
            if self.renormalized_csp is not None:
                # Si no se generaron soluciones renormalizadas, pero el CSP renormalizado existe,
                # y el CSP original no es satisfacible, entonces la ausencia de soluciones es correcta.
                # Si el original es satisfacible, pero no se encontraron soluciones renormalizadas, es un fallo.
                if not is_satisfiable(self.original_csp):
                    return 1.0
                else:
                    return 0.0
            # Si renormalized_csp es None, la lógica ya se maneja al principio de la función.
            return 0.0 # Si no hay soluciones renormalizadas y el CSP renormalizado es None, y el original es satisfacible, es un fallo.

        correct_refinements = 0
        for sol_rg in renormalized_solutions:
            try:
                sol_original = refine_solution(sol_rg, self.original_csp, self.partition, page_manager=self.page_manager)
                if verify_solution(self.original_csp, sol_original):
                    correct_refinements += 1
            except ValueError:
                pass
        
        return correct_refinements / len(renormalized_solutions)

    def validate_complexity_reduction(self) -> bool:
        '''
        Valida que la renormalización reduce la complejidad del problema.
        
        Compara el número de variables del CSP original y el renormalizado.
        '''
        num_vars_original = len(self.original_csp.variables)
        if self.renormalized_csp is None:
            return False # No hay reducción si el CSP renormalizado es None
        num_vars_renormalized = len(self.renormalized_csp.variables)
        
        return num_vars_renormalized < num_vars_original

    def validate_effective_domains(self, num_groups_to_test: int = 3) -> float:
        '''
        Valida la corrección de los dominios efectivos.
        
        Para un subconjunto de grupos, verifica que cada configuración en su
        dominio efectivo es realmente válida para el subproblema original.
        '''
        if self.renormalized_csp is None or not self.partition:
            return 1.0 # Si no hay CSP renormalizado o partición, se asume correcto (no hay dominios efectivos que validar)

        groups_to_check = random.sample(self.partition, min(num_groups_to_test, len(self.partition)))
        
        total_configs = 0
        correct_configs = 0

        for group_idx, group in enumerate(groups_to_check):
            group_name = f"G{self.partition.index(group)}"
            effective_domain = self.renormalized_csp.domains.get(group_name)
            
            if not effective_domain:
                continue

            subproblem_vars = sorted(list(group))
            subproblem_domains = {v: self.original_csp.domains[v] for v in subproblem_vars}
            subproblem_constraints = []
            for const in self.original_csp.constraints:
                if all(v in group for v in const.scope):
                    subproblem_constraints.append(const)
            
            original_subproblem = {
                'variables': subproblem_vars,
                'domains': subproblem_domains,
                'constraints': subproblem_constraints
            }

            for config in effective_domain:
                total_configs += 1
                assignment = dict(zip(subproblem_vars, config))
                if verify_solution(CSP(set(subproblem_vars), subproblem_domains, subproblem_constraints), assignment):
                    correct_configs += 1
        
        return correct_configs / total_configs if total_configs > 0 else 1.0

    def measure_performance(self, num_runs: int = 3) -> Tuple[float, float]:
        '''
        Mide el speedup y la reducción de memoria.
        
        Args:
            num_runs: Número de veces que se ejecuta el solver para promediar el tiempo.
        
        Returns:
            Tupla (speedup_medido, memory_reduction_medida).
        '''
        times_original = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = is_satisfiable(self.original_csp)
            times_original.append(time.perf_counter() - start_time)
        avg_time_original = np.mean(times_original)

        times_renormalized = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = is_satisfiable(self.renormalized_csp)
            times_renormalized.append(time.perf_counter() - start_time)
        avg_time_renormalized = np.mean(times_renormalized)

        speedup = avg_time_original / (avg_time_renormalized + 1e-9) if avg_time_renormalized > 0 else float('inf')

        mem_original = len(self.original_csp.variables)
        if self.renormalized_csp is None:
            mem_renormalized = 0
        else:
            mem_renormalized = len(self.renormalized_csp.variables)
        memory_reduction = 1.0 - (mem_renormalized / mem_original) if mem_original > 0 else 0.0
        
        self._metrics["speedup_measured"] = speedup
        self._metrics["memory_reduction"] = memory_reduction
        
        return speedup, memory_reduction

    def generate_certificate(self) -> ValidationCertificate:
        '''
        Genera un `ValidationCertificate` para el nivel de renormalización.
        '''
        sat_preserved = self.validate_satisfiability_preservation()
        sol_preservation_rate = self.validate_solution_preservation()
        complexity_reduced = self.validate_complexity_reduction()
        effective_domains_correctness = self.validate_effective_domains()
        speedup, memory_reduction = self.measure_performance()

        invariants_verified = []
        runtime_tests_passed = 0
        runtime_tests_failed = 0

        if sat_preserved:
            invariants_verified.append("satisfiability_preservation")
            runtime_tests_passed += 1
        else:
            runtime_tests_failed += 1

        if complexity_reduced:
            invariants_verified.append("complexity_reduction")
            runtime_tests_passed += 1
        else:
            runtime_tests_failed += 1
            
        overall_correctness_rate = sol_preservation_rate * effective_domains_correctness

        return create_certificate(
            level_name="renormalization",
            type_checked=True,
            invariants_verified=invariants_verified,
            runtime_tests_passed=runtime_tests_passed,
            runtime_tests_failed=runtime_tests_failed,
            speedup_measured=speedup,
            memory_reduction=memory_reduction,
            correctness_rate=overall_correctness_rate,
            metadata={
                "original_csp_name": self.original_csp.name,
                "k_factor": self.k,
                "partition_strategy": self.partition_strategy,
                "original_vars": len(self.original_csp.variables),
                "renormalized_vars": len(self.renormalized_csp.variables) if self.renormalized_csp else 0,
                "original_constraints": len(self.original_csp.constraints),
                "renormalized_constraints": len(self.renormalized_csp.constraints) if self.renormalized_csp else 0,
                "satisfiability_preserved": sat_preserved,
                "solution_preservation_rate": sol_preservation_rate,
                "complexity_reduced": complexity_reduced,
                "effective_domains_correctness": effective_domains_correctness
            }
        )

    def _generate_renormalized_solutions(self, csp: CSP, num_solutions: int) -> List[Dict[str, Any]]:
        '''
        Genera un número limitado de soluciones para un CSP usando backtracking.
        '''
        # Usar la función generate_solutions_backtracking del módulo simple_backtracking_solver
        return generate_solutions_backtracking(csp, num_solutions=num_solutions)


class RenormalizationTestSuite:
    '''
    Suite de tests para el validador de renormalización.
    '''
    def __init__(self):
        self.cert_repo = CertificateRepository("certificates_data")
        self.test_results: List[Tuple[str, ValidationCertificate]] = []

    def run_test(self, test_name: str, csp_generator: Callable[[], CSP], k: int, partition_strategy: str) -> ValidationCertificate:
        '''
        Ejecuta un test de renormalización y genera un certificado.
        '''
        print(f"\n--- Running Renormalization Test: {test_name} ---")
        original_csp = csp_generator()
        try:
            # Inicializar PageManager para este test
            page_manager = PageManager(l3_storage_dir=f"./page_storage_{test_name.replace(' ', '_')}")

            renormalized_csp, partition = renormalize_csp(
                original_csp,
                k,
                partition_strategy,
                page_manager=page_manager
            )

            if renormalized_csp is None:
                original_satisfiable = is_satisfiable(original_csp)
                if not original_satisfiable:
                    success_cert = create_certificate(
                        level_name="renormalization",
                        type_checked=True,
                        invariants_verified=["satisfiability_preservation"],
                        runtime_tests_passed=1,
                        runtime_tests_failed=0,
                        speedup_measured=0.0,
                        memory_reduction=1.0,
                        correctness_rate=1.0,
                        metadata={
                            "k_factor": k,
                            "partition_strategy": partition_strategy,
                            "original_vars": len(original_csp.variables),
                            "renormalized_vars": 0,
                            "status": "UNSAT_CORRECTLY_IDENTIFIED"
                        }
                    )
                    self.cert_repo.store(success_cert)
                    self.test_results.append((test_name, success_cert))
                    print(f"Test \033[32m\'{test_name}\'\033[0m completed. Certificate generated: {success_cert.signature}")
                    return success_cert
                else:
                    raise ValueError("Renormalization returned None for a satisfiable CSP.")

            validator = RenormalizationValidator(original_csp, renormalized_csp, partition, k, partition_strategy, page_manager=page_manager)
            certificate = validator.generate_certificate()
            self.cert_repo.store(certificate)
            self.test_results.append((test_name, certificate))
            print(f"Test \'{test_name}\' completed. Certificate generated: {certificate.signature}")
            print(certificate)
            return certificate

        except Exception as e:
            error_certificate = create_certificate(
                level_name="renormalization",
                type_checked=False,
                invariants_verified=[],
                runtime_tests_passed=0,
                runtime_tests_failed=1,
                speedup_measured=0.0,
                memory_reduction=0.0,
                correctness_rate=0.0,
                metadata={
                    "test_name": test_name,
                    "original_csp_name": original_csp.name,
                    "k_factor": k,
                    "partition_strategy": partition_strategy,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            self.cert_repo.store(error_certificate)
            self.test_results.append((test_name, error_certificate))
            print(f"Test \033[31m'{test_name}' failed with an exception.\033[0m Certificate generated: {error_certificate.signature}")
            print(error_certificate)
            return error_certificate

    def run_all_tests(self) -> List[ValidationCertificate]:
        print("\n==================================================")
        print("Running Renormalization Validation Test Suite")
        print("==================================================")

        certificates = []

        # Test 1: Simple Renormalization and Refinement Test
        certificates.append(self.run_test(
            "Simple Renormalization and Refinement Test",
            lambda: CSP(
                variables={'x', 'y', 'z'},
                domains={'x': {1, 2}, 'y': {1, 2}, 'z': {1, 2}},
                constraints=[
                    Constraint(scope={'x', 'y'}, relation=lambda x, y: x != y),
                    Constraint(scope={'y', 'z'}, relation=lambda y, z: y == z)
                ],
                name="SimpleCSP"
            ),
            k=2,
            partition_strategy='simple'
        ))

        # Test 2: N-Queens (N=4)
        certificates.append(self.run_test(
            "N-Queens (N=4)",
            lambda: generate_nqueens(4, name="NQueens_4"),
            k=2,
            partition_strategy='topological'
        ))

        # Test 3: Random CSP (10 vars)
        certificates.append(self.run_test(
            "Random CSP (10 vars)",
            lambda: generate_random_csp(10, 3, num_constraints=int(10 * 9 * 0.5), name="RandomCSP_10_vars"),
            k=3,
            partition_strategy='metis'
        ))

        # Test 4: Random CSP (20 vars)
        certificates.append(self.run_test(
            "Random CSP (20 vars)",
            lambda: generate_random_csp(20, 4, num_constraints=int(20 * 19 * 0.4), name="RandomCSP_20_vars"),
            k=4,
            partition_strategy='metis'
        ))

        print("\n==================================================")
        print("Renormalization Validation Test Suite Summary")
        print("==================================================")
        for test_name, cert in self.test_results:
            status_color = "\033[32m" if cert.is_valid() else "\033[31m"
            status_text = "VALID" if cert.is_valid() else "INVALID"
            print(f"- {test_name}: {status_color}{status_text}\033[0m (Correctness: {cert.correctness_rate:.1%}, Speedup: {cert.speedup_measured:.2f}x, Mem. Red.: {cert.memory_reduction:.1%})")

        return certificates


if __name__ == "__main__":
    import traceback
    suite = RenormalizationTestSuite()
    all_certificates = suite.run_all_tests()
    # for cert in all_certificates:
    #     print(cert)



# Helper function to generate solutions for testing


