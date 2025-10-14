"""
Generador de problemas de Job Shop Scheduling.

Este módulo implementa la familia de problemas de planificación de trabajos (Job Shop Scheduling),
un problema clásico de optimización combinatoria donde se deben asignar trabajos a máquinas
respetando restricciones de precedencia y capacidad.

El problema consiste en:
- n trabajos (jobs), cada uno con m operaciones
- m máquinas
- Cada operación requiere una máquina específica y tiene una duración
- Las operaciones de un trabajo deben ejecutarse en orden (precedencia)
- Cada máquina solo puede procesar una operación a la vez

Objetivo: Minimizar el makespan (tiempo total de finalización)

Características:
- Generación de instancias aleatorias
- Instancias clásicas predefinidas (Fisher-Thompson, Lawrence, etc.)
- Restricciones de precedencia y capacidad
- Validador de soluciones
- Cálculo de makespan

Referencias:
- Garey, M. R., Johnson, D. S., & Sethi, R. (1976). The complexity of flowshop and jobshop scheduling
- Fisher, H., & Thompson, G. L. (1963). Probabilistic learning combinations of local job-shop scheduling rules
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import random

from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.problems.base import ProblemFamily
from lattice_weaver.problems.catalog import register_family

logger = logging.getLogger(__name__)


# Instancias clásicas de Job Shop Scheduling
# Formato: (n_jobs, n_machines, operations)
# operations = lista de trabajos, cada trabajo es lista de (máquina, duración)
CLASSIC_INSTANCES = {
    'ft06': {
        'n_jobs': 6,
        'n_machines': 6,
        'operations': [
            [(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
            [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
            [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
            [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
            [(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
            [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)],
        ]
    },
    'ft10': {
        'n_jobs': 10,
        'n_machines': 10,
        'operations': [
            [(0, 29), (1, 78), (2, 9), (3, 36), (4, 49), (5, 11), (6, 62), (7, 56), (8, 44), (9, 21)],
            [(0, 43), (2, 90), (4, 75), (9, 11), (3, 69), (1, 28), (6, 46), (5, 46), (7, 72), (8, 30)],
            [(1, 91), (0, 85), (3, 39), (2, 74), (8, 90), (5, 10), (7, 12), (6, 89), (9, 45), (4, 33)],
            [(1, 81), (2, 95), (0, 71), (4, 99), (6, 9), (8, 52), (7, 85), (3, 98), (9, 22), (5, 43)],
            [(2, 14), (0, 6), (1, 22), (5, 61), (3, 26), (4, 69), (8, 21), (7, 49), (9, 72), (6, 53)],
            [(2, 84), (1, 2), (5, 52), (3, 95), (8, 48), (9, 72), (0, 47), (6, 65), (4, 6), (7, 25)],
            [(1, 46), (0, 37), (3, 61), (2, 13), (6, 32), (5, 21), (9, 32), (8, 89), (7, 30), (4, 55)],
            [(2, 31), (0, 86), (1, 46), (5, 74), (4, 32), (6, 88), (8, 19), (9, 48), (7, 36), (3, 79)],
            [(0, 76), (1, 69), (3, 76), (5, 51), (2, 85), (9, 11), (6, 40), (7, 89), (4, 26), (8, 74)],
            [(1, 85), (0, 13), (2, 61), (6, 7), (8, 64), (9, 76), (5, 47), (3, 52), (4, 90), (7, 45)],
        ]
    },
}


class JobShopSchedulingProblem(ProblemFamily):
    """
    Familia de problemas de Job Shop Scheduling.
    
    El problema consiste en asignar trabajos a máquinas respetando:
    - Precedencia: operaciones de un trabajo deben ejecutarse en orden
    - Capacidad: cada máquina procesa una operación a la vez
    - No preemption: una operación no puede interrumpirse
    
    Parámetros:
        instance_name (str): Nombre de instancia clásica o 'random' para generar aleatoria
        n_jobs (int): Número de trabajos (solo para instancias aleatorias)
        n_machines (int): Número de máquinas (solo para instancias aleatorias)
        max_duration (int): Duración máxima de operaciones (solo para aleatorias)
        seed (int): Semilla para generación aleatoria (opcional)
    
    Instancias clásicas:
        - 'ft06': Fisher-Thompson 6x6 (6 trabajos, 6 máquinas)
        - 'ft10': Fisher-Thompson 10x10 (10 trabajos, 10 máquinas)
        - 'random': Instancia aleatoria
    
    Nota: Este CSP modela la asignación de slots de tiempo a operaciones.
    Variables: operation_{job}_{op_index} -> slot de tiempo
    Restricciones: precedencia, capacidad de máquinas
    
    Ejemplo:
        >>> from lattice_weaver.problems import get_catalog
        >>> catalog = get_catalog()
        >>> engine = catalog.generate_problem('scheduling', instance_name='ft06')
        >>> # Resolver y obtener makespan
        >>> metadata = catalog.get_metadata('scheduling', instance_name='ft06')
        >>> print(f"Makespan óptimo conocido: {metadata['optimal_makespan']}")
    """
    
    def __init__(self):
        super().__init__(
            name='scheduling',
            description='Job Shop Scheduling - asignar trabajos a máquinas respetando precedencia y capacidad'
        )
        logger.info(f"Inicializada familia de problemas: {self.name}")
    
    def get_default_params(self) -> Dict[str, Any]:
        """Retorna parámetros por defecto."""
        return {
            'instance_name': 'ft06',
        }
    
    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna el esquema de parámetros para validación.
        
        Returns:
            Diccionario con especificación de parámetros
        """
        return {
            'instance_name': {
                'type': str,
                'required': True,
                'choices': list(CLASSIC_INSTANCES.keys()) + ['random'],
                'description': 'Nombre de instancia clásica o "random" para generar aleatoria'
            },
            'n_jobs': {
                'type': int,
                'required': False,
                'min': 2,
                'max': 20,
                'description': 'Número de trabajos (solo para instancias aleatorias)'
            },
            'n_machines': {
                'type': int,
                'required': False,
                'min': 2,
                'max': 20,
                'description': 'Número de máquinas (solo para instancias aleatorias)'
            },
            'max_duration': {
                'type': int,
                'required': False,
                'min': 1,
                'max': 100,
                'default': 10,
                'description': 'Duración máxima de operaciones (solo para aleatorias)'
            },
            'seed': {
                'type': int,
                'required': False,
                'description': 'Semilla para generación aleatoria'
            }
        }
    
    def generate(self, **params) -> CSP:
        """
        Genera un problema de Job Shop Scheduling.
        
        Args:
            **params: Parámetros del problema
        
        Returns:
            ArcEngine con el problema configurado
        """
        # Validar parámetros
        self.validate_params(**params)
        
        instance_name = params['instance_name']
        seed = params.get('seed', None)
        
        if seed is not None:
            random.seed(seed)
        
        logger.info(f"Generando problema Job Shop Scheduling: instance={instance_name}")
        
        # Obtener instancia (clásica o aleatoria)
        if instance_name == 'random':
            n_jobs = params.get('n_jobs', 5)
            n_machines = params.get('n_machines', 5)
            max_duration = params.get('max_duration', 10)
            operations = self._generate_random_instance(n_jobs, n_machines, max_duration, seed)
        else:
            instance = CLASSIC_INSTANCES[instance_name]
            n_jobs = instance['n_jobs']
            n_machines = instance['n_machines']
            operations = instance['operations']
        
        # Calcular horizonte de tiempo (upper bound del makespan)
        total_duration = sum(sum(duration for _, duration in job) for job in operations)
        horizon = total_duration  # Cota superior: todos los trabajos en secuencia
        
        # Crear ArcEngine
        csp_problem = CSP(variables=set(), domains={}, constraints=[], name=f"JobShopScheduling_{instance_name}")
        
        # Variables: operation_{job}_{op_index} -> tiempo de inicio
        # Dominio: [0, horizon - duration]
        for job_id, job_ops in enumerate(operations):
            for op_index, (machine, duration) in enumerate(job_ops):
                var_name = f'op_{job_id}_{op_index}'
                # Dominio: tiempos de inicio válidos
                domain = list(range(horizon - duration + 1))
                csp_problem.add_variable(var_name, domain)
        
        logger.debug(f"Añadidas {sum(len(job) for job in operations)} variables (operaciones)")
        
        # Restricciones de precedencia: op_i debe terminar antes de que op_{i+1} comience
        constraint_count = 0
        for job_id, job_ops in enumerate(operations):
            for op_index in range(len(job_ops) - 1):
                var1 = f'op_{job_id}_{op_index}'
                var2 = f'op_{job_id}_{op_index + 1}'
                duration1 = job_ops[op_index][1]
                
                # Crear closure para capturar duration1
                def make_precedence_constraint(d):
                    def precedence(t1, t2):
                        return t1 + d <= t2
                    return precedence
                
                constraint = make_precedence_constraint(duration1)
                constraint.__name__ = f'prec_{job_id}_{op_index}'
                cid = f'precedence_{job_id}_{op_index}'
                csp_problem.add_constraint(Constraint(scope=frozenset({var1, var2}), relation=constraint, name=cid))
                constraint_count += 1
        
        logger.debug(f"Añadidas {constraint_count} restricciones de precedencia")
        
        # Restricciones de capacidad: operaciones en la misma máquina no pueden solaparse
        # Agrupar operaciones por máquina
        machine_ops = {}
        for job_id, job_ops in enumerate(operations):
            for op_index, (machine, duration) in enumerate(job_ops):
                if machine not in machine_ops:
                    machine_ops[machine] = []
                machine_ops[machine].append((job_id, op_index, duration))
        
        # Añadir restricciones de no solapamiento para cada máquina
        for machine, ops in machine_ops.items():
            for i in range(len(ops)):
                for j in range(i + 1, len(ops)):
                    job1, op1, dur1 = ops[i]
                    job2, op2, dur2 = ops[j]
                    var1 = f'op_{job1}_{op1}'
                    var2 = f'op_{job2}_{op2}'
                    
                    # Crear closure para capturar duraciones
                    def make_no_overlap_constraint(d1, d2):
                        def no_overlap(t1, t2):
                            # op1 termina antes de que op2 comience O op2 termina antes de que op1 comience
                            return (t1 + d1 <= t2) or (t2 + d2 <= t1)
                        return no_overlap
                    
                    constraint = make_no_overlap_constraint(dur1, dur2)
                    constraint.__name__ = f'no_overlap_m{machine}_{job1}_{op1}_{job2}_{op2}'
                    cid = f'capacity_m{machine}_{job1}_{op1}_{job2}_{op2}'
                    csp_problem.add_constraint(Constraint(scope=frozenset({var1, var2}), relation=constraint, name=cid))
                    constraint_count += 1
        
        logger.info(f"Problema Job Shop Scheduling generado: {n_jobs} trabajos, {n_machines} máquinas, {constraint_count} restricciones")
        
        return csp_problem
    
    def _generate_random_instance(self, n_jobs: int, n_machines: int, max_duration: int, seed: Optional[int] = None) -> List[List[Tuple[int, int]]]:
        """
        Genera una instancia aleatoria de Job Shop Scheduling.
        
        Args:
            n_jobs: Número de trabajos
            n_machines: Número de máquinas
            max_duration: Duración máxima de operaciones
            seed: Semilla para reproducibilidad
        
        Returns:
            Lista de trabajos, cada trabajo es lista de (máquina, duración)
        """
        if seed is not None:
            random.seed(seed)
        
        operations = []
        for _ in range(n_jobs):
            # Cada trabajo tiene n_machines operaciones (una por máquina)
            # Orden de máquinas aleatorio
            machines = list(range(n_machines))
            random.shuffle(machines)
            
            job_ops = []
            for machine in machines:
                duration = random.randint(1, max_duration)
                job_ops.append((machine, duration))
            
            operations.append(job_ops)
        
        return operations
    
    def validate_solution(self, solution: Dict[str, int], **params) -> bool:
        """
        Valida una solución del problema de Job Shop Scheduling.
        
        Args:
            solution: Diccionario {operation: tiempo_inicio}
            **params: Parámetros del problema
        
        Returns:
            True si la solución es válida, False en caso contrario
        """
        instance_name = params['instance_name']
        seed = params.get('seed', None)
        
        # Obtener instancia
        if instance_name == 'random':
            n_jobs = params.get('n_jobs', 5)
            n_machines = params.get('n_machines', 5)
            max_duration = params.get('max_duration', 10)
            operations = self._generate_random_instance(n_jobs, n_machines, max_duration, seed)
        else:
            instance = CLASSIC_INSTANCES[instance_name]
            operations = instance['operations']
        
        # Verificar que todas las operaciones estén asignadas
        expected_ops = set()
        for job_id, job_ops in enumerate(operations):
            for op_index in range(len(job_ops)):
                expected_ops.add(f'op_{job_id}_{op_index}')
        
        if set(solution.keys()) != expected_ops:
            logger.debug("Solución incompleta o con operaciones extra")
            return False
        
        # Verificar restricciones de precedencia
        for job_id, job_ops in enumerate(operations):
            for op_index in range(len(job_ops) - 1):
                var1 = f'op_{job_id}_{op_index}'
                var2 = f'op_{job_id}_{op_index + 1}'
                t1 = solution[var1]
                t2 = solution[var2]
                duration1 = job_ops[op_index][1]
                
                if t1 + duration1 > t2:
                    logger.debug(f"Violación de precedencia: {var1} termina después de que {var2} comienza")
                    return False
        
        # Verificar restricciones de capacidad (no solapamiento en máquinas)
        machine_ops = {}
        for job_id, job_ops in enumerate(operations):
            for op_index, (machine, duration) in enumerate(job_ops):
                if machine not in machine_ops:
                    machine_ops[machine] = []
                var_name = f'op_{job_id}_{op_index}'
                start_time = solution[var_name]
                end_time = start_time + duration
                machine_ops[machine].append((start_time, end_time, var_name))
        
        for machine, ops in machine_ops.items():
            # Ordenar por tiempo de inicio
            ops_sorted = sorted(ops, key=lambda x: x[0])
            
            # Verificar no solapamiento
            for i in range(len(ops_sorted) - 1):
                _, end1, name1 = ops_sorted[i]
                start2, _, name2 = ops_sorted[i + 1]
                
                if end1 > start2:
                    logger.debug(f"Solapamiento en máquina {machine}: {name1} y {name2}")
                    return False
        
        return True
    
    def get_metadata(self, **params) -> Dict[str, Any]:
        """
        Retorna metadatos del problema.
        
        Args:
            **params: Parámetros del problema
        
        Returns:
            Diccionario con metadatos
        """
        instance_name = params['instance_name']
        seed = params.get('seed', None)
        
        # Obtener instancia
        if instance_name == 'random':
            n_jobs = params.get('n_jobs', 5)
            n_machines = params.get('n_machines', 5)
            max_duration = params.get('max_duration', 10)
            operations = self._generate_random_instance(n_jobs, n_machines, max_duration, seed)
        else:
            instance = CLASSIC_INSTANCES[instance_name]
            n_jobs = instance['n_jobs']
            n_machines = instance['n_machines']
            operations = instance['operations']
        
        # Calcular estadísticas
        total_ops = sum(len(job) for job in operations)
        total_duration = sum(sum(duration for _, duration in job) for job in operations)
        avg_duration = total_duration / total_ops if total_ops > 0 else 0
        
        # Calcular número de restricciones
        n_precedence = sum(len(job) - 1 for job in operations)
        
        # Calcular restricciones de capacidad
        machine_ops_count = {}
        for job_ops in operations:
            for machine, _ in job_ops:
                machine_ops_count[machine] = machine_ops_count.get(machine, 0) + 1
        
        n_capacity = sum(count * (count - 1) // 2 for count in machine_ops_count.values())
        
        # Makespan óptimo conocido para instancias clásicas
        optimal_makespan = {
            'ft06': 55,
            'ft10': 930,
        }.get(instance_name, None)
        
        # Estimar dificultad
        if n_jobs <= 5 and n_machines <= 5:
            difficulty = 'easy'
        elif n_jobs <= 10 and n_machines <= 10:
            difficulty = 'medium'
        else:
            difficulty = 'hard'
        
        metadata = {
            'family': self.name,
            'instance_name': instance_name,
            'n_jobs': n_jobs,
            'n_machines': n_machines,
            'n_operations': total_ops,
            'n_variables': total_ops,
            'n_precedence_constraints': n_precedence,
            'n_capacity_constraints': n_capacity,
            'n_constraints': n_precedence + n_capacity,
            'total_duration': total_duration,
            'avg_operation_duration': round(avg_duration, 2),
            'complexity': 'NP-hard',
            'problem_type': 'scheduling',
            'difficulty': difficulty
        }
        
        if optimal_makespan is not None:
            metadata['optimal_makespan'] = optimal_makespan
        
        return metadata


# Auto-registrar la familia en el catálogo global
register_family(JobShopSchedulingProblem())

logger.info("Familia JobShopSchedulingProblem registrada en el catálogo global")

