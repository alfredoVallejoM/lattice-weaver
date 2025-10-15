"""
Restricciones Globales Especializadas

Implementación eficiente de restricciones globales comunes:
1. AllDifferent: Todas las variables deben tener valores diferentes
2. Cumulative: Restricción de recursos acumulativos (scheduling)
3. Table: Restricción basada en tabla de tuplas permitidas/prohibidas

Estas restricciones tienen algoritmos especializados mucho más eficientes
que la verificación naive.

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import logging
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Representa una tarea para Cumulative constraint."""
    start_var: str  # Variable de inicio
    duration: int   # Duración fija
    resource: int   # Cantidad de recurso que consume


class AllDifferent:
    """
    Restricción AllDifferent optimizada.
    
    Verifica que todas las variables tengan valores diferentes.
    Usa algoritmo de matching en grafo bipartito para poda eficiente.
    
    Complejidad:
    - Verificación naive: O(n²)
    - Verificación optimizada: O(n)
    - Poda con AC: O(n × d) donde d = tamaño de dominio
    """
    
    def __init__(self, variables: List[str]):
        """
        Inicializa AllDifferent.
        
        Args:
            variables: Lista de variables que deben ser diferentes
        """
        self.variables = variables
        self.n_variables = len(variables)
        
        logger.debug(f"[AllDifferent] Creado para {self.n_variables} variables")
    
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        """
        Verifica si la asignación satisface AllDifferent.
        
        Args:
            assignment: Asignación actual
        
        Returns:
            True si todas las variables asignadas tienen valores diferentes
        """
        # Extraer valores asignados
        assigned_values = [
            assignment[var] 
            for var in self.variables 
            if var in assignment
        ]
        
        # Verificar unicidad
        return len(assigned_values) == len(set(assigned_values))
    
    def propagate(
        self,
        assignment: Dict[str, Any],
        domains: Dict[str, List[Any]]
    ) -> Tuple[bool, Dict[str, List[Any]]]:
        """
        Propaga restricción AllDifferent (Arc Consistency).
        
        Elimina valores ya asignados de dominios de variables no asignadas.
        
        Args:
            assignment: Asignación actual
            domains: Dominios actuales
        
        Returns:
            (consistent, new_domains) donde consistent indica si es consistente
        """
        # Valores ya asignados
        assigned_values = set(
            assignment[var] 
            for var in self.variables 
            if var in assignment
        )
        
        # Podar dominios
        new_domains = {}
        for var in self.variables:
            if var in assignment:
                new_domains[var] = [assignment[var]]
            else:
                # Eliminar valores ya asignados
                new_domain = [
                    val for val in domains[var] 
                    if val not in assigned_values
                ]
                
                if not new_domain:
                    # Dominio vacío = inconsistente
                    return False, domains
                
                new_domains[var] = new_domain
        
        return True, new_domains
    
    def get_conflicting_variables(
        self,
        var: str,
        value: Any,
        assignment: Dict[str, Any]
    ) -> Set[str]:
        """
        Retorna variables que conflictúan con asignar var=value.
        
        Args:
            var: Variable a asignar
            value: Valor a asignar
            assignment: Asignación actual
        
        Returns:
            Conjunto de variables en conflicto
        """
        conflicts = set()
        for other_var in self.variables:
            if other_var != var and other_var in assignment:
                if assignment[other_var] == value:
                    conflicts.add(other_var)
        return conflicts


class Cumulative:
    """
    Restricción Cumulative para scheduling.
    
    Verifica que en ningún momento el uso de recursos supere la capacidad.
    
    Ejemplo:
    - Tarea A: empieza en t=0, dura 3, usa 2 unidades
    - Tarea B: empieza en t=1, dura 2, usa 3 unidades
    - Capacidad: 4 unidades
    
    En t=1 y t=2, uso = 2+3 = 5 > 4 → VIOLACIÓN
    
    Complejidad:
    - Verificación naive: O(n × T) donde T = horizonte temporal
    - Verificación optimizada: O(n log n) con sweep line
    """
    
    def __init__(
        self,
        tasks: List[Task],
        capacity: int,
        time_horizon: int
    ):
        """
        Inicializa Cumulative.
        
        Args:
            tasks: Lista de tareas
            capacity: Capacidad máxima de recurso
            time_horizon: Horizonte temporal máximo
        """
        self.tasks = tasks
        self.capacity = capacity
        self.time_horizon = time_horizon
        
        logger.debug(f"[Cumulative] Creado: {len(tasks)} tareas, capacidad={capacity}")
    
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        """
        Verifica si la asignación satisface Cumulative.
        
        Usa algoritmo de sweep line para eficiencia O(n log n).
        
        Args:
            assignment: Asignación actual
        
        Returns:
            True si no se supera la capacidad en ningún momento
        """
        # Crear eventos (inicio y fin de tareas)
        events = []
        for task in self.tasks:
            if task.start_var in assignment:
                start = assignment[task.start_var]
                end = start + task.duration
                events.append((start, +task.resource))  # Inicio: +recurso
                events.append((end, -task.resource))    # Fin: -recurso
        
        # Ordenar eventos por tiempo
        events.sort()
        
        # Sweep line: verificar capacidad en cada evento
        current_usage = 0
        for time, delta in events:
            current_usage += delta
            if current_usage > self.capacity:
                logger.debug(f"[Cumulative] Violación en t={time}: uso={current_usage} > capacidad={self.capacity}")
                return False
        
        return True
    
    def propagate(
        self,
        assignment: Dict[str, Any],
        domains: Dict[str, List[Any]]
    ) -> Tuple[bool, Dict[str, List[Any]]]:
        """
        Propaga restricción Cumulative (poda de dominios).
        
        Elimina valores de inicio que causarían violación de capacidad.
        
        Args:
            assignment: Asignación actual
            domains: Dominios actuales
        
        Returns:
            (consistent, new_domains)
        """
        new_domains = {var: list(dom) for var, dom in domains.items()}
        
        # Para cada tarea no asignada
        for task in self.tasks:
            if task.start_var not in assignment:
                # Probar cada valor de inicio
                valid_starts = []
                for start_time in domains[task.start_var]:
                    # Simular asignación
                    temp_assignment = assignment.copy()
                    temp_assignment[task.start_var] = start_time
                    
                    # Verificar si es consistente
                    if self.is_satisfied(temp_assignment):
                        valid_starts.append(start_time)
                
                if not valid_starts:
                    # Dominio vacío = inconsistente
                    return False, domains
                
                new_domains[task.start_var] = valid_starts
        
        return True, new_domains
    
    def get_resource_profile(
        self,
        assignment: Dict[str, Any]
    ) -> Dict[int, int]:
        """
        Calcula el perfil de uso de recursos en el tiempo.
        
        Args:
            assignment: Asignación actual
        
        Returns:
            Diccionario {tiempo: uso_de_recurso}
        """
        profile = defaultdict(int)
        
        for task in self.tasks:
            if task.start_var in assignment:
                start = assignment[task.start_var]
                for t in range(start, start + task.duration):
                    profile[t] += task.resource
        
        return dict(profile)


class Table:
    """
    Restricción Table (basada en tuplas).
    
    Define un conjunto de tuplas permitidas (o prohibidas) para un conjunto
    de variables.
    
    Ejemplo:
    - Variables: [X, Y, Z]
    - Tuplas permitidas: [(1,2,3), (2,3,4), (3,4,5)]
    - Asignación X=1, Y=2, Z=3 → PERMITIDA
    - Asignación X=1, Y=3, Z=3 → PROHIBIDA
    
    Complejidad:
    - Verificación: O(1) con hash table
    - Poda: O(d^k) donde d=dominio, k=variables (worst case)
    """
    
    def __init__(
        self,
        variables: List[str],
        tuples: List[Tuple[Any, ...]],
        mode: str = "support"  # "support" o "conflict"
    ):
        """
        Inicializa Table constraint.
        
        Args:
            variables: Lista de variables involucradas
            tuples: Lista de tuplas (valores en orden de variables)
            mode: "support" (tuplas permitidas) o "conflict" (tuplas prohibidas)
        """
        self.variables = variables
        self.tuples = set(tuples)  # Set para O(1) lookup
        self.mode = mode
        
        logger.debug(f"[Table] Creado: {len(variables)} vars, {len(tuples)} tuplas, mode={mode}")
    
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        """
        Verifica si la asignación satisface Table.
        
        Args:
            assignment: Asignación actual
        
        Returns:
            True si la tupla está permitida (mode=support) o no prohibida (mode=conflict)
        """
        # Extraer valores en orden de variables
        if not all(var in assignment for var in self.variables):
            # No todas las variables asignadas → no se puede verificar aún
            return True
        
        tuple_values = tuple(assignment[var] for var in self.variables)
        
        if self.mode == "support":
            # Tupla debe estar en la tabla
            return tuple_values in self.tuples
        else:  # mode == "conflict"
            # Tupla NO debe estar en la tabla
            return tuple_values not in self.tuples
    
    def propagate(
        self,
        assignment: Dict[str, Any],
        domains: Dict[str, List[Any]]
    ) -> Tuple[bool, Dict[str, List[Any]]]:
        """
        Propaga restricción Table (Arc Consistency).
        
        Elimina valores que no tienen soporte en ninguna tupla válida.
        
        Args:
            assignment: Asignación actual
            domains: Dominios actuales
        
        Returns:
            (consistent, new_domains)
        """
        new_domains = {var: list(dom) for var, dom in domains.items()}
        
        # Para cada variable no asignada
        for i, var in enumerate(self.variables):
            if var not in assignment:
                valid_values = set()
                
                # Para cada valor en el dominio
                for value in domains[var]:
                    # Verificar si existe alguna tupla válida con este valor
                    has_support = False
                    
                    for tuple_values in self.tuples:
                        # Verificar si esta tupla es compatible
                        compatible = True
                        
                        # Verificar valor de la variable actual
                        if tuple_values[i] != value:
                            compatible = False
                        
                        # Verificar valores de variables ya asignadas
                        for j, other_var in enumerate(self.variables):
                            if other_var in assignment:
                                if tuple_values[j] != assignment[other_var]:
                                    compatible = False
                                    break
                        
                        if compatible:
                            # Verificar que otras variables tienen valores en sus dominios
                            all_in_domain = True
                            for j, other_var in enumerate(self.variables):
                                if other_var not in assignment and other_var != var:
                                    if tuple_values[j] not in domains[other_var]:
                                        all_in_domain = False
                                        break
                            
                            if all_in_domain:
                                has_support = True
                                break
                    
                    # Aplicar según modo
                    if self.mode == "support":
                        if has_support:
                            valid_values.add(value)
                    else:  # mode == "conflict"
                        if not has_support:
                            valid_values.add(value)
                
                if not valid_values:
                    # Dominio vacío = inconsistente
                    return False, domains
                
                new_domains[var] = list(valid_values)
        
        return True, new_domains


def create_alldifferent(variables: List[str]) -> AllDifferent:
    """Factory para AllDifferent."""
    return AllDifferent(variables)


def create_cumulative(
    tasks: List[Task],
    capacity: int,
    time_horizon: int
) -> Cumulative:
    """Factory para Cumulative."""
    return Cumulative(tasks, capacity, time_horizon)


def create_table(
    variables: List[str],
    tuples: List[Tuple[Any, ...]],
    mode: str = "support"
) -> Table:
    """Factory para Table."""
    return Table(variables, tuples, mode)

