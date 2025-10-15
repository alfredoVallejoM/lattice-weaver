"""
Adaptador CSP-to-FCA: Conversión de CSP a Contexto Formal

Este módulo proporciona herramientas para convertir un CSP en un contexto formal
de FCA, permitiendo análisis estructural y detección de implicaciones.

Autor: Manus AI
Fecha: 15 de Octubre, 2025
"""

from typing import Dict, Set, List, Any, Tuple, FrozenSet
from ..csp_problem import CSP, Constraint
from ...lattice_core.context import FormalContext
from ...lattice_core.builder import LatticeBuilder


class CSPToFCAAdapter:
    """
    Adaptador para convertir un CSP en un contexto formal de FCA.
    
    El contexto formal se construye de la siguiente manera:
    - **Objetos (G)**: Variables del CSP
    - **Atributos (M)**: Propiedades derivadas de:
      * Tamaño del dominio (small, medium, large)
      * Grado de conectividad (low, medium, high degree)
      * Tipo de restricciones involucradas
    - **Incidencias (I)**: Variable tiene propiedad
    
    Esto permite analizar la estructura del CSP y detectar implicaciones
    que pueden simplificar el problema.
    """
    
    def __init__(self, csp: CSP):
        """
        Inicializa el adaptador.
        
        Args:
            csp: Problema CSP a convertir
        """
        self.csp = csp
        self.context: FormalContext = None
        self.lattice_builder: LatticeBuilder = None
        self._degree_cache: Dict[str, int] = {}
    
    def build_context(self) -> FormalContext:
        """
        Construye el contexto formal a partir del CSP.
        
        Returns:
            Contexto formal construido
        """
        context = FormalContext()
        
        # Añadir variables como objetos
        for var in self.csp.variables:
            context.add_object(var)
        
        # Calcular grados de las variables (número de restricciones)
        self._compute_degrees()
        
        # Añadir atributos basados en tamaño de dominio
        self._add_domain_attributes(context)
        
        # Añadir atributos basados en grado de conectividad
        self._add_degree_attributes(context)
        
        # Añadir atributos basados en tipos de restricciones
        self._add_constraint_type_attributes(context)
        
        self.context = context
        return context
    
    def _compute_degrees(self):
        """Calcula el grado de cada variable (número de restricciones)."""
        self._degree_cache.clear()
        
        for var in self.csp.variables:
            degree = sum(1 for c in self.csp.constraints if var in c.scope)
            self._degree_cache[var] = degree
    
    def _add_domain_attributes(self, context: FormalContext):
        """
        Añade atributos basados en el tamaño del dominio.
        
        Categorías:
        - domain_size_1: Dominio de tamaño 1 (ya asignado)
        - domain_size_small: Dominio de tamaño 2-5
        - domain_size_medium: Dominio de tamaño 6-20
        - domain_size_large: Dominio de tamaño > 20
        """
        for var in self.csp.variables:
            domain_size = len(self.csp.domains[var])
            
            if domain_size == 1:
                context.add_incidence(var, 'domain_size_1')
            elif domain_size <= 5:
                context.add_incidence(var, 'domain_size_small')
            elif domain_size <= 20:
                context.add_incidence(var, 'domain_size_medium')
            else:
                context.add_incidence(var, 'domain_size_large')
    
    def _add_degree_attributes(self, context: FormalContext):
        """
        Añade atributos basados en el grado de conectividad.
        
        Categorías:
        - degree_isolated: Grado 0 (sin restricciones)
        - degree_low: Grado 1-2
        - degree_medium: Grado 3-5
        - degree_high: Grado > 5
        """
        for var in self.csp.variables:
            degree = self._degree_cache[var]
            
            if degree == 0:
                context.add_incidence(var, 'degree_isolated')
            elif degree <= 2:
                context.add_incidence(var, 'degree_low')
            elif degree <= 5:
                context.add_incidence(var, 'degree_medium')
            else:
                context.add_incidence(var, 'degree_high')
    
    def _add_constraint_type_attributes(self, context: FormalContext):
        """
        Añade atributos basados en tipos de restricciones.
        
        Categorías:
        - has_unary_constraint: Tiene restricción unaria
        - has_binary_constraint: Tiene restricción binaria
        - has_global_constraint: Tiene restricción global (n-aria)
        """
        for var in self.csp.variables:
            has_unary = False
            has_binary = False
            has_global = False
            
            for constraint in self.csp.constraints:
                if var not in constraint.scope:
                    continue
                
                scope_size = len(constraint.scope)
                if scope_size == 1:
                    has_unary = True
                elif scope_size == 2:
                    has_binary = True
                else:
                    has_global = True
            
            if has_unary:
                context.add_incidence(var, 'has_unary_constraint')
            if has_binary:
                context.add_incidence(var, 'has_binary_constraint')
            if has_global:
                context.add_incidence(var, 'has_global_constraint')
    
    def build_lattice(self) -> List[Tuple[FrozenSet, FrozenSet]]:
        """
        Construye el retículo de conceptos del CSP.
        
        Returns:
            Lista de conceptos formales (extent, intent)
        """
        if self.context is None:
            self.build_context()
        
        self.lattice_builder = LatticeBuilder(self.context)
        return self.lattice_builder.build_lattice()
    
    def extract_implications(self) -> List[Tuple[FrozenSet, FrozenSet]]:
        """
        Extrae implicaciones del retículo de conceptos.
        
        Una implicación A → B significa que si una variable tiene
        todas las propiedades en A, entonces también tiene todas
        las propiedades en B.
        
        Returns:
            Lista de implicaciones (antecedente, consecuente)
        """
        if self.lattice_builder is None:
            self.build_lattice()
        
        implications = []
        concepts = self.lattice_builder.concepts
        
        # Para cada par de conceptos (c1, c2)
        for i, (extent1, intent1) in enumerate(concepts):
            for j, (extent2, intent2) in enumerate(concepts):
                if i == j:
                    continue
                
                # Si intent1 ⊂ intent2 (estricto), entonces intent1 → intent2
                if intent1 < intent2:  # Subconjunto estricto
                    # La implicación es: intent1 → (intent2 - intent1)
                    consequent = intent2 - intent1
                    if consequent:  # Solo si hay algo nuevo
                        implications.append((intent1, consequent))
        
        return implications
    
    def get_variable_properties(self, var: str) -> FrozenSet[str]:
        """
        Obtiene las propiedades de una variable.
        
        Args:
            var: Nombre de la variable
        
        Returns:
            Conjunto de propiedades de la variable
        """
        if self.context is None:
            self.build_context()
        
        return self.context.prime_objects({var})
    
    def get_variables_with_properties(self, properties: Set[str]) -> FrozenSet[str]:
        """
        Obtiene las variables que tienen todas las propiedades dadas.
        
        Args:
            properties: Conjunto de propiedades
        
        Returns:
            Conjunto de variables que tienen todas las propiedades
        """
        if self.context is None:
            self.build_context()
        
        return self.context.prime_attributes(properties)
    
    def get_degree(self, var: str) -> int:
        """
        Obtiene el grado de una variable (número de restricciones).
        
        Args:
            var: Nombre de la variable
        
        Returns:
            Grado de la variable
        """
        if not self._degree_cache:
            self._compute_degrees()
        
        return self._degree_cache.get(var, 0)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen del análisis FCA del CSP.
        
        Returns:
            Diccionario con estadísticas del análisis
        """
        if self.context is None:
            self.build_context()
        
        if self.lattice_builder is None:
            self.build_lattice()
        
        implications = self.extract_implications()
        
        return {
            'num_variables': len(self.csp.variables),
            'num_attributes': len(self.context.attributes),
            'num_concepts': len(self.lattice_builder.concepts),
            'num_implications': len(implications),
            'avg_degree': sum(self._degree_cache.values()) / len(self._degree_cache) if self._degree_cache else 0,
            'max_degree': max(self._degree_cache.values()) if self._degree_cache else 0,
            'min_degree': min(self._degree_cache.values()) if self._degree_cache else 0,
        }


def analyze_csp_structure(csp: CSP) -> Dict[str, Any]:
    """
    Función de conveniencia para analizar la estructura de un CSP usando FCA.
    
    Args:
        csp: Problema CSP a analizar
    
    Returns:
        Diccionario con el resumen del análisis
    """
    adapter = CSPToFCAAdapter(csp)
    return adapter.get_summary()

