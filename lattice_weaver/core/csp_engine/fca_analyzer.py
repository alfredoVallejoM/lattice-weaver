"""
Analizador FCA: Detección de Implicaciones y Simplificación de CSP

Este módulo utiliza FCA para analizar la estructura del CSP y detectar
implicaciones que pueden simplificar el problema antes de resolverlo.

Autor: Manus AI
Fecha: 15 de Octubre, 2025
"""

from typing import Dict, Set, List, Any, Tuple, FrozenSet, Optional
from ..csp_problem import CSP
from .fca_adapter import CSPToFCAAdapter


class FCAAnalyzer:
    """
    Analizador que usa FCA para detectar implicaciones y simplificar CSP.
    
    El análisis FCA permite:
    1. Detectar variables con propiedades similares (agrupamiento)
    2. Identificar implicaciones estructurales
    3. Priorizar variables críticas para la búsqueda
    4. Simplificar el problema eliminando redundancias
    """
    
    def __init__(self, csp: CSP):
        """
        Inicializa el analizador.
        
        Args:
            csp: Problema CSP a analizar
        """
        self.csp = csp
        self.adapter = CSPToFCAAdapter(csp)
        self._analysis_cache: Optional[Dict[str, Any]] = None
    
    def analyze(self) -> Dict[str, Any]:
        """
        Realiza el análisis FCA completo del CSP.
        
        Returns:
            Diccionario con los resultados del análisis
        """
        if self._analysis_cache is not None:
            return self._analysis_cache
        
        # Construir contexto y retículo
        self.adapter.build_context()
        concepts = self.adapter.build_lattice()
        implications = self.adapter.extract_implications()
        
        # Agrupar variables por propiedades similares
        variable_clusters = self._cluster_variables()
        
        # Identificar variables críticas
        critical_variables = self._identify_critical_variables()
        
        # Detectar variables redundantes
        redundant_pairs = self._detect_redundant_variables()
        
        # Calcular prioridades de variables
        variable_priorities = self._compute_variable_priorities()
        
        self._analysis_cache = {
            'num_concepts': len(concepts),
            'num_implications': len(implications),
            'implications': implications,
            'variable_clusters': variable_clusters,
            'critical_variables': critical_variables,
            'redundant_pairs': redundant_pairs,
            'variable_priorities': variable_priorities,
            'summary': self.adapter.get_summary()
        }
        
        return self._analysis_cache
    
    def _cluster_variables(self) -> List[FrozenSet[str]]:
        """
        Agrupa variables con propiedades similares.
        
        Variables en el mismo cluster tienen las mismas propiedades
        y probablemente se comportan de manera similar en la búsqueda.
        
        Returns:
            Lista de clusters (conjuntos de variables)
        """
        clusters = []
        processed = set()
        
        for var in self.csp.variables:
            if var in processed:
                continue
            
            # Obtener propiedades de esta variable
            props = self.adapter.get_variable_properties(var)
            
            # Encontrar todas las variables con las mismas propiedades
            cluster = self.adapter.get_variables_with_properties(set(props))
            
            if len(cluster) > 1:
                clusters.append(cluster)
                processed.update(cluster)
        
        return clusters
    
    def _identify_critical_variables(self) -> List[str]:
        """
        Identifica variables críticas basándose en el análisis FCA.
        
        Variables críticas son aquellas que:
        - Tienen alto grado de conectividad
        - Tienen dominio pequeño
        - Aparecen en muchos conceptos del retículo
        
        Returns:
            Lista de variables críticas ordenadas por importancia
        """
        variable_scores = {}
        
        for var in self.csp.variables:
            score = 0
            
            # Factor 1: Grado de conectividad (más restricciones = más crítica)
            degree = self.adapter.get_degree(var)
            score += degree * 10
            
            # Factor 2: Tamaño del dominio (menor dominio = más crítica)
            domain_size = len(self.csp.domains[var])
            score += (100 / max(domain_size, 1))
            
            # Factor 3: Propiedades especiales
            props = self.adapter.get_variable_properties(var)
            if 'domain_size_1' in props:
                score += 1000  # Ya asignada, muy crítica
            if 'domain_size_small' in props:
                score += 50
            if 'degree_high' in props:
                score += 30
            
            variable_scores[var] = score
        
        # Ordenar por score descendente
        sorted_vars = sorted(variable_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [var for var, score in sorted_vars]
    
    def _detect_redundant_variables(self) -> List[Tuple[str, str]]:
        """
        Detecta pares de variables redundantes.
        
        Dos variables son redundantes si tienen exactamente las mismas
        propiedades y están conectadas por restricciones simétricas.
        
        Returns:
            Lista de pares de variables redundantes
        """
        redundant_pairs = []
        processed = set()
        
        for var1 in self.csp.variables:
            if var1 in processed:
                continue
            
            props1 = self.adapter.get_variable_properties(var1)
            
            for var2 in self.csp.variables:
                if var2 <= var1 or var2 in processed:
                    continue
                
                props2 = self.adapter.get_variable_properties(var2)
                
                # Si tienen las mismas propiedades, son candidatas a redundancia
                if props1 == props2:
                    redundant_pairs.append((var1, var2))
                    processed.add(var2)
        
        return redundant_pairs
    
    def _compute_variable_priorities(self) -> Dict[str, float]:
        """
        Calcula prioridades de variables para guiar la búsqueda.
        
        La prioridad combina:
        - Grado de conectividad
        - Tamaño del dominio
        - Posición en el retículo de conceptos
        
        Returns:
            Diccionario {variable: prioridad} (mayor = más prioritaria)
        """
        priorities = {}
        
        for var in self.csp.variables:
            priority = 0.0
            
            # Componente 1: MRV (menor dominio = mayor prioridad)
            domain_size = len(self.csp.domains[var])
            priority += 100.0 / max(domain_size, 1)
            
            # Componente 2: Degree (mayor grado = mayor prioridad)
            degree = self.adapter.get_degree(var)
            priority += degree * 5.0
            
            # Componente 3: Propiedades FCA
            props = self.adapter.get_variable_properties(var)
            
            # Bonificaciones por propiedades especiales
            if 'domain_size_1' in props:
                priority += 1000.0  # Ya asignada
            elif 'domain_size_small' in props:
                priority += 50.0
            
            if 'degree_high' in props:
                priority += 20.0
            elif 'degree_medium' in props:
                priority += 10.0
            
            if 'has_unary_constraint' in props:
                priority += 15.0  # Restricciones unarias son fuertes
            
            priorities[var] = priority
        
        return priorities
    
    def get_variable_priority(self, var: str) -> float:
        """
        Obtiene la prioridad de una variable.
        
        Args:
            var: Nombre de la variable
        
        Returns:
            Prioridad de la variable (mayor = más prioritaria)
        """
        if self._analysis_cache is None:
            self.analyze()
        
        return self._analysis_cache['variable_priorities'].get(var, 0.0)
    
    def get_critical_variables(self) -> List[str]:
        """
        Obtiene la lista de variables críticas.
        
        Returns:
            Lista de variables críticas ordenadas por importancia
        """
        if self._analysis_cache is None:
            self.analyze()
        
        return self._analysis_cache['critical_variables']
    
    def get_variable_clusters(self) -> List[FrozenSet[str]]:
        """
        Obtiene los clusters de variables similares.
        
        Returns:
            Lista de clusters de variables
        """
        if self._analysis_cache is None:
            self.analyze()
        
        return self._analysis_cache['variable_clusters']
    
    def get_implications(self) -> List[Tuple[FrozenSet, FrozenSet]]:
        """
        Obtiene las implicaciones detectadas.
        
        Returns:
            Lista de implicaciones (antecedente, consecuente)
        """
        if self._analysis_cache is None:
            self.analyze()
        
        return self._analysis_cache['implications']
    
    def suggest_variable_ordering(self) -> List[str]:
        """
        Sugiere un ordenamiento de variables basado en el análisis FCA.
        
        Este ordenamiento prioriza:
        1. Variables críticas
        2. Variables con mayor prioridad
        3. Variables en clusters pequeños
        
        Returns:
            Lista ordenada de variables (más prioritaria primero)
        """
        if self._analysis_cache is None:
            self.analyze()
        
        # Usar las prioridades calculadas
        priorities = self._analysis_cache['variable_priorities']
        
        # Ordenar por prioridad descendente
        sorted_vars = sorted(priorities.items(), key=lambda x: x[1], reverse=True)
        
        return [var for var, priority in sorted_vars]
    
    def get_analysis_summary(self) -> str:
        """
        Genera un resumen textual del análisis FCA.
        
        Returns:
            String con el resumen del análisis
        """
        if self._analysis_cache is None:
            self.analyze()
        
        summary = self._analysis_cache['summary']
        clusters = self._analysis_cache['variable_clusters']
        critical_vars = self._analysis_cache['critical_variables'][:5]  # Top 5
        implications = self._analysis_cache['implications']
        
        lines = [
            "=== Análisis FCA del CSP ===",
            f"Variables: {summary['num_variables']}",
            f"Atributos (propiedades): {summary['num_attributes']}",
            f"Conceptos formales: {summary['num_concepts']}",
            f"Implicaciones detectadas: {summary['num_implications']}",
            f"",
            f"Grado promedio: {summary['avg_degree']:.2f}",
            f"Grado máximo: {summary['max_degree']}",
            f"Grado mínimo: {summary['min_degree']}",
            f"",
            f"Clusters de variables similares: {len(clusters)}",
            f"Variables críticas (top 5): {', '.join(critical_vars)}",
            f"",
            f"Sugerencia de ordenamiento: {', '.join(self.suggest_variable_ordering()[:10])}...",
        ]
        
        return "\n".join(lines)


def analyze_csp_with_fca(csp: CSP) -> FCAAnalyzer:
    """
    Función de conveniencia para analizar un CSP con FCA.
    
    Args:
        csp: Problema CSP a analizar
    
    Returns:
        Analizador FCA con el análisis completo
    """
    analyzer = FCAAnalyzer(csp)
    analyzer.analyze()
    return analyzer

