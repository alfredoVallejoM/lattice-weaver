"""
Optimizaciones de Rendimiento para AC-3

Implementa diversas optimizaciones para mejorar el rendimiento del motor CSP.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict
import time
import logging

logger = logging.getLogger(__name__)


class ArcRevisionCache:
    """
    Caché de revisiones de arcos.
    
    Almacena resultados de revisiones previas para evitar recomputación.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Inicializa el caché.
        
        Args:
            max_size: Tamaño máximo del caché
        """
        self.cache: Dict[Tuple, Tuple[bool, List]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, xi: str, xj: str, cid: str, 
            domain_xi_hash: int, domain_xj_hash: int) -> Optional[Tuple[bool, List]]:
        """
        Obtiene resultado de caché.
        
        Args:
            xi: Variable a revisar
            xj: Variable de soporte
            cid: ID de restricción
            domain_xi_hash: Hash del dominio de xi
            domain_xj_hash: Hash del dominio de xj
        
        Returns:
            (revised, removed_values) o None si no está en caché
        """
        key = (xi, xj, cid, domain_xi_hash, domain_xj_hash)
        
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, xi: str, xj: str, cid: str,
            domain_xi_hash: int, domain_xj_hash: int,
            revised: bool, removed_values: List):
        """
        Almacena resultado en caché.
        
        Args:
            xi: Variable revisada
            xj: Variable de soporte
            cid: ID de restricción
            domain_xi_hash: Hash del dominio de xi
            domain_xj_hash: Hash del dominio de xj
            revised: Si se revisó
            removed_values: Valores eliminados
        """
        if len(self.cache) >= self.max_size:
            # Eliminar entrada aleatoria (LRU sería mejor)
            self.cache.pop(next(iter(self.cache)))
        
        key = (xi, xj, cid, domain_xi_hash, domain_xj_hash)
        self.cache[key] = (revised, removed_values.copy())
    
    def clear(self):
        """Limpia el caché."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_hit_rate(self) -> float:
        """
        Calcula tasa de aciertos.
        
        Returns:
            Hit rate (0.0 - 1.0)
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del caché.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate()
        }


class ArcOrderingStrategy:
    """
    Estrategia de ordenamiento de arcos.
    
    Ordena arcos para minimizar iteraciones de AC-3.
    """
    
    @staticmethod
    def order_by_domain_size(arcs: List[Tuple[str, str, str]], 
                            engine) -> List[Tuple[str, str, str]]:
        """
        Ordena arcos por tamaño de dominio (menor primero).
        
        Heurística: Revisar primero variables con dominios pequeños
        puede detectar inconsistencias más rápido.
        
        Args:
            arcs: Lista de arcos (xi, xj, cid)
            engine: ArcEngine
        
        Returns:
            Arcos ordenados
        """
        def domain_size(arc):
            xi, xj, cid = arc
            return len(engine.variables[xi])
        
        return sorted(arcs, key=domain_size)
    
    @staticmethod
    def order_by_constraint_tightness(arcs: List[Tuple[str, str, str]],
                                     engine) -> List[Tuple[str, str, str]]:
        """
        Ordena arcos por "tightness" de restricción.
        
        Heurística: Restricciones más restrictivas primero.
        
        Args:
            arcs: Lista de arcos
            engine: ArcEngine
        
        Returns:
            Arcos ordenados
        """
        # Simplificación: usar grado de la variable
        def degree(arc):
            xi, xj, cid = arc
            return engine.graph.degree(xi)
        
        return sorted(arcs, key=degree, reverse=True)
    
    @staticmethod
    def order_by_last_revision(arcs: List[Tuple[str, str, str]],
                               revision_history: Dict[Tuple, int]) -> List[Tuple[str, str, str]]:
        """
        Ordena arcos por última revisión (más reciente primero).
        
        Heurística: Arcos revisados recientemente tienen más probabilidad
        de necesitar revisión nuevamente.
        
        Args:
            arcs: Lista de arcos
            revision_history: {(xi, xj, cid): timestamp}
        
        Returns:
            Arcos ordenados
        """
        def last_revision(arc):
            return revision_history.get(arc, 0)
        
        return sorted(arcs, key=last_revision, reverse=True)


class RedundantArcDetector:
    """
    Detector de arcos redundantes.
    
    Identifica arcos que no necesitan revisión.
    """
    
    @staticmethod
    def is_redundant(xi: str, xj: str, cid: str, engine) -> bool:
        """
        Verifica si un arco es redundante.
        
        Un arco es redundante si:
        - El dominio de xi es singleton
        - El dominio de xj no ha cambiado desde última revisión
        
        Args:
            xi: Variable a revisar
            xj: Variable de soporte
            cid: ID de restricción
            engine: ArcEngine
        
        Returns:
            True si es redundante
        """
        # Singleton: no puede reducirse más
        if len(engine.variables[xi]) == 1:
            return True
        
        # Dominio vacío de xj: inconsistencia ya detectada
        if len(engine.variables[xj]) == 0:
            return True
        
        return False
    
    @staticmethod
    def filter_redundant_arcs(arcs: List[Tuple[str, str, str]], 
                             engine) -> List[Tuple[str, str, str]]:
        """
        Filtra arcos redundantes.
        
        Args:
            arcs: Lista de arcos
            engine: ArcEngine
        
        Returns:
            Arcos no redundantes
        """
        filtered = []
        
        for xi, xj, cid in arcs:
            if not RedundantArcDetector.is_redundant(xi, xj, cid, engine):
                filtered.append((xi, xj, cid))
        
        return filtered


class PerformanceMonitor:
    """
    Monitor de rendimiento de AC-3.
    
    Rastrea métricas de rendimiento.
    """
    
    def __init__(self):
        """Inicializa el monitor."""
        self.start_time = None
        self.end_time = None
        self.iterations = 0
        self.revisions = 0
        self.successful_revisions = 0
        self.domain_reductions = 0
        self.arc_evaluations = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def start(self):
        """Inicia el monitoreo."""
        self.start_time = time.time()
    
    def end(self):
        """Finaliza el monitoreo."""
        self.end_time = time.time()
    
    def record_iteration(self):
        """Registra una iteración."""
        self.iterations += 1
    
    def record_revision(self, revised: bool, removed_count: int = 0):
        """
        Registra una revisión.
        
        Args:
            revised: Si se revisó
            removed_count: Cantidad de valores eliminados
        """
        self.revisions += 1
        if revised:
            self.successful_revisions += 1
            self.domain_reductions += removed_count
    
    def record_arc_evaluation(self):
        """Registra una evaluación de arco."""
        self.arc_evaluations += 1
    
    def record_cache_hit(self):
        """Registra un hit de caché."""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Registra un miss de caché."""
        self.cache_misses += 1
    
    def get_elapsed_time(self) -> float:
        """
        Obtiene tiempo transcurrido.
        
        Returns:
            Tiempo en segundos
        """
        if self.start_time is None:
            return 0.0
        
        end = self.end_time or time.time()
        return end - self.start_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de rendimiento.
        
        Returns:
            Diccionario con estadísticas
        """
        elapsed = self.get_elapsed_time()
        
        return {
            'elapsed_time': elapsed,
            'iterations': self.iterations,
            'revisions': self.revisions,
            'successful_revisions': self.successful_revisions,
            'domain_reductions': self.domain_reductions,
            'arc_evaluations': self.arc_evaluations,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0.0
            ),
            'revisions_per_second': (
                self.revisions / elapsed if elapsed > 0 else 0
            ),
            'avg_reductions_per_revision': (
                self.domain_reductions / self.successful_revisions
                if self.successful_revisions > 0 else 0
            )
        }
    
    def print_statistics(self):
        """Imprime estadísticas."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("ESTADÍSTICAS DE RENDIMIENTO")
        print("=" * 60)
        print(f"Tiempo transcurrido: {stats['elapsed_time']:.4f}s")
        print(f"Iteraciones: {stats['iterations']}")
        print(f"Revisiones: {stats['revisions']}")
        print(f"Revisiones exitosas: {stats['successful_revisions']}")
        print(f"Reducciones de dominio: {stats['domain_reductions']}")
        print(f"Evaluaciones de arco: {stats['arc_evaluations']}")
        print(f"Cache hits: {stats['cache_hits']}")
        print(f"Cache misses: {stats['cache_misses']}")
        print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
        print(f"Revisiones/segundo: {stats['revisions_per_second']:.2f}")
        print(f"Reducciones promedio/revisión: {stats['avg_reductions_per_revision']:.2f}")
        print("=" * 60)


class OptimizedAC3:
    """
    AC-3 optimizado con todas las optimizaciones.
    
    Combina caché, ordenamiento, detección de redundancia y monitoreo.
    """
    
    def __init__(self, arc_engine, 
                 use_cache: bool = True,
                 use_ordering: bool = True,
                 use_redundancy_filter: bool = True,
                 use_monitoring: bool = True):
        """
        Inicializa AC-3 optimizado.
        
        Args:
            arc_engine: ArcEngine
            use_cache: Habilitar caché
            use_ordering: Habilitar ordenamiento
            use_redundancy_filter: Habilitar filtrado de redundancia
            use_monitoring: Habilitar monitoreo
        """
        self.engine = arc_engine
        self.use_cache = use_cache
        self.use_ordering = use_ordering
        self.use_redundancy_filter = use_redundancy_filter
        self.use_monitoring = use_monitoring
        
        self.cache = ArcRevisionCache() if use_cache else None
        self.monitor = PerformanceMonitor() if use_monitoring else None
        self.revision_history: Dict[Tuple, int] = {}
    
    def enforce_arc_consistency_optimized(self) -> bool:
        """
        Ejecuta AC-3 con optimizaciones.
        
        Returns:
            True si es consistente
        """
        if self.monitor:
            self.monitor.start()
        
        # Inicializar cola
        queue = []
        for cid, c in self.engine.constraints.items():
            queue.append((c.var1, c.var2, cid))
            queue.append((c.var2, c.var1, cid))
        
        iteration = 0
        
        while queue:
            iteration += 1
            
            if self.monitor:
                self.monitor.record_iteration()
            
            # Filtrar redundantes
            if self.use_redundancy_filter:
                queue = RedundantArcDetector.filter_redundant_arcs(queue, self.engine)
            
            if not queue:
                break
            
            # Ordenar
            if self.use_ordering:
                queue = ArcOrderingStrategy.order_by_domain_size(queue, self.engine)
            
            # Procesar primer arco
            xi, xj, cid = queue.pop(0)
            
            # Intentar usar caché
            revised = False
            removed_values = []
            
            if self.use_cache and self.cache:
                domain_xi_hash = hash(frozenset(self.engine.variables[xi].get_values()))
                domain_xj_hash = hash(frozenset(self.engine.variables[xj].get_values()))
                
                cached = self.cache.get(xi, xj, cid, domain_xi_hash, domain_xj_hash)
                
                if cached is not None:
                    revised, removed_values = cached
                    if self.monitor:
                        self.monitor.record_cache_hit()
                else:
                    if self.monitor:
                        self.monitor.record_cache_miss()
            
            # Si no hay caché, revisar
            if not self.use_cache or self.cache is None or cached is None:
                from .ac31 import revise_with_last_support
                revised, removed_values = revise_with_last_support(self.engine, xi, xj, cid)
                
                # Guardar en caché
                if self.use_cache and self.cache:
                    self.cache.put(xi, xj, cid, domain_xi_hash, domain_xj_hash,
                                  revised, removed_values)
            
            # Registrar revisión
            if self.monitor:
                self.monitor.record_revision(revised, len(removed_values))
            
            # Actualizar historial
            self.revision_history[(xi, xj, cid)] = iteration
            
            if revised:
                if not self.engine.variables[xi]:
                    if self.monitor:
                        self.monitor.end()
                    return False
                
                # Agregar arcos afectados
                for neighbor in self.engine.graph.neighbors(xi):
                    if neighbor != xj:
                        c_id = self.engine.graph.get_edge_data(neighbor, xi)['cid']
                        queue.append((neighbor, xi, c_id))
        
        if self.monitor:
            self.monitor.end()
        
        logger.info(f"AC-3 optimizado completado en {iteration} iteraciones")
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas combinadas.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {}
        
        if self.monitor:
            stats['performance'] = self.monitor.get_statistics()
        
        if self.cache:
            stats['cache'] = self.cache.get_statistics()
        
        return stats
    
    def print_statistics(self):
        """Imprime todas las estadísticas."""
        if self.monitor:
            self.monitor.print_statistics()
        
        if self.cache:
            cache_stats = self.cache.get_statistics()
            print("\nESTADÍSTICAS DE CACHÉ")
            print("=" * 60)
            print(f"Tamaño: {cache_stats['size']}/{cache_stats['max_size']}")
            print(f"Hits: {cache_stats['hits']}")
            print(f"Misses: {cache_stats['misses']}")
            print(f"Hit rate: {cache_stats['hit_rate']:.2%}")
            print("=" * 60)


def create_optimized_ac3(arc_engine, **kwargs) -> OptimizedAC3:
    """
    Crea una instancia de AC-3 optimizado.
    
    Args:
        arc_engine: ArcEngine
        **kwargs: Opciones de optimización
    
    Returns:
        Instancia de OptimizedAC3
    """
    return OptimizedAC3(arc_engine, **kwargs)

