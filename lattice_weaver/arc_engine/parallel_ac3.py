"""
AC-3 Paralelo

Este módulo implementa una versión paralelizada del algoritmo AC-3 que divide
los arcos en chunks y los procesa en paralelo usando ThreadPoolExecutor.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing as mp
from typing import List, Tuple, Set
import logging

logger = logging.getLogger(__name__)


class ParallelAC3:
    """
    AC-3 paralelizado para aprovechar múltiples cores.
    
    Esta clase divide la cola de arcos en chunks y los procesa en paralelo,
    usando locks para acceso seguro a los dominios compartidos.
    
    Attributes:
        arc_engine: Referencia al ArcEngine
        num_workers: Número de workers paralelos
        domain_lock: Lock para acceso concurrente a dominios
    """
    
    def __init__(self, arc_engine, num_workers: int = None):
        """
        Inicializa el motor AC-3 paralelo.
        
        Args:
            arc_engine: Instancia de ArcEngine
            num_workers: Número de workers (default: min(4, CPU count))
        """
        self.arc_engine = arc_engine
        self.num_workers = num_workers or min(4, mp.cpu_count())
        self.domain_lock = Lock()  # Para acceso concurrente a dominios
        
        logger.info(f"ParallelAC3 inicializado con {self.num_workers} workers")
    
    def enforce_arc_consistency_parallel(self) -> bool:
        """
        Ejecuta AC-3 en paralelo.
        
        Returns:
            False si se encuentra una inconsistencia, True si es consistente
        
        Complejidad: O(ed³/p) donde p = número de workers
        """
        # 1. Inicializar cola de arcos
        queue = self._initialize_queue()
        
        if len(queue) == 0:
            return True
        
        logger.info(f"Procesando {len(queue)} arcos con {self.num_workers} workers")
        
        # 2. Procesar en paralelo
        iteration = 0
        while queue:
            iteration += 1
            
            # Dividir en chunks
            chunks = self._partition_arcs(queue, self.num_workers)
            
            # Procesar chunks en paralelo
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for chunk in chunks:
                    future = executor.submit(self._process_chunk, chunk)
                    futures.append(future)
                
                # Recolectar resultados
                all_consistent = True
                new_arcs = []
                
                for future in as_completed(futures):
                    consistent, arcs = future.result()
                    if not consistent:
                        all_consistent = False
                        # Cancelar workers restantes
                        for f in futures:
                            f.cancel()
                        return False
                    new_arcs.extend(arcs)
            
            # Actualizar cola con nuevos arcos
            queue = list(set(new_arcs))  # Eliminar duplicados
            
            logger.debug(f"Iteración {iteration}: {len(new_arcs)} arcos añadidos")
        
        logger.info(f"AC-3 paralelo completado en {iteration} iteraciones")
        return True
    
    def _process_chunk(self, arcs: List[Tuple[str, str, str]]) -> Tuple[bool, List]:
        """
        Procesa un chunk de arcos.
        
        Args:
            arcs: Lista de tuplas (xi, xj, constraint_id)
        
        Returns:
            Tupla (consistente, nuevos_arcos)
        """
        new_arcs = []
        
        for (xi, xj, constraint_id) in arcs:
            # Revise con lock para acceso seguro a dominios
            with self.domain_lock:
                # Verificar que el dominio no esté vacío antes de revisar
                if not self.arc_engine.variables[xi]:
                    return False, []
                
                # Ejecutar revise (del módulo ac31)
                from .ac31 import revise_with_last_support
                revised, removed_values = revise_with_last_support(
                    self.arc_engine, xi, xj, constraint_id
                )
            
            if revised:
                # Verificar inconsistencia
                with self.domain_lock:
                    if not self.arc_engine.variables[xi]:
                        return False, []  # Inconsistencia encontrada
                
                # Añadir arcos afectados
                neighbors = self._get_neighbors(xi)
                for xk in neighbors:
                    if xk != xj:
                        # Encontrar el constraint_id para (xk, xi)
                        c_id = self._find_constraint_id(xk, xi)
                        if c_id:
                            new_arcs.append((xk, xi, c_id))
        
        return True, new_arcs
    
    def _get_neighbors(self, var: str) -> List[str]:
        """
        Obtiene los vecinos de una variable en el grafo de restricciones.
        
        Args:
            var: Nombre de la variable
        
        Returns:
            Lista de variables vecinas
        """
        return list(self.arc_engine.graph.neighbors(var))
    
    def _find_constraint_id(self, var1: str, var2: str) -> str:
        """
        Encuentra el ID de la restricción entre dos variables.
        
        Args:
            var1, var2: Nombres de las variables
        
        Returns:
            ID de la restricción o None si no existe
        """
        if self.arc_engine.graph.has_edge(var1, var2):
            edge_data = self.arc_engine.graph.get_edge_data(var1, var2)
            return edge_data.get('cid')
        return None
    
    def _partition_arcs(self, arcs: List, num_chunks: int) -> List[List]:
        """
        Divide arcos en chunks balanceados.
        
        Args:
            arcs: Lista de arcos
            num_chunks: Número de chunks
        
        Returns:
            Lista de chunks
        """
        if len(arcs) < num_chunks:
            # Si hay menos arcos que chunks, un arco por chunk
            return [[arc] for arc in arcs]
        
        chunk_size = len(arcs) // num_chunks
        chunks = []
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < num_chunks - 1 else len(arcs)
            chunks.append(arcs[start:end])
        
        return chunks
    
    def _initialize_queue(self) -> List[Tuple[str, str, str]]:
        """
        Inicializa la cola con todos los arcos.
        
        Returns:
            Lista de tuplas (xi, xj, constraint_id)
        """
        queue = []
        for cid, constraint in self.arc_engine.constraints.items():
            queue.append((constraint.var1, constraint.var2, cid))
            queue.append((constraint.var2, constraint.var1, cid))
        return queue
    
    def get_stats(self) -> dict:
        """
        Retorna estadísticas del procesamiento paralelo.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            'num_workers': self.num_workers,
            'total_constraints': len(self.arc_engine.constraints),
            'total_arcs': len(self.arc_engine.constraints) * 2,
            'total_variables': len(self.arc_engine.variables)
        }
    
    def set_num_workers(self, num_workers: int):
        """
        Cambia el número de workers.
        
        Args:
            num_workers: Nuevo número de workers
        """
        if num_workers < 1:
            raise ValueError("El número de workers debe ser al menos 1")
        self.num_workers = num_workers
        logger.info(f"Número de workers actualizado a {self.num_workers}")
    
    def __repr__(self) -> str:
        return f"ParallelAC3(workers={self.num_workers}, constraints={len(self.arc_engine.constraints)})"

