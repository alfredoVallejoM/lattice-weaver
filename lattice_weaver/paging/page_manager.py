
"""
Módulo para la gestión del sistema de paginación multinivel de LatticeWeaver.

Este módulo define la clase `PageManager`, que orquesta la interacción entre
los diferentes niveles de caché (L1, L2, L3) para gestionar eficientemente
las páginas de datos, optimizando el acceso a la memoria y la persistencia.

Autor: LatticeWeaver Development Team
Fecha: 14 de Octubre de 2025
"""

from typing import Optional, List, Dict, Any

from lattice_weaver.paging.page import Page
from lattice_weaver.paging.cache_levels import L1Cache, L2Cache, L3Cache, CacheLevel

class PageManager:
    """
    Orquesta el sistema de paginación multinivel.

    Gestiona las solicitudes de páginas, la asignación de memoria, el prefetching
    y las políticas de desalojo a través de una jerarquía de caché (L1, L2, L3).
    Su objetivo es minimizar la latencia de acceso a los datos y optimizar el uso
    de los recursos de memoria.

    Attributes:
        l1_cache (L1Cache): Caché de primer nivel (más rápida, menor capacidad).
        l2_cache (L2Cache): Caché de segundo nivel (intermedia en velocidad y capacidad).
        l3_cache (L3Cache): Caché de tercer nivel (más lenta, mayor capacidad, persistente).
        cache_hierarchy (List[CacheLevel]): Lista ordenada de los niveles de caché.
    """
    def __init__(self, l1_capacity: int = 100, l2_capacity: int = 500, l3_capacity: int = 1000, l3_storage_dir: str = "./page_storage"):
        self.l1_cache = L1Cache(capacity=l1_capacity)
        self.l2_cache = L2Cache(capacity=l2_capacity)
        self.l3_cache = L3Cache(capacity=l3_capacity, storage_dir=l3_storage_dir)
        self.cache_hierarchy: List[CacheLevel] = [self.l1_cache, self.l2_cache, self.l3_cache]

    def get_page(self, page_id: str) -> Optional[Page]:
        """
        Intenta recuperar una página de la jerarquía de caché.
        Si la encuentra en un nivel inferior, la promueve a los niveles superiores
        para optimizar futuros accesos (política de promoción de caché).
        """
        page = None
        found_at_level = -1

        # 1. Buscar en la jerarquía de caché
        for i, cache in enumerate(self.cache_hierarchy):
            page = cache.get(page_id)
            if page:
                found_at_level = i
                break
        
        if not page:
            return None

        # 2. Promoción de Página (si se encontró en un nivel inferior)
        if found_at_level > 0:
            # La página a promover es la que se encontró
            page_to_promote = page

            # Remover la página del nivel donde se encontró para moverla hacia arriba
            self.cache_hierarchy[found_at_level].remove(page_id)

            # Promover la página encontrada a L1 y manejar los desalojos en cascada
            # La forma más sencilla de hacer esto es removerla del nivel inferior
            # y luego usar put_page, que ya maneja la propagación hacia abajo.
            self.put_page(page_to_promote)

        return page

    def put_page(self, page: Page) -> None:
        """
        Coloca una página en la caché L1. Si la L1 está llena, la página desalojada
        se intenta colocar en la L2, y así sucesivamente hasta la L3 (política de desalojo en cascada).
        Asegura que no haya copias de la página en niveles inferiores antes de colocarla en L1 para mantener la coherencia.
        """
        # Asegurarse de que no haya copias de la página en niveles inferiores antes de colocarla en L1
        self.l2_cache.remove(page.id)
        self.l3_cache.remove(page.id)

        evicted_from_l1 = self.l1_cache.put(page)
        if evicted_from_l1:
            evicted_from_l2 = self.l2_cache.put(evicted_from_l1)
            if evicted_from_l2:
                self.l3_cache.put(evicted_from_l2)

    def remove_page(self, page_id: str) -> None:
        """
        Elimina una página de todos los niveles de caché.
        """
        for cache in self.cache_hierarchy:
            cache.remove(page_id)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Devuelve estadísticas de rendimiento (hits, misses, hit_rate, tamaño, capacidad)
        para cada nivel de la jerarquía de caché.
        """
        stats = {}
        for cache in self.cache_hierarchy:
            stats[cache.name] = {
                "hits": cache.hits,
                "misses": cache.misses,
                "hit_rate": cache.get_hit_rate(),
                "size": len(cache),
                "capacity": cache.capacity
            }
        return stats

    def __repr__(self):
        return f"PageManager(L1: {len(self.l1_cache)}/{self.l1_cache.capacity}, L2: {len(self.l2_cache)}/{self.l2_cache.capacity}, L3: {len(self.l3_cache)}/{self.l3_cache.capacity})"

