from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from collections import OrderedDict

from lattice_weaver.paging.page import Page
from lattice_weaver.paging.serializer import PageSerializer
from pathlib import Path
import os

class CacheLevel(ABC):
    """
    Clase base abstracta para un nivel de caché.
    """
    def __init__(self, name: str, capacity: int):
        self.name = name
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    @abstractmethod
    def get(self, page_id: str) -> Optional[Page]:
        pass

    @abstractmethod
    def put(self, page: Page) -> Optional[Page]:
        pass

    @abstractmethod
    def remove(self, page_id: str) -> Optional[Page]:
        pass

    def get_hit_rate(self) -> float:
        total_accesses = self.hits + self.misses
        return self.hits / total_accesses if total_accesses > 0 else 0.0

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.name} Cache (Capacity: {self.capacity}, Hits: {self.hits}, Misses: {self.misses})>"


class L1Cache(CacheLevel):
    """
    Implementación de la caché de Nivel 1 (L1) en memoria.
    Utiliza una política LRU (Least Recently Used) para el desalojo.
    """
    def __init__(self, capacity: int = 100):
        super().__init__("L1", capacity)
        self._cache: OrderedDict[str, Page] = OrderedDict()

    def get(self, page_id: str) -> Optional[Page]:
        if page_id in self._cache:
            self.hits += 1
            page = self._cache.pop(page_id)  # Mover al final (más recientemente usado)
            self._cache[page_id] = page
            return page
        self.misses += 1
        return None

    def put(self, page: Page) -> Optional[Page]:
        removed_page = None
        if page.id in self._cache:
            self._cache.pop(page.id)
        elif len(self._cache) >= self.capacity:
            # Desalojar el elemento LRU
            removed_page = self._cache.popitem(last=False)[1]
        
        self._cache[page.id] = page
        return removed_page

    def remove(self, page_id: str) -> Optional[Page]:
        return self._cache.pop(page_id) if page_id in self._cache else None

    def __len__(self):
        return len(self._cache)

    def __contains__(self, page_id: str) -> bool:
        return page_id in self._cache


class L2Cache(CacheLevel):
    """
    Implementación de la caché de Nivel 2 (L2) en memoria comprimida.
    Utiliza una política LRU y serializa/deserializa páginas con compresión.
    """
    def __init__(self, capacity: int = 500):
        super().__init__("L2", capacity)
        self._cache: OrderedDict[str, bytes] = OrderedDict() # Almacena bytes comprimidos

    def get(self, page_id: str) -> Optional[Page]:
        if page_id in self._cache:
            self.hits += 1
            compressed_data = self._cache.pop(page_id)
            self._cache[page_id] = compressed_data # Mover al final
            return PageSerializer.deserialize(compressed_data)
        self.misses += 1
        return None

    def put(self, page: Page) -> Optional[Page]:
        removed_page = None
        compressed_data = PageSerializer.serialize(page)

        if page.id in self._cache:
            self._cache.pop(page.id)
        elif len(self._cache) >= self.capacity:
            # Desalojar el elemento LRU
            removed_compressed_data = self._cache.popitem(last=False)[1]
            removed_page = PageSerializer.deserialize(removed_compressed_data)
        
        self._cache[page.id] = compressed_data
        return removed_page

    def remove(self, page_id: str) -> Optional[Page]:
        if page_id in self._cache:
            compressed_data = self._cache.pop(page_id)
            return PageSerializer.deserialize(compressed_data)
        return None

    def __len__(self):
        return len(self._cache)

    def __contains__(self, page_id: str) -> bool:
        return page_id in self._cache


class L3Cache(CacheLevel):
    """
    Implementación de la caché de Nivel 3 (L3) en disco (SSD/HDD).
    Persiste páginas en archivos y utiliza una política LRU.
    """
    def __init__(self, capacity: int = 1000, storage_dir: str = "./page_storage"):
        super().__init__("L3", capacity)
        self.storage_path = Path(storage_dir)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._cache_metadata: OrderedDict[str, Path] = OrderedDict() # page_id -> file_path
        self._load_existing_pages()

    def _load_existing_pages(self):
        # Cargar páginas existentes en el disco, respetando la capacidad y el orden LRU
        # Obtener todos los archivos .page y ordenarlos por tiempo de última modificación (más reciente al final)
        all_files = sorted(self.storage_path.glob("*.page"), key=os.path.getmtime)
        
        # Cargar solo las páginas que caben en la caché (las más recientes)
        files_to_load = all_files[-self.capacity:]

        for file_path in files_to_load:
            page_id = file_path.stem
            self._cache_metadata[page_id] = file_path # Añadir al final para mantener el orden LRU
        
        # Eliminar del disco las páginas más antiguas que no se cargaron
        for file_path in all_files[:-self.capacity]:
            try:
                file_path.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Error al eliminar archivo antiguo {file_path.name} de L3 durante la carga inicial: {e}")


    def get(self, page_id: str) -> Optional[Page]:
        if page_id in self._cache_metadata:
            self.hits += 1
            file_path = self._cache_metadata.pop(page_id)
            self._cache_metadata[page_id] = file_path # Mover al final
            try:
                with open(file_path, "rb") as f:
                    compressed_data = f.read()
                return PageSerializer.deserialize(compressed_data)
            except FileNotFoundError:
                self.misses += 1
                return None
        self.misses += 1
        return None

    def put(self, page: Page) -> Optional[Page]:
        removed_page = None
        file_path = self.storage_path / f"{page.id}.page"
        compressed_data = PageSerializer.serialize(page)

        if page.id in self._cache_metadata:
            self._cache_metadata.pop(page.id)
        elif len(self._cache_metadata) >= self.capacity:
            # Desalojar el elemento LRU
            oldest_id, oldest_file_path = self._cache_metadata.popitem(last=False)
            try:
                # Necesitamos deserializar la página desalojada para devolverla ANTES de eliminar el archivo
                with open(oldest_file_path, "rb") as f:
                    removed_compressed_data = f.read()
                removed_page = PageSerializer.deserialize(removed_compressed_data)
                oldest_file_path.unlink() # Eliminar archivo del disco
            except FileNotFoundError:
                pass # Ya fue eliminado o nunca existió
            except Exception as e:
                print(f"Error al desalojar página {oldest_id} de L3: {e}")
        
        with open(file_path, "wb") as f:
            f.write(compressed_data)
        # Asegurarse de que la página se añade al final (más recientemente usada)
        self._cache_metadata[page.id] = file_path
        # Si la caché excede la capacidad, desalojar el elemento LRU
        while len(self._cache_metadata) > self.capacity:
            oldest_id, oldest_file_path = self._cache_metadata.popitem(last=False)
            try:
                oldest_file_path.unlink() # Eliminar archivo del disco
            except FileNotFoundError:
                pass # Ya fue eliminado o nunca existió
            except Exception as e:
                print(f"Error al desalojar página {oldest_id} de L3 durante put: {e}")
        return removed_page

    def remove(self, page_id: str) -> Optional[Page]:
        if page_id in self._cache_metadata:
            file_path = self._cache_metadata.pop(page_id)
            try:
                with open(file_path, "rb") as f:
                    compressed_data = f.read()
                file_path.unlink() # Eliminar archivo del disco
                return PageSerializer.deserialize(compressed_data)
            except FileNotFoundError:
                return None # La página ya no existe en disco, pero estaba en metadatos
            except Exception as e:
                print(f"Error al remover página {page_id} de L3: {e}")
                return None
        return None

    def __len__(self):
        return len(self._cache_metadata)

    def __contains__(self, page_id: str) -> bool:
        return page_id in self._cache_metadata

