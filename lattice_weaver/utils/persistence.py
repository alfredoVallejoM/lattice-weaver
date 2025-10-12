"""
PersistenceManager: I/O incremental y checkpointing para LatticeWeaver.

Este módulo proporciona persistencia eficiente de grafos de estados usando
formato JSONL (JSON Lines) con compresión gzip opcional. Permite escritura
incremental sin bloquear el análisis y carga desde checkpoints.

Autor: Manus AI
Fecha: 11 de Octubre de 2025
"""

import json
import gzip
from pathlib import Path
from typing import Any, Dict, Optional, TextIO
from .state_manager import CanonicalState


class PersistenceManager:
    """
    Gestor de persistencia incremental para grafos de estados.
    
    Usa formato JSONL (una línea JSON por registro) con compresión gzip
    opcional. Esto permite:
    - Escritura incremental sin cargar todo en memoria
    - Lectura selectiva de registros
    - Compresión eficiente (70-80% reducción)
    
    Attributes:
        filepath: Ruta del archivo de persistencia
        compress: Si se debe usar compresión gzip
        file_handle: Handle del archivo abierto
        records_written: Contador de registros escritos
    """
    
    def __init__(self, filepath: str, compress: bool = True):
        """
        Inicializa el gestor de persistencia.
        
        Args:
            filepath: Ruta del archivo (se añade .gz si compress=True)
            compress: Si se debe usar compresión gzip
        """
        self.filepath = Path(filepath)
        self.compress = compress
        
        if self.compress and not self.filepath.suffix == '.gz':
            self.filepath = self.filepath.with_suffix(self.filepath.suffix + '.gz')
        
        self.file_handle: Optional[TextIO] = None
        self.records_written = 0
        self._is_open = False
    
    def open_for_writing(self):
        """Abre el archivo para escritura incremental."""
        if self._is_open:
            return
        
        # Crear directorio si no existe
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Abrir archivo
        if self.compress:
            self.file_handle = gzip.open(self.filepath, 'wt', encoding='utf-8')
        else:
            self.file_handle = open(self.filepath, 'w', encoding='utf-8')
        
        self._is_open = True
        self.records_written = 0
        
        # Escribir metadatos de inicio
        self._write_record({
            'type': 'metadata',
            'version': '4.0.0',
            'format': 'jsonl',
            'compressed': self.compress
        })
    
    def _write_record(self, record: dict):
        """
        Escribe un registro JSON en el archivo.
        
        Args:
            record: Diccionario a serializar
        """
        if not self._is_open:
            raise RuntimeError("PersistenceManager no está abierto. Llama a open_for_writing() primero.")
        
        json_line = json.dumps(record, ensure_ascii=False)
        self.file_handle.write(json_line + '\n')
        self.records_written += 1
    
    def write_node(self, node_id: int, state: CanonicalState, level: int):
        """
        Escribe un nodo del grafo de estados.
        
        Args:
            node_id: ID del nodo
            state: Estado canónico del nodo
            level: Nivel en el grafo
        """
        self._write_record({
            'type': 'node',
            'id': node_id,
            'state': state.to_dict(),
            'level': level
        })
    
    def write_edge(self, source_id: int, target_id: int, constraint: tuple):
        """
        Escribe una arista del grafo de estados.
        
        Args:
            source_id: ID del nodo origen
            target_id: ID del nodo destino
            constraint: Restricción que genera la arista
        """
        self._write_record({
            'type': 'edge',
            'source': source_id,
            'target': target_id,
            'constraint': list(constraint)
        })
    
    def write_homotopy(self, homotopy_id: int, square: list):
        """
        Escribe una homotopía detectada.
        
        Args:
            homotopy_id: ID de la homotopía
            square: Lista de 4 nodos que forman el cuadrado conmutativo
        """
        self._write_record({
            'type': 'homotopy',
            'id': homotopy_id,
            'square': square
        })
    
    def write_statistics(self, stats: dict):
        """
        Escribe estadísticas del análisis.
        
        Args:
            stats: Diccionario con estadísticas
        """
        self._write_record({
            'type': 'statistics',
            'data': stats
        })
    
    def close(self):
        """Cierra el archivo de persistencia."""
        if not self._is_open:
            return
        
        # Escribir metadatos de cierre
        self._write_record({
            'type': 'metadata',
            'event': 'close',
            'records_written': self.records_written
        })
        
        self.file_handle.close()
        self._is_open = False
    
    def __enter__(self):
        """Soporte para context manager."""
        self.open_for_writing()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Soporte para context manager."""
        self.close()
    
    @staticmethod
    def load_from_checkpoint(filepath: str) -> dict:
        """
        Carga un grafo de estados desde un archivo JSONL.
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            Diccionario con nodos, aristas y homotopías
        """
        filepath = Path(filepath)
        
        # Detectar si está comprimido
        is_compressed = filepath.suffix == '.gz'
        
        # Abrir archivo
        if is_compressed:
            file_handle = gzip.open(filepath, 'rt', encoding='utf-8')
        else:
            file_handle = open(filepath, 'r', encoding='utf-8')
        
        # Leer registros
        nodes = {}
        edges = []
        homotopies = []
        statistics = {}
        
        try:
            for line in file_handle:
                record = json.loads(line.strip())
                
                record_type = record.get('type')
                
                if record_type == 'node':
                    nodes[record['id']] = {
                        'state': record['state'],
                        'level': record['level']
                    }
                
                elif record_type == 'edge':
                    edges.append({
                        'source': record['source'],
                        'target': record['target'],
                        'constraint': tuple(record['constraint'])
                    })
                
                elif record_type == 'homotopy':
                    homotopies.append({
                        'id': record['id'],
                        'square': record['square']
                    })
                
                elif record_type == 'statistics':
                    statistics = record['data']
        
        finally:
            file_handle.close()
        
        return {
            'nodes': nodes,
            'edges': edges,
            'homotopies': homotopies,
            'statistics': statistics
        }
    
    @staticmethod
    def estimate_file_size(num_nodes: int, num_edges: int, num_homotopies: int,
                          compressed: bool = True) -> float:
        """
        Estima el tamaño del archivo de persistencia.
        
        Args:
            num_nodes: Número de nodos
            num_edges: Número de aristas
            num_homotopies: Número de homotopías
            compressed: Si se usará compresión
            
        Returns:
            Tamaño estimado en MB
        """
        # Estimaciones de tamaño por registro (en bytes)
        bytes_per_node = 200  # Promedio
        bytes_per_edge = 100
        bytes_per_homotopy = 80
        
        total_bytes = (
            num_nodes * bytes_per_node +
            num_edges * bytes_per_edge +
            num_homotopies * bytes_per_homotopy
        )
        
        # Compresión reduce ~75%
        if compressed:
            total_bytes *= 0.25
        
        return total_bytes / (1024 * 1024)


class CheckpointManager:
    """
    Gestor de checkpoints para análisis largos.
    
    Permite guardar y restaurar el estado completo del análisis,
    útil para análisis que toman mucho tiempo o para debugging.
    """
    
    @staticmethod
    def save_checkpoint(filepath: str, state_manager, homotopy_analyzer):
        """
        Guarda un checkpoint completo del análisis.
        
        Args:
            filepath: Ruta del archivo de checkpoint
            state_manager: StateManager a guardar
            homotopy_analyzer: HomotopyAnalyzer a guardar
        """
        checkpoint_data = {
            'version': '4.0.0',
            'state_manager': state_manager.export_states(),
            'homotopy_analyzer': {
                'graph': {
                    'nodes': list(homotopy_analyzer.state_graph.nodes),
                    'edges': [(u, v, data) for u, v, data in homotopy_analyzer.state_graph.edges(data=True)]
                },
                'statistics': homotopy_analyzer.get_statistics()
            }
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    @staticmethod
    def load_checkpoint(filepath: str) -> dict:
        """
        Carga un checkpoint completo.
        
        Args:
            filepath: Ruta del archivo de checkpoint
            
        Returns:
            Diccionario con los datos del checkpoint
        """
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            return json.load(f)

