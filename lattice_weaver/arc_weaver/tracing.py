"""
SearchSpaceTracer - Captura de Evolución del Espacio de Búsqueda.

Este módulo implementa un sistema de trazado de bajo overhead para capturar
la evolución del espacio de búsqueda durante la resolución de CSPs.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
Versión: 1.0
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Set, List, Literal
from collections import deque
import time
import csv
import json
from pathlib import Path
import threading
import queue


@dataclass(frozen=True)
class SearchEvent:
    """
    Representa un evento atómico en el proceso de búsqueda.
    
    Attributes:
        timestamp: Tiempo del evento (segundos desde epoch)
        event_type: Tipo de evento
        variable: Variable involucrada (si aplica)
        value: Valor asignado/eliminado (si aplica)
        source_variable: Variable que causó la poda (para domain_pruned)
        pruned_values: Conjunto de valores eliminados (para domain_pruned)
        depth: Profundidad en el árbol de búsqueda
        metadata: Información adicional específica del evento
    
    Examples:
        >>> event = SearchEvent(
        ...     timestamp=time.time(),
        ...     event_type='variable_assigned',
        ...     variable='x1',
        ...     value=3,
        ...     depth=1
        ... )
    """
    timestamp: float
    event_type: Literal[
        'search_started',
        'search_ended',
        'variable_assigned',
        'backtrack',
        'domain_pruned',
        'solution_found',
        'ac3_call',
        'cluster_operation'
    ]
    variable: Optional[str] = None
    value: Optional[Any] = None
    source_variable: Optional[str] = None
    pruned_values: Optional[Set[Any]] = None
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el evento a diccionario para serialización."""
        d = asdict(self)
        # Convertir set a list para JSON
        if d['pruned_values'] is not None:
            d['pruned_values'] = list(d['pruned_values'])
        return d


class SearchSpaceTracer:
    """
    Sistema de trazado de bajo overhead para capturar eventos de búsqueda.
    
    Soporta dos modos de operación:
    - Síncrono: Los eventos se almacenan en memoria (adecuado para debugging)
    - Asíncrono: Los eventos se escriben a disco en un thread separado (bajo overhead)
    
    Attributes:
        enabled: Si el tracer está habilitado
        output_path: Ruta del archivo de salida (None = solo memoria)
        buffer_size: Tamaño del buffer antes de escribir a disco
        events: Lista de eventos capturados (modo síncrono)
        
    Examples:
        >>> tracer = SearchSpaceTracer(enabled=True, output_path="trace.csv")
        >>> tracer.start()
        >>> tracer.record(SearchEvent(
        ...     timestamp=time.time(),
        ...     event_type='search_started'
        ... ))
        >>> tracer.stop()
        >>> stats = tracer.get_statistics()
        >>> print(f"Nodos explorados: {stats['nodes_explored']}")
    """
    
    def __init__(
        self,
        enabled: bool = True,
        output_path: Optional[str] = None,
        buffer_size: int = 1000,
        output_format: Literal['csv', 'jsonl'] = 'csv',
        async_mode: bool = False
    ):
        """
        Inicializa el tracer.
        
        Args:
            enabled: Si el tracer está habilitado
            output_path: Ruta del archivo de salida (None = solo memoria)
            buffer_size: Tamaño del buffer antes de escribir a disco
            output_format: Formato de salida ('csv' o 'jsonl')
            async_mode: Si True, usa un worker thread para escritura asíncrona
        """
        self.enabled = enabled
        self.output_path = output_path
        self.buffer_size = buffer_size
        self.output_format = output_format
        self.async_mode = async_mode and output_path is not None
        
        # Estado interno
        self.events: List[SearchEvent] = []
        self._buffer: deque = deque(maxlen=buffer_size)
        self._file_handle = None
        self._csv_writer = None
        self._started = False
        
        # Modo asíncrono
        self._async_queue: Optional[queue.Queue] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_worker = threading.Event()
        
        # Estadísticas incrementales
        self._stats = {
            'nodes_explored': 0,
            'backtracks': 0,
            'solutions_found': 0,
            'ac3_calls': 0,
            'domain_prunes': 0,
            'cluster_operations': 0,
            'max_depth': 0
        }
    
    def start(self):
        """Inicia el tracer y abre el archivo de salida si es necesario."""
        if not self.enabled:
            return
        
        self._started = True
        
        if self.async_mode:
            # Modo asíncrono: iniciar worker thread
            self._async_queue = queue.Queue(maxsize=self.buffer_size * 2)
            self._stop_worker.clear()
            self._worker_thread = threading.Thread(
                target=self._async_worker,
                daemon=True
            )
            self._worker_thread.start()
        elif self.output_path:
            # Modo síncrono: abrir archivo directamente
            path = Path(self.output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.output_format == 'csv':
                self._file_handle = open(path, 'w', newline='', encoding='utf-8')
                self._csv_writer = csv.DictWriter(
                    self._file_handle,
                    fieldnames=[
                        'timestamp', 'event_type', 'variable', 'value',
                        'source_variable', 'pruned_values', 'depth', 'metadata'
                    ]
                )
                self._csv_writer.writeheader()
            else:  # jsonl
                self._file_handle = open(path, 'w', encoding='utf-8')
    
    def stop(self):
        """Detiene el tracer y asegura que todos los eventos se escriban."""
        if not self.enabled or not self._started:
            return
        
        if self.async_mode:
            # Modo asíncrono: señalar al worker que termine
            self._stop_worker.set()
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5.0)
            self._async_queue = None
            self._worker_thread = None
        else:
            # Modo síncrono: vaciar buffer y cerrar archivo
            self._flush_buffer()
            
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None
                self._csv_writer = None
        
        self._started = False
    
    def record(self, event: SearchEvent):
        """
        Registra un nuevo evento de búsqueda.
        
        Args:
            event: Evento a registrar
        """
        if not self.enabled:
            return
        
        # Actualizar estadísticas incrementales
        self._update_stats(event)
        
        # Almacenar en memoria
        self.events.append(event)
        
        # Escribir a archivo
        if self.async_mode and self._async_queue:
            # Modo asíncrono: enviar al worker thread
            try:
                self._async_queue.put_nowait(event)
            except queue.Full:
                # Si la cola está llena, esperar un poco
                try:
                    self._async_queue.put(event, timeout=0.1)
                except queue.Full:
                    # Si sigue llena, ignorar (pérdida de evento en trace, no en memoria)
                    pass
        elif self.output_path:
            # Modo síncrono: añadir al buffer
            self._buffer.append(event)
            
            # Vaciar buffer si está lleno
            if len(self._buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def _update_stats(self, event: SearchEvent):
        """Actualiza las estadísticas incrementales."""
        if event.event_type == 'variable_assigned':
            self._stats['nodes_explored'] += 1
            self._stats['max_depth'] = max(self._stats['max_depth'], event.depth)
        elif event.event_type == 'backtrack':
            self._stats['backtracks'] += 1
        elif event.event_type == 'solution_found':
            self._stats['solutions_found'] += 1
        elif event.event_type == 'ac3_call':
            self._stats['ac3_calls'] += 1
        elif event.event_type == 'domain_pruned':
            self._stats['domain_prunes'] += 1
        elif event.event_type == 'cluster_operation':
            self._stats['cluster_operations'] += 1
    
    def _flush_buffer(self):
        """Escribe el buffer al archivo de salida."""
        if not self._buffer or not self._file_handle:
            return
        
        while self._buffer:
            event = self._buffer.popleft()
            
            if self.output_format == 'csv':
                # Escribir como CSV
                row = event.to_dict()
                # Convertir metadata a JSON string para CSV
                row['metadata'] = json.dumps(row['metadata'])
                # Convertir pruned_values a string
                if row['pruned_values'] is not None:
                    row['pruned_values'] = json.dumps(row['pruned_values'])
                self._csv_writer.writerow(row)
            else:  # jsonl
                # Escribir como JSON Lines
                self._file_handle.write(json.dumps(event.to_dict()) + '\n')
        
        # Flush del archivo
        if self._file_handle:
            self._file_handle.flush()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Calcula y devuelve estadísticas agregadas del trace.
        
        Returns:
            Diccionario con estadísticas
            
        Examples:
            >>> stats = tracer.get_statistics()
            >>> print(f"Nodos: {stats['nodes_explored']}")
            >>> print(f"Backtracks: {stats['backtracks']}")
            >>> print(f"Tasa de backtrack: {stats['backtrack_rate']:.2%}")
        """
        if not self.enabled or not self.events:
            return self._stats.copy()
        
        stats = self._stats.copy()
        
        # Calcular métricas derivadas
        if stats['nodes_explored'] > 0:
            stats['backtrack_rate'] = stats['backtracks'] / stats['nodes_explored']
        else:
            stats['backtrack_rate'] = 0.0
        
        # Calcular duración total
        if len(self.events) >= 2:
            start_time = self.events[0].timestamp
            end_time = self.events[-1].timestamp
            stats['total_duration'] = end_time - start_time
        else:
            stats['total_duration'] = 0.0
        
        # Calcular eventos por segundo
        if stats['total_duration'] > 0:
            stats['events_per_second'] = len(self.events) / stats['total_duration']
        else:
            stats['events_per_second'] = 0.0
        
        stats['total_events'] = len(self.events)
        
        return stats
    
    def to_dataframe(self):
        """
        Convierte los eventos en memoria a un DataFrame de pandas.
        
        Returns:
            DataFrame con los eventos
            
        Raises:
            ImportError: Si pandas no está instalado
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install pandas"
            )
        
        if not self.events:
            return pd.DataFrame()
        
        # Convertir eventos a lista de diccionarios
        data = [event.to_dict() for event in self.events]
        
        return pd.DataFrame(data)
    
    def _async_worker(self):
        """
        Worker thread para escritura asíncrona de eventos.
        
        Este método se ejecuta en un thread separado y consume eventos
        de la cola para escribirlos al archivo.
        """
        # Abrir archivo en el worker thread
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handle = None
        csv_writer = None
        
        try:
            if self.output_format == 'csv':
                file_handle = open(path, 'w', newline='', encoding='utf-8')
                csv_writer = csv.DictWriter(
                    file_handle,
                    fieldnames=[
                        'timestamp', 'event_type', 'variable', 'value',
                        'source_variable', 'pruned_values', 'depth', 'metadata'
                    ]
                )
                csv_writer.writeheader()
            else:  # jsonl
                file_handle = open(path, 'w', encoding='utf-8')
            
            # Buffer local para batch writing
            local_buffer = []
            
            while not self._stop_worker.is_set():
                if self._async_queue is None:
                    break
                
                if self._async_queue.empty() and self._stop_worker.is_set():
                    break
                try:
                    # Intentar obtener evento de la cola
                    event = self._async_queue.get(timeout=0.1)
                    local_buffer.append(event)
                    
                    # Escribir en batch cuando el buffer local está lleno
                    if len(local_buffer) >= self.buffer_size:
                        self._write_events_batch(file_handle, csv_writer, local_buffer)
                        local_buffer.clear()
                        
                except queue.Empty:
                    # Si no hay eventos, escribir lo que haya en el buffer
                    if local_buffer:
                        self._write_events_batch(file_handle, csv_writer, local_buffer)
                        local_buffer.clear()
            
            # Escribir eventos restantes
            if local_buffer:
                self._write_events_batch(file_handle, csv_writer, local_buffer)
        
        finally:
            if file_handle:
                file_handle.close()
    
    def _write_events_batch(self, file_handle, csv_writer, events):
        """Escribe un batch de eventos al archivo."""
        for event in events:
            if self.output_format == 'csv':
                row = event.to_dict()
                row['metadata'] = json.dumps(row['metadata'])
                if row['pruned_values'] is not None:
                    row['pruned_values'] = json.dumps(row['pruned_values'])
                csv_writer.writerow(row)
            else:  # jsonl
                file_handle.write(json.dumps(event.to_dict()) + '\n')
        
        # Flush después de cada batch
        if file_handle:
            file_handle.flush()
    
    def clear(self):
        """Limpia todos los eventos y estadísticas."""
        self.events.clear()
        self._buffer.clear()
        self._stats = {
            'nodes_explored': 0,
            'backtracks': 0,
            'solutions_found': 0,
            'ac3_calls': 0,
            'domain_prunes': 0,
            'cluster_operations': 0,
            'max_depth': 0
        }
    
    def __enter__(self):
        """Context manager: inicio."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: fin."""
        self.stop()
        return False


# Funciones de utilidad para cargar traces

def load_trace_csv(path: str):
    """
    Carga un archivo de trace en formato CSV.
    
    Args:
        path: Ruta del archivo CSV
        
    Returns:
        DataFrame de pandas con los eventos
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for load_trace_csv(). "
            "Install it with: pip install pandas"
        )
    
    df = pd.read_csv(path)
    
    # Parsear columnas JSON
    if 'metadata' in df.columns:
        df['metadata'] = df['metadata'].apply(
            lambda x: json.loads(x) if pd.notna(x) else {}
        )
    
    if 'pruned_values' in df.columns:
        df['pruned_values'] = df['pruned_values'].apply(
            lambda x: json.loads(x) if pd.notna(x) else None
        )
    
    return df


def load_trace_jsonl(path: str):
    """
    Carga un archivo de trace en formato JSON Lines.
    
    Args:
        path: Ruta del archivo JSONL
        
    Returns:
        DataFrame de pandas con los eventos
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for load_trace_jsonl(). "
            "Install it with: pip install pandas"
        )
    
    events = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            events.append(json.loads(line))
    
    return pd.DataFrame(events)


def load_trace(path: str):
    """
    Carga un archivo de trace automáticamente detectando el formato.
    
    Args:
        path: Ruta del archivo
        
    Returns:
        DataFrame de pandas con los eventos
    """
    path_obj = Path(path)
    
    if path_obj.suffix == '.csv':
        return load_trace_csv(path)
    elif path_obj.suffix in ['.jsonl', '.json']:
        return load_trace_jsonl(path)
    else:
        raise ValueError(
            f"Formato de archivo no soportado: {path_obj.suffix}. "
            "Use .csv o .jsonl"
        )

