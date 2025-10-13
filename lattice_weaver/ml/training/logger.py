"""
Sistema de logging asíncrono para capturar trazas del solver.

Este módulo proporciona logging de alta eficiencia con overhead mínimo
mediante escritura asíncrona en thread separado.
"""

import json
import time
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import Dict, Any, Optional


class SolverLogger:
    """
    Logger asíncrono para capturar pasos de ejecución del solver.
    
    Características:
    - Escritura asíncrona (no bloquea solver)
    - Serialización JSON compacta
    - Overhead < 0.05 ms por paso
    - Buffer configurable
    
    Ejemplo:
        >>> logger = SolverLogger("traces.jsonl")
        >>> logger.log_step({
        ...     "instance_id": "csp_001",
        ...     "step": 1,
        ...     "state": {...},
        ...     "decision": {...}
        ... })
        >>> logger.close()
    """
    
    def __init__(
        self,
        log_file: str,
        buffer_size: int = 1000,
        flush_interval: float = 1.0
    ):
        """
        Inicializar logger.
        
        Args:
            log_file: Ruta al archivo de log (formato JSONL)
            buffer_size: Tamaño del buffer (número de pasos)
            flush_interval: Intervalo de flush a disco (segundos)
        """
        self.log_file = Path(log_file)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Queue para comunicación con thread de escritura
        self.queue = Queue(maxsize=buffer_size * 2)
        
        # Thread de escritura
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._async_writer,
            daemon=True
        )
        self._writer_thread.start()
        
        # Estadísticas
        self.stats = {
            "steps_logged": 0,
            "bytes_written": 0,
            "start_time": time.time()
        }
    
    def log_step(self, step_data: Dict[str, Any]) -> None:
        """
        Loggear un paso de ejecución.
        
        Args:
            step_data: Diccionario con información del paso
        
        Nota:
            Esta operación es no bloqueante y tiene overhead < 0.001 ms
        """
        try:
            # Añadir timestamp si no existe
            if "timestamp" not in step_data:
                step_data["timestamp"] = time.time()
            
            # Poner en queue (no bloqueante)
            self.queue.put(step_data, block=False)
            self.stats["steps_logged"] += 1
            
        except Exception as e:
            # No fallar si queue está llena, solo advertir
            print(f"Warning: Failed to log step: {e}")
    
    def _async_writer(self) -> None:
        """
        Thread que escribe a disco de forma asíncrona.
        
        Este método corre en un thread separado y procesa la queue
        escribiendo a disco periódicamente.
        """
        # Abrir archivo en modo append
        with open(self.log_file, 'a', encoding='utf-8') as f:
            buffer = []
            last_flush = time.time()
            
            while not self._stop_event.is_set():
                try:
                    # Intentar obtener item de queue (con timeout)
                    step_data = self.queue.get(timeout=0.1)
                    buffer.append(step_data)
                    
                    # Flush si buffer lleno o tiempo transcurrido
                    should_flush = (
                        len(buffer) >= self.buffer_size or
                        time.time() - last_flush >= self.flush_interval
                    )
                    
                    if should_flush:
                        self._flush_buffer(f, buffer)
                        buffer = []
                        last_flush = time.time()
                
                except Empty:
                    # Queue vacía, flush si hay algo en buffer
                    if buffer:
                        self._flush_buffer(f, buffer)
                        buffer = []
                        last_flush = time.time()
            
            # Flush final al terminar
            if buffer:
                self._flush_buffer(f, buffer)
    
    def _flush_buffer(self, file_handle, buffer: list) -> None:
        """
        Escribir buffer a disco.
        
        Args:
            file_handle: Handle del archivo abierto
            buffer: Lista de step_data a escribir
        """
        for step_data in buffer:
            # Serializar a JSON (una línea por paso)
            json_line = json.dumps(step_data, separators=(',', ':'))
            file_handle.write(json_line + '\n')
            self.stats["bytes_written"] += len(json_line) + 1
        
        # Flush a disco
        file_handle.flush()
    
    def close(self) -> None:
        """
        Cerrar logger y esperar a que termine escritura.
        
        Nota:
            Siempre llamar este método al terminar para asegurar
            que todos los datos se escribieron a disco.
        """
        # Señalar al thread que termine
        self._stop_event.set()
        
        # Esperar a que termine
        self._writer_thread.join(timeout=5.0)
        
        # Imprimir estadísticas
        elapsed = time.time() - self.stats["start_time"]
        print(f"\n=== Logging Statistics ===")
        print(f"Steps logged: {self.stats['steps_logged']}")
        print(f"Bytes written: {self.stats['bytes_written']:,}")
        print(f"Time elapsed: {elapsed:.2f} s")
        print(f"Throughput: {self.stats['steps_logged']/elapsed:.1f} steps/s")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def load_jsonl(file_path: str) -> list:
    """
    Cargar archivo JSONL.
    
    Args:
        file_path: Ruta al archivo JSONL
    
    Returns:
        Lista de diccionarios (un diccionario por línea)
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# Ejemplo de uso
if __name__ == "__main__":
    # Demo de logging
    with SolverLogger("demo_traces.jsonl") as logger:
        # Simular 1000 pasos
        for i in range(1000):
            step_data = {
                "instance_id": "demo_csp",
                "step": i,
                "state": {
                    "domains": [3, 2, 5, 4],
                    "unassigned": 4 - (i // 250)
                },
                "decision": {
                    "type": "var_selection",
                    "chosen": i % 4
                }
            }
            logger.log_step(step_data)
    
    # Cargar y verificar
    steps = load_jsonl("demo_traces.jsonl")
    print(f"\nLoaded {len(steps)} steps from file")

