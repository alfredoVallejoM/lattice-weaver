from typing import List, Dict, Any
import time
import threading

from lattice_weaver.paging.page_manager import PageManager
from lattice_weaver.validation.paging_validator import PagingValidator, ValidationCertificate

class PagingMonitor:
    def __init__(self, page_manager: PageManager, paging_validator: PagingValidator, interval: int = 60):
        self.page_manager = page_manager
        self.paging_validator = paging_validator
        self.interval = interval  # Intervalo de monitoreo en segundos
        self._monitoring_thread = None
        self._stop_event = threading.Event()
        self.monitoring_results: List[Dict[str, Any]] = []

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            print("\n--- Ejecutando monitoreo de paginación ---")
            # Ejecutar validaciones de coherencia
            # Nota: Para una validación continua, se necesitaría una forma de especificar qué páginas validar
            # Por ahora, solo mediremos el rendimiento.
            
            # Medir rendimiento
            performance_metrics = self.paging_validator.measure_performance()
            self.monitoring_results.append({
                "timestamp": time.time(),
                "metrics": performance_metrics
            })
            print(f"Métricas de rendimiento registradas: {performance_metrics}")
            
            self._stop_event.wait(self.interval)

    def start_monitoring(self):
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_event.clear()
            self._monitoring_thread = threading.Thread(target=self._monitor_loop)
            self._monitoring_thread.daemon = True  # Permite que el programa principal se cierre incluso si el hilo está corriendo
            self._monitoring_thread.start()
            print(f"Monitoreo de paginación iniciado con un intervalo de {self.interval} segundos.")

    def stop_monitoring(self):
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_event.set()
            self._monitoring_thread.join()
            print("Monitoreo de paginación detenido.")

    def get_monitoring_results(self) -> List[Dict[str, Any]]:
        return self.monitoring_results

    def __del__(self):
        self.stop_monitoring()

