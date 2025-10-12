"""
MetricsCollector: Recolección de métricas de rendimiento y estadísticas.

Este módulo proporciona un sistema completo de recolección de métricas para
analizar el rendimiento de LatticeWeaver y compararlo con otros solvers.

Autor: Manus AI
Fecha: 11 de Octubre de 2025
"""

import time
import psutil
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import csv


@dataclass
class PhaseMetrics:
    """
    Métricas de una fase de ejecución.
    
    Attributes:
        name: Nombre de la fase
        start_time: Tiempo de inicio (timestamp)
        end_time: Tiempo de fin (timestamp)
        duration: Duración en segundos
        memory_start: Memoria al inicio (MB)
        memory_end: Memoria al final (MB)
        memory_peak: Memoria pico durante la fase (MB)
    """
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    memory_start: float = 0.0
    memory_end: float = 0.0
    memory_peak: float = 0.0
    
    def to_dict(self) -> dict:
        """Serializa las métricas a un diccionario."""
        return {
            'name': self.name,
            'duration_seconds': round(self.duration, 6),
            'duration_ms': round(self.duration * 1000, 3),
            'memory_start_mb': round(self.memory_start, 2),
            'memory_end_mb': round(self.memory_end, 2),
            'memory_peak_mb': round(self.memory_peak, 2),
            'memory_delta_mb': round(self.memory_end - self.memory_start, 2)
        }


class MetricsCollector:
    """
    Recolector de métricas de rendimiento.
    
    Recopila métricas detalladas de tiempo, memoria y operaciones durante
    la ejecución de LatticeWeaver, permitiendo análisis de rendimiento y
    comparaciones con otros solvers.
    
    Attributes:
        phases: Métricas de cada fase
        active_phases: Fases actualmente en ejecución
        custom_metrics: Métricas personalizadas
        process: Proceso actual de Python
    """
    
    def __init__(self):
        """Inicializa el recolector de métricas."""
        self.phases: Dict[str, PhaseMetrics] = {}
        self.active_phases: Dict[str, PhaseMetrics] = {}
        self.custom_metrics: Dict[str, Any] = {}
        
        # Proceso actual para monitoreo de memoria
        self.process = psutil.Process(os.getpid())
        
        # Tiempo de inicio global
        self.global_start_time = time.time()
    
    def _get_memory_mb(self) -> float:
        """
        Obtiene el uso de memoria actual en MB.
        
        Returns:
            Memoria en MB
        """
        return self.process.memory_info().rss / (1024 * 1024)
    
    def start_timer(self, phase: str):
        """
        Inicia el temporizador para una fase.
        
        Args:
            phase: Nombre de la fase
        """
        if phase in self.active_phases:
            print(f"Warning: La fase '{phase}' ya está activa.")
            return
        
        metrics = PhaseMetrics(name=phase)
        metrics.start_time = time.time()
        metrics.memory_start = self._get_memory_mb()
        metrics.memory_peak = metrics.memory_start
        
        self.active_phases[phase] = metrics
    
    def stop_timer(self, phase: str):
        """
        Detiene el temporizador para una fase.
        
        Args:
            phase: Nombre de la fase
        """
        if phase not in self.active_phases:
            print(f"Warning: La fase '{phase}' no está activa.")
            return
        
        metrics = self.active_phases[phase]
        metrics.end_time = time.time()
        metrics.duration = metrics.end_time - metrics.start_time
        metrics.memory_end = self._get_memory_mb()
        metrics.memory_peak = max(metrics.memory_peak, metrics.memory_end)
        
        # Mover a fases completadas
        self.phases[phase] = metrics
        del self.active_phases[phase]
    
    def update_memory_peak(self, phase: str):
        """
        Actualiza el pico de memoria para una fase activa.
        
        Args:
            phase: Nombre de la fase
        """
        if phase in self.active_phases:
            current_memory = self._get_memory_mb()
            self.active_phases[phase].memory_peak = max(
                self.active_phases[phase].memory_peak,
                current_memory
            )
    
    def record_metric(self, name: str, value: Any):
        """
        Registra una métrica personalizada.
        
        Args:
            name: Nombre de la métrica
            value: Valor de la métrica
        """
        self.custom_metrics[name] = value
    
    def increment_metric(self, name: str, delta: int = 1):
        """
        Incrementa una métrica contador.
        
        Args:
            name: Nombre de la métrica
            delta: Cantidad a incrementar
        """
        if name not in self.custom_metrics:
            self.custom_metrics[name] = 0
        self.custom_metrics[name] += delta
    
    def get_phase_duration(self, phase: str) -> Optional[float]:
        """
        Obtiene la duración de una fase.
        
        Args:
            phase: Nombre de la fase
            
        Returns:
            Duración en segundos, o None si la fase no existe
        """
        if phase in self.phases:
            return self.phases[phase].duration
        return None
    
    def get_total_duration(self) -> float:
        """
        Obtiene la duración total desde el inicio.
        
        Returns:
            Duración en segundos
        """
        return time.time() - self.global_start_time
    
    def get_summary(self) -> dict:
        """
        Obtiene un resumen completo de las métricas.
        
        Returns:
            Diccionario con todas las métricas
        """
        return {
            'total_duration_seconds': round(self.get_total_duration(), 6),
            'phases': {name: metrics.to_dict() for name, metrics in self.phases.items()},
            'custom_metrics': self.custom_metrics.copy(),
            'memory_current_mb': round(self._get_memory_mb(), 2)
        }
    
    def print_summary(self):
        """Imprime un resumen legible de las métricas."""
        print("\n" + "="*60)
        print("RESUMEN DE MÉTRICAS")
        print("="*60)
        
        print(f"\nDuración Total: {self.get_total_duration():.6f} segundos")
        print(f"Memoria Actual: {self._get_memory_mb():.2f} MB")
        
        if self.phases:
            print("\nFases:")
            print("-" * 60)
            for name, metrics in self.phases.items():
                print(f"  {name}:")
                print(f"    Duración: {metrics.duration*1000:.3f} ms")
                print(f"    Memoria: {metrics.memory_start:.2f} → {metrics.memory_end:.2f} MB (pico: {metrics.memory_peak:.2f} MB)")
        
        if self.custom_metrics:
            print("\nMétricas Personalizadas:")
            print("-" * 60)
            for name, value in self.custom_metrics.items():
                print(f"  {name}: {value}")
        
        print("="*60 + "\n")
    
    def export_to_json(self, filepath: str):
        """
        Exporta las métricas a un archivo JSON.
        
        Args:
            filepath: Ruta del archivo
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.get_summary(), f, indent=2)
    
    def export_to_csv(self, filepath: str):
        """
        Exporta las métricas de fases a un archivo CSV.
        
        Args:
            filepath: Ruta del archivo
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Encabezados
            writer.writerow([
                'Phase', 'Duration (s)', 'Duration (ms)',
                'Memory Start (MB)', 'Memory End (MB)', 'Memory Peak (MB)', 'Memory Delta (MB)'
            ])
            
            # Datos
            for name, metrics in self.phases.items():
                writer.writerow([
                    name,
                    round(metrics.duration, 6),
                    round(metrics.duration * 1000, 3),
                    round(metrics.memory_start, 2),
                    round(metrics.memory_end, 2),
                    round(metrics.memory_peak, 2),
                    round(metrics.memory_end - metrics.memory_start, 2)
                ])
    
    def reset(self):
        """Reinicia todas las métricas."""
        self.phases.clear()
        self.active_phases.clear()
        self.custom_metrics.clear()
        self.global_start_time = time.time()


class BenchmarkComparison:
    """
    Comparación de métricas entre múltiples solvers.
    
    Permite comparar el rendimiento de LatticeWeaver con otros solvers
    en los mismos problemas.
    """
    
    def __init__(self):
        """Inicializa el comparador."""
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def add_result(self, solver_name: str, problem_name: str, metrics: dict):
        """
        Añade un resultado de un solver.
        
        Args:
            solver_name: Nombre del solver
            problem_name: Nombre del problema
            metrics: Métricas del resultado
        """
        if solver_name not in self.results:
            self.results[solver_name] = {}
        
        self.results[solver_name][problem_name] = metrics
    
    def get_comparison_table(self, metric: str = 'duration') -> List[List[Any]]:
        """
        Genera una tabla de comparación para una métrica.
        
        Args:
            metric: Nombre de la métrica a comparar
            
        Returns:
            Lista de listas (tabla) con la comparación
        """
        if not self.results:
            return []
        
        # Obtener todos los problemas
        all_problems = set()
        for solver_results in self.results.values():
            all_problems.update(solver_results.keys())
        
        all_problems = sorted(all_problems)
        
        # Encabezados
        table = [['Problem'] + list(self.results.keys())]
        
        # Datos
        for problem in all_problems:
            row = [problem]
            for solver_name in self.results.keys():
                if problem in self.results[solver_name]:
                    value = self.results[solver_name][problem].get(metric, 'N/A')
                    row.append(value)
                else:
                    row.append('N/A')
            table.append(row)
        
        return table
    
    def print_comparison(self, metric: str = 'duration'):
        """
        Imprime una tabla de comparación.
        
        Args:
            metric: Nombre de la métrica a comparar
        """
        table = self.get_comparison_table(metric)
        
        if not table:
            print("No hay resultados para comparar.")
            return
        
        print(f"\nComparación de {metric}:")
        print("="*80)
        
        # Calcular anchos de columna
        col_widths = [max(len(str(row[i])) for row in table) + 2 for i in range(len(table[0]))]
        
        # Imprimir tabla
        for i, row in enumerate(table):
            print("".join(str(cell).ljust(col_widths[j]) for j, cell in enumerate(row)))
            if i == 0:
                print("-"*80)
        
        print("="*80 + "\n")
    
    def export_comparison(self, filepath: str, metric: str = 'duration'):
        """
        Exporta la comparación a un archivo CSV.
        
        Args:
            filepath: Ruta del archivo
            metric: Métrica a comparar
        """
        table = self.get_comparison_table(metric)
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(table)

