"""
Visualizaciones para Benchmarks

Este módulo crea visualizaciones interactivas usando Plotly para análisis
de resultados de benchmarks.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
import json


@dataclass
class BenchmarkResult:
    """Resultado de un benchmark individual."""
    problem_name: str
    algorithm_name: str
    time_ms: float
    memory_mb: float
    nodes_explored: int
    backtracks: int
    success: bool
    problem_size: Optional[int] = None


class BenchmarkVisualizer:
    """
    Creador de visualizaciones para benchmarks.
    
    Genera gráficos interactivos con Plotly para analizar resultados
    de benchmarks y comparar algoritmos.
    
    Example:
        >>> visualizer = BenchmarkVisualizer()
        >>> results = {...}
        >>> fig = visualizer.create_time_comparison_chart(results)
        >>> fig.write_html("time_comparison.html")
    """
    
    def __init__(self, theme: str = "plotly_white"):
        """
        Inicializa el visualizador.
        
        Args:
            theme: Tema de Plotly ("plotly", "plotly_white", "plotly_dark")
        """
        self.theme = theme
        self.colors = px.colors.qualitative.Set2
    
    def create_time_comparison_chart(
        self,
        results: Dict[str, BenchmarkResult],
        title: str = "Comparación de Tiempos de Ejecución"
    ) -> go.Figure:
        """
        Crea gráfico de barras comparando tiempos de ejecución.
        
        Args:
            results: Diccionario de resultados (key: nombre, value: resultado)
            title: Título del gráfico
        
        Returns:
            Figura de Plotly
        
        Example:
            >>> results = {
            ...     "BT": BenchmarkResult("nqueens_4", "Backtracking", 0.09, ...),
            ...     "FC": BenchmarkResult("nqueens_4", "Forward Checking", 0.24, ...)
            ... }
            >>> fig = visualizer.create_time_comparison_chart(results)
        """
        algorithms = list(results.keys())
        times = [results[alg].time_ms for alg in algorithms]
        
        fig = go.Figure(data=[
            go.Bar(
                x=algorithms,
                y=times,
                text=[f"{t:.2f} ms" for t in times],
                textposition='auto',
                marker_color=self.colors[:len(algorithms)]
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Algoritmo",
            yaxis_title="Tiempo (ms)",
            template=self.theme,
            hovermode='x unified'
        )
        
        return fig
    
    def create_memory_comparison_chart(
        self,
        results: Dict[str, BenchmarkResult],
        title: str = "Comparación de Uso de Memoria"
    ) -> go.Figure:
        """
        Crea gráfico de barras comparando uso de memoria.
        
        Args:
            results: Diccionario de resultados
            title: Título del gráfico
        
        Returns:
            Figura de Plotly
        """
        algorithms = list(results.keys())
        memory = [results[alg].memory_mb for alg in algorithms]
        
        fig = go.Figure(data=[
            go.Bar(
                x=algorithms,
                y=memory,
                text=[f"{m:.2f} MB" for m in memory],
                textposition='auto',
                marker_color=self.colors[:len(algorithms)]
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Algoritmo",
            yaxis_title="Memoria (MB)",
            template=self.theme,
            hovermode='x unified'
        )
        
        return fig
    
    def create_speedup_heatmap(
        self,
        results: Dict[str, Dict[str, BenchmarkResult]],
        baseline: str = "Backtracking",
        title: str = "Heatmap de Speedups"
    ) -> go.Figure:
        """
        Crea heatmap de speedups relativos a un algoritmo baseline.
        
        Args:
            results: Diccionario anidado {problema: {algoritmo: resultado}}
            baseline: Algoritmo baseline para calcular speedups
            title: Título del gráfico
        
        Returns:
            Figura de Plotly
        
        Example:
            >>> results = {
            ...     "nqueens_4": {
            ...         "Backtracking": BenchmarkResult(..., time_ms=0.09),
            ...         "Forward Checking": BenchmarkResult(..., time_ms=0.24)
            ...     },
            ...     "nqueens_6": {...}
            ... }
            >>> fig = visualizer.create_speedup_heatmap(results)
        """
        # Extraer problemas y algoritmos
        problems = list(results.keys())
        algorithms = list(next(iter(results.values())).keys())
        
        # Calcular speedups
        speedup_matrix = []
        for alg in algorithms:
            speedups = []
            for problem in problems:
                baseline_time = results[problem][baseline].time_ms
                alg_time = results[problem][alg].time_ms
                speedup = baseline_time / alg_time if alg_time > 0 else 0
                speedups.append(speedup)
            speedup_matrix.append(speedups)
        
        fig = go.Figure(data=go.Heatmap(
            z=speedup_matrix,
            x=problems,
            y=algorithms,
            colorscale='RdYlGn',
            text=[[f"{v:.2f}x" for v in row] for row in speedup_matrix],
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Speedup")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Problema",
            yaxis_title="Algoritmo",
            template=self.theme
        )
        
        return fig
    
    def create_scalability_chart(
        self,
        results: Dict[int, Dict[str, BenchmarkResult]],
        title: str = "Análisis de Escalabilidad",
        log_scale: bool = True
    ) -> go.Figure:
        """
        Crea gráfico de escalabilidad (tiempo vs tamaño del problema).
        
        Args:
            results: Diccionario {tamaño: {algoritmo: resultado}}
            title: Título del gráfico
            log_scale: Si usar escala logarítmica
        
        Returns:
            Figura de Plotly
        
        Example:
            >>> results = {
            ...     4: {"BT": BenchmarkResult(...), "FC": BenchmarkResult(...)},
            ...     6: {"BT": BenchmarkResult(...), "FC": BenchmarkResult(...)},
            ...     8: {"BT": BenchmarkResult(...), "FC": BenchmarkResult(...)}
            ... }
            >>> fig = visualizer.create_scalability_chart(results)
        """
        sizes = sorted(results.keys())
        algorithms = list(next(iter(results.values())).keys())
        
        fig = go.Figure()
        
        for i, alg in enumerate(algorithms):
            times = [results[size][alg].time_ms for size in sizes]
            
            fig.add_trace(go.Scatter(
                x=sizes,
                y=times,
                mode='lines+markers',
                name=alg,
                line=dict(color=self.colors[i % len(self.colors)], width=2),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Tamaño del Problema",
            yaxis_title="Tiempo (ms)",
            template=self.theme,
            hovermode='x unified'
        )
        
        if log_scale:
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log")
        
        return fig
    
    def create_success_rate_chart(
        self,
        results: Dict[str, BenchmarkResult],
        title: str = "Tasa de Éxito por Algoritmo"
    ) -> go.Figure:
        """
        Crea pie chart de tasa de éxito.
        
        Args:
            results: Diccionario de resultados
            title: Título del gráfico
        
        Returns:
            Figura de Plotly
        """
        algorithms = list(results.keys())
        success_counts = [1 if results[alg].success else 0 for alg in algorithms]
        
        fig = go.Figure(data=[go.Pie(
            labels=algorithms,
            values=success_counts,
            marker_colors=self.colors[:len(algorithms)]
        )])
        
        fig.update_layout(
            title=title,
            template=self.theme
        )
        
        return fig
    
    def create_nodes_vs_time_scatter(
        self,
        results: Dict[str, BenchmarkResult],
        title: str = "Nodos Explorados vs Tiempo"
    ) -> go.Figure:
        """
        Crea scatter plot de nodos explorados vs tiempo.
        
        Args:
            results: Diccionario de resultados
            title: Título del gráfico
        
        Returns:
            Figura de Plotly
        """
        algorithms = list(results.keys())
        nodes = [results[alg].nodes_explored for alg in algorithms]
        times = [results[alg].time_ms for alg in algorithms]
        
        fig = go.Figure(data=[go.Scatter(
            x=nodes,
            y=times,
            mode='markers+text',
            text=algorithms,
            textposition='top center',
            marker=dict(
                size=12,
                color=self.colors[:len(algorithms)]
            )
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title="Nodos Explorados",
            yaxis_title="Tiempo (ms)",
            template=self.theme
        )
        
        return fig
    
    def create_comprehensive_dashboard(
        self,
        results: Dict[str, Dict[str, BenchmarkResult]],
        output_file: str = "dashboard.html"
    ) -> Path:
        """
        Crea dashboard completo con múltiples visualizaciones.
        
        Args:
            results: Diccionario anidado de resultados
            output_file: Archivo de salida
        
        Returns:
            Path al archivo generado
        """
        from plotly.subplots import make_subplots
        
        # Crear subplot con 2x2 gráficos
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Comparación de Tiempos",
                "Comparación de Memoria",
                "Nodos Explorados",
                "Speedups"
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ]
        )
        
        # Aquí se agregarían los datos a cada subplot
        # (Simplificado para el ejemplo)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Dashboard de Benchmarks - LatticeWeaver",
            template=self.theme
        )
        
        output_path = Path(output_file)
        fig.write_html(str(output_path))
        
        return output_path


# ============================================================================
# Funciones de Utilidad
# ============================================================================

def load_benchmark_results(json_file: Path) -> Dict[str, BenchmarkResult]:
    """
    Carga resultados de benchmarks desde archivo JSON.
    
    Args:
        json_file: Path al archivo JSON
    
    Returns:
        Diccionario de resultados
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = {}
    for key, value in data.items():
        results[key] = BenchmarkResult(**value)
    
    return results


def save_all_charts(
    results: Dict[str, BenchmarkResult],
    output_dir: Path
):
    """
    Guarda todos los gráficos estándar en un directorio.
    
    Args:
        results: Diccionario de resultados
        output_dir: Directorio de salida
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = BenchmarkVisualizer()
    
    # Gráfico de tiempos
    fig = visualizer.create_time_comparison_chart(results)
    fig.write_html(str(output_dir / "time_comparison.html"))
    
    # Gráfico de memoria
    fig = visualizer.create_memory_comparison_chart(results)
    fig.write_html(str(output_dir / "memory_comparison.html"))
    
    # Scatter de nodos vs tiempo
    fig = visualizer.create_nodes_vs_time_scatter(results)
    fig.write_html(str(output_dir / "nodes_vs_time.html"))
    
    print(f"✅ Gráficos guardados en {output_dir}")

