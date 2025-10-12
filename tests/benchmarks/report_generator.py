"""
Generador de Reportes HTML para Benchmarks

Este módulo genera reportes HTML profesionales con visualizaciones
interactivas de resultados de benchmarks.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import json

from .visualizations import BenchmarkVisualizer, BenchmarkResult


class ReportGenerator:
    """
    Generador de reportes HTML desde resultados de benchmarks.
    
    Genera reportes profesionales con tablas, gráficos interactivos
    y análisis de resultados.
    
    Example:
        >>> generator = ReportGenerator(output_dir="reports")
        >>> results = {...}
        >>> report_path = generator.generate_benchmark_report(results)
        >>> print(f"Reporte generado: {report_path}")
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Inicializa el generador de reportes.
        
        Args:
            output_dir: Directorio donde guardar los reportes
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = BenchmarkVisualizer()
    
    def generate_benchmark_report(
        self,
        results: Dict[str, BenchmarkResult],
        output_file: str = "benchmark_report.html",
        title: str = "Benchmark Report"
    ) -> Path:
        """
        Genera reporte HTML principal con todos los benchmarks.
        
        Args:
            results: Diccionario de resultados
            output_file: Nombre del archivo de salida
            title: Título del reporte
        
        Returns:
            Path al archivo generado
        
        Example:
            >>> results = {
            ...     "BT": BenchmarkResult("nqueens_4", "Backtracking", ...),
            ...     "FC": BenchmarkResult("nqueens_4", "Forward Checking", ...)
            ... }
            >>> path = generator.generate_benchmark_report(results)
        """
        # Calcular métricas del resumen
        total_benchmarks = len(results)
        algorithms = set(r.algorithm_name for r in results.values())
        total_algorithms = len(algorithms)
        
        # Encontrar mejor algoritmo (menor tiempo promedio)
        avg_times = {}
        for alg in algorithms:
            alg_results = [r for r in results.values() if r.algorithm_name == alg]
            avg_times[alg] = sum(r.time_ms for r in alg_results) / len(alg_results)
        
        best_algorithm = min(avg_times, key=avg_times.get)
        
        # Calcular speedup promedio vs el más lento
        slowest_time = max(avg_times.values())
        fastest_time = min(avg_times.values())
        avg_speedup = slowest_time / fastest_time if fastest_time > 0 else 1.0
        
        # Crear gráficos
        time_chart = self.visualizer.create_time_comparison_chart(results)
        memory_chart = self.visualizer.create_memory_comparison_chart(results)
        nodes_chart = self.visualizer.create_nodes_vs_time_scatter(results)
        
        # Convertir gráficos a JSON para insertar en HTML
        time_chart_json = time_chart.to_json()
        memory_chart_json = memory_chart.to_json()
        nodes_chart_json = nodes_chart.to_json()
        
        # Generar HTML
        html = self._generate_html(
            title=title,
            total_benchmarks=total_benchmarks,
            total_algorithms=total_algorithms,
            best_algorithm=best_algorithm,
            avg_speedup=avg_speedup,
            results=results,
            time_chart_json=time_chart_json,
            memory_chart_json=memory_chart_json,
            nodes_chart_json=nodes_chart_json
        )
        
        # Guardar archivo
        output_path = self.output_dir / output_file
        output_path.write_text(html, encoding='utf-8')
        
        return output_path
    
    def generate_comparison_report(
        self,
        results: Dict[str, Dict[str, BenchmarkResult]],
        output_file: str = "comparison_report.html",
        baseline: str = "Backtracking"
    ) -> Path:
        """
        Genera reporte de comparación entre algoritmos.
        
        Args:
            results: Diccionario anidado {problema: {algoritmo: resultado}}
            output_file: Nombre del archivo de salida
            baseline: Algoritmo baseline para speedups
        
        Returns:
            Path al archivo generado
        """
        # Crear heatmap de speedups
        speedup_heatmap = self.visualizer.create_speedup_heatmap(results, baseline)
        
        # Generar HTML de comparación
        html = self._generate_comparison_html(
            results=results,
            baseline=baseline,
            speedup_heatmap_json=speedup_heatmap.to_json()
        )
        
        output_path = self.output_dir / output_file
        output_path.write_text(html, encoding='utf-8')
        
        return output_path
    
    def generate_scalability_report(
        self,
        results: Dict[int, Dict[str, BenchmarkResult]],
        output_file: str = "scalability_report.html"
    ) -> Path:
        """
        Genera reporte de análisis de escalabilidad.
        
        Args:
            results: Diccionario {tamaño: {algoritmo: resultado}}
            output_file: Nombre del archivo de salida
        
        Returns:
            Path al archivo generado
        """
        # Crear gráfico de escalabilidad
        scalability_chart = self.visualizer.create_scalability_chart(results)
        
        # Generar HTML
        html = self._generate_scalability_html(
            results=results,
            scalability_chart_json=scalability_chart.to_json()
        )
        
        output_path = self.output_dir / output_file
        output_path.write_text(html, encoding='utf-8')
        
        return output_path
    
    def _generate_html(
        self,
        title: str,
        total_benchmarks: int,
        total_algorithms: int,
        best_algorithm: str,
        avg_speedup: float,
        results: Dict[str, BenchmarkResult],
        time_chart_json: str,
        memory_chart_json: str,
        nodes_chart_json: str
    ) -> str:
        """Genera HTML completo del reporte."""
        generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generar filas de la tabla
        table_rows = ""
        for key, result in results.items():
            table_rows += f"""
            <tr>
                <td>{result.problem_name}</td>
                <td>{result.algorithm_name}</td>
                <td>{result.time_ms:.3f}</td>
                <td>{result.memory_mb:.2f}</td>
                <td>{result.nodes_explored}</td>
                <td>{result.backtracks}</td>
                <td>{'✅' if result.success else '❌'}</td>
            </tr>
            """
        
        html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - LatticeWeaver</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <header>
        <h1>🔬 LatticeWeaver Benchmarking</h1>
        <nav>
            <a href="benchmark_report.html">Benchmarks</a>
            <a href="comparison_report.html">Comparación</a>
            <a href="scalability_report.html">Escalabilidad</a>
        </nav>
    </header>
    
    <main>
        <section class="summary">
            <h2>{title}</h2>
            <div class="metrics">
                <div class="metric">
                    <h3>{total_benchmarks}</h3>
                    <p>Benchmarks Ejecutados</p>
                </div>
                <div class="metric">
                    <h3>{total_algorithms}</h3>
                    <p>Algoritmos Comparados</p>
                </div>
                <div class="metric">
                    <h3>{best_algorithm}</h3>
                    <p>Mejor Algoritmo</p>
                </div>
                <div class="metric">
                    <h3>{avg_speedup:.2f}x</h3>
                    <p>Speedup Promedio</p>
                </div>
            </div>
        </section>
        
        <section class="results">
            <h2>📊 Resultados Detallados</h2>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Problema</th>
                        <th>Algoritmo</th>
                        <th>Tiempo (ms)</th>
                        <th>Memoria (MB)</th>
                        <th>Nodos</th>
                        <th>Backtracks</th>
                        <th>Éxito</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </section>
        
        <section class="visualizations">
            <h2>📈 Visualizaciones</h2>
            
            <div class="chart-container">
                <h3>Comparación de Tiempos</h3>
                <div id="time-comparison-chart"></div>
            </div>
            
            <div class="chart-container">
                <h3>Comparación de Memoria</h3>
                <div id="memory-comparison-chart"></div>
            </div>
            
            <div class="chart-container">
                <h3>Nodos Explorados vs Tiempo</h3>
                <div id="nodes-chart"></div>
            </div>
        </section>
    </main>
    
    <footer>
        <p>Generado por LatticeWeaver v5.0 - {generation_date}</p>
    </footer>
    
    <script>
        // Insertar gráficos Plotly
        var timeChart = {time_chart_json};
        Plotly.newPlot('time-comparison-chart', timeChart.data, timeChart.layout);
        
        var memoryChart = {memory_chart_json};
        Plotly.newPlot('memory-comparison-chart', memoryChart.data, memoryChart.layout);
        
        var nodesChart = {nodes_chart_json};
        Plotly.newPlot('nodes-chart', nodesChart.data, nodesChart.layout);
    </script>
</body>
</html>
        """
        
        return html
    
    def _generate_comparison_html(
        self,
        results: Dict[str, Dict[str, BenchmarkResult]],
        baseline: str,
        speedup_heatmap_json: str
    ) -> str:
        """Genera HTML del reporte de comparación."""
        generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparison Report - LatticeWeaver</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <header>
        <h1>🔬 LatticeWeaver Benchmarking</h1>
        <nav>
            <a href="benchmark_report.html">Benchmarks</a>
            <a href="comparison_report.html">Comparación</a>
            <a href="scalability_report.html">Escalabilidad</a>
        </nav>
    </header>
    
    <main>
        <section class="summary">
            <h2>⚖️ Comparación de Algoritmos</h2>
            <p>Baseline: <strong>{baseline}</strong></p>
        </section>
        
        <section class="visualizations">
            <div class="chart-container">
                <h3>Heatmap de Speedups</h3>
                <div id="speedup-heatmap"></div>
            </div>
        </section>
    </main>
    
    <footer>
        <p>Generado por LatticeWeaver v5.0 - {generation_date}</p>
    </footer>
    
    <script>
        var heatmap = {speedup_heatmap_json};
        Plotly.newPlot('speedup-heatmap', heatmap.data, heatmap.layout);
    </script>
</body>
</html>
        """
        
        return html
    
    def _generate_scalability_html(
        self,
        results: Dict[int, Dict[str, BenchmarkResult]],
        scalability_chart_json: str
    ) -> str:
        """Genera HTML del reporte de escalabilidad."""
        generation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scalability Report - LatticeWeaver</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <header>
        <h1>🔬 LatticeWeaver Benchmarking</h1>
        <nav>
            <a href="benchmark_report.html">Benchmarks</a>
            <a href="comparison_report.html">Comparación</a>
            <a href="scalability_report.html">Escalabilidad</a>
        </nav>
    </header>
    
    <main>
        <section class="summary">
            <h2>📈 Análisis de Escalabilidad</h2>
        </section>
        
        <section class="visualizations">
            <div class="chart-container">
                <h3>Tiempo vs Tamaño del Problema</h3>
                <div id="scalability-chart"></div>
            </div>
        </section>
    </main>
    
    <footer>
        <p>Generado por LatticeWeaver v5.0 - {generation_date}</p>
    </footer>
    
    <script>
        var chart = {scalability_chart_json};
        Plotly.newPlot('scalability-chart', chart.data, chart.layout);
    </script>
</body>
</html>
        """
        
        return html
    
    def _get_css(self) -> str:
        """Retorna CSS para los reportes."""
        return """
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --bg-color: #ecf0f1;
            --text-color: #2c3e50;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }
        
        header {
            background-color: var(--primary-color);
            color: white;
            padding: 1.5rem 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        header h1 {
            margin: 0;
            font-size: 1.8rem;
        }
        
        nav {
            margin-top: 0.5rem;
        }
        
        nav a {
            color: white;
            text-decoration: none;
            margin-right: 1.5rem;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        
        nav a:hover {
            background-color: var(--secondary-color);
        }
        
        main {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 2rem;
        }
        
        .summary {
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .summary h2 {
            color: var(--primary-color);
            margin-bottom: 1rem;
        }
        
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }
        
        .metric {
            text-align: center;
            padding: 1.5rem;
            background: var(--bg-color);
            border-radius: 8px;
            transition: transform 0.3s;
        }
        
        .metric:hover {
            transform: translateY(-5px);
        }
        
        .metric h3 {
            font-size: 2.5rem;
            margin: 0;
            color: var(--secondary-color);
        }
        
        .metric p {
            margin: 0.5rem 0 0 0;
            color: #7f8c8d;
            font-size: 0.9rem;
        }
        
        .results {
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .results h2 {
            color: var(--primary-color);
            margin-bottom: 1.5rem;
        }
        
        .results-table {
            width: 100%;
            border-collapse: collapse;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .results-table thead {
            background-color: var(--primary-color);
            color: white;
        }
        
        .results-table th,
        .results-table td {
            padding: 1rem;
            text-align: left;
        }
        
        .results-table tbody tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        .results-table tbody tr:hover {
            background-color: #e8f4f8;
        }
        
        .visualizations {
            margin-top: 2rem;
        }
        
        .visualizations h2 {
            color: var(--primary-color);
            margin-bottom: 1.5rem;
        }
        
        .chart-container {
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .chart-container h3 {
            margin-top: 0;
            color: var(--primary-color);
            margin-bottom: 1rem;
        }
        
        footer {
            text-align: center;
            padding: 2rem;
            color: #7f8c8d;
            margin-top: 4rem;
        }
        """


# ============================================================================
# Funciones de Utilidad
# ============================================================================

def quick_report(
    results: Dict[str, BenchmarkResult],
    output_dir: str = "reports"
) -> Path:
    """
    Genera reporte rápido con configuración por defecto.
    
    Args:
        results: Diccionario de resultados
        output_dir: Directorio de salida
    
    Returns:
        Path al reporte generado
    
    Example:
        >>> results = {...}
        >>> report_path = quick_report(results)
        >>> print(f"Reporte: {report_path}")
    """
    generator = ReportGenerator(output_dir=output_dir)
    return generator.generate_benchmark_report(results)

