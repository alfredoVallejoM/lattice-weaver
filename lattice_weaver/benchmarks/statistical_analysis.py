"""
Análisis estadístico de resultados de benchmarks.

Este módulo proporciona herramientas para analizar los resultados de los benchmarks
del compilador multiescala, generando estadísticas, visualizaciones y reportes.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict


class BenchmarkAnalyzer:
    """
    Analizador de resultados de benchmarks.
    
    Esta clase procesa los archivos JSON de resultados de benchmarks y genera
    análisis estadísticos, visualizaciones y reportes.
    """
    
    def __init__(self, results_dir: Path):
        """
        Inicializa el analizador con el directorio de resultados.
        
        Args:
            results_dir: Directorio que contiene los archivos JSON de resultados.
        """
        self.results_dir = Path(results_dir)
        self.results = []
        self.df = None
    
    def load_results(self, pattern: str = "comprehensive_results_*.json"):
        """
        Carga todos los archivos de resultados que coincidan con el patrón.
        
        Args:
            pattern: Patrón glob para buscar archivos de resultados.
        """
        result_files = sorted(self.results_dir.glob(pattern))
        
        if not result_files:
            raise FileNotFoundError(f"No se encontraron archivos de resultados en {self.results_dir}")
        
        # Cargar el archivo más reciente
        latest_file = result_files[-1]
        print(f"Cargando resultados desde: {latest_file}")
        
        with open(latest_file, 'r') as f:
            self.results = json.load(f)
        
        # Convertir a DataFrame para análisis
        self.df = pd.DataFrame(self.results)
        
        # Limpiar datos: eliminar filas con errores
        if 'error' in self.df.columns:
            self.df = self.df[self.df['error'].isna()]
        
        print(f"Cargados {len(self.df)} resultados válidos")
    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Calcula estadísticas descriptivas de los resultados.
        
        Returns:
            Diccionario con estadísticas por estrategia y tipo de problema.
        """
        stats = {}
        
        # Estadísticas por estrategia
        stats['by_strategy'] = {}
        for strategy in self.df['strategy'].unique():
            strategy_df = self.df[self.df['strategy'] == strategy]
            stats['by_strategy'][strategy] = {
                'mean_total_time': strategy_df['total_time'].mean(),
                'std_total_time': strategy_df['total_time'].std(),
                'mean_compilation_time': strategy_df['compilation_time'].mean(),
                'mean_solving_time': strategy_df['solving_time'].mean(),
                'mean_memory': strategy_df['peak_memory_mb'].mean(),
                'success_rate': strategy_df['solution_found'].mean(),
                'mean_compression_ratio': strategy_df['compression_ratio'].mean()
            }
        
        # Estadísticas por tipo de problema
        stats['by_problem'] = {}
        for problem_type in self.df['problem_type'].unique():
            problem_df = self.df[self.df['problem_type'] == problem_type]
            stats['by_problem'][problem_type] = {
                'mean_total_time': problem_df['total_time'].mean(),
                'success_rate': problem_df['solution_found'].mean(),
                'num_benchmarks': len(problem_df)
            }
        
        # Comparación con baseline (NoCompilation)
        if 'NoCompilation' in self.df['strategy'].values:
            baseline_df = self.df[self.df['strategy'] == 'NoCompilation']
            stats['compilation_overhead'] = {}
            
            for strategy in self.df['strategy'].unique():
                if strategy == 'NoCompilation':
                    continue
                
                strategy_df = self.df[self.df['strategy'] == strategy]
                
                # Calcular mejora promedio vs baseline
                improvements = []
                for problem_type in self.df['problem_type'].unique():
                    baseline_time = baseline_df[baseline_df['problem_type'] == problem_type]['total_time'].mean()
                    strategy_time = strategy_df[strategy_df['problem_type'] == problem_type]['total_time'].mean()
                    
                    if not np.isnan(baseline_time) and not np.isnan(strategy_time) and baseline_time > 0:
                        improvement = ((baseline_time - strategy_time) / baseline_time) * 100
                        improvements.append(improvement)
                
                if improvements:
                    stats['compilation_overhead'][strategy] = {
                        'mean_improvement': np.mean(improvements),
                        'median_improvement': np.median(improvements),
                        'std_improvement': np.std(improvements)
                    }
        
        return stats
    
    def plot_performance_comparison(self, output_path: Optional[Path] = None):
        """
        Genera un gráfico de comparación de rendimiento entre estrategias.
        
        Args:
            output_path: Ruta donde guardar el gráfico. Si es None, se muestra en pantalla.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Tiempo total por estrategia
        ax1 = axes[0, 0]
        strategy_times = self.df.groupby('strategy')['total_time'].mean().sort_values()
        strategy_times.plot(kind='barh', ax=ax1, color='skyblue')
        ax1.set_xlabel('Tiempo Total Promedio (s)')
        ax1.set_title('Tiempo Total por Estrategia')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Tasa de éxito por estrategia
        ax2 = axes[0, 1]
        success_rates = self.df.groupby('strategy')['solution_found'].mean() * 100
        success_rates.plot(kind='barh', ax=ax2, color='lightgreen')
        ax2.set_xlabel('Tasa de Éxito (%)')
        ax2.set_title('Tasa de Éxito por Estrategia')
        ax2.set_xlim(0, 100)
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Tiempo de compilación vs tiempo de resolución
        ax3 = axes[1, 0]
        compilation_vs_solving = self.df.groupby('strategy')[['compilation_time', 'solving_time']].mean()
        compilation_vs_solving.plot(kind='bar', ax=ax3, stacked=True)
        ax3.set_xlabel('Estrategia')
        ax3.set_ylabel('Tiempo (s)')
        ax3.set_title('Tiempo de Compilación vs Resolución')
        ax3.legend(['Compilación', 'Resolución'])
        ax3.grid(axis='y', alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Uso de memoria por estrategia
        ax4 = axes[1, 1]
        memory_usage = self.df.groupby('strategy')['peak_memory_mb'].mean()
        memory_usage.plot(kind='barh', ax=ax4, color='salmon')
        ax4.set_xlabel('Memoria Pico Promedio (MB)')
        ax4.set_title('Uso de Memoria por Estrategia')
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_scalability_analysis(self, output_path: Optional[Path] = None):
        """
        Genera un gráfico de análisis de escalabilidad por tamaño de problema.
        
        Args:
            output_path: Ruta donde guardar el gráfico. Si es None, se muestra en pantalla.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Escalabilidad por tipo de problema
        ax1 = axes[0]
        for problem_type in self.df['problem_type'].unique():
            problem_df = self.df[self.df['problem_type'] == problem_type]
            if 'problem_size' in problem_df.columns:
                scalability = problem_df.groupby('problem_size')['total_time'].mean()
                ax1.plot(scalability.index, scalability.values, marker='o', label=problem_type)
        
        ax1.set_xlabel('Tamaño del Problema')
        ax1.set_ylabel('Tiempo Total Promedio (s)')
        ax1.set_title('Escalabilidad por Tipo de Problema')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Escalabilidad por estrategia (solo N-Queens)
        ax2 = axes[1]
        nqueens_df = self.df[self.df['problem_type'] == 'N-Queens']
        if not nqueens_df.empty and 'problem_size' in nqueens_df.columns:
            for strategy in nqueens_df['strategy'].unique():
                strategy_df = nqueens_df[nqueens_df['strategy'] == strategy]
                scalability = strategy_df.groupby('problem_size')['total_time'].mean()
                ax2.plot(scalability.index, scalability.values, marker='o', label=strategy)
            
            ax2.set_xlabel('Tamaño del Problema (N)')
            ax2.set_ylabel('Tiempo Total Promedio (s)')
            ax2.set_title('Escalabilidad de N-Queens por Estrategia')
            ax2.legend()
            ax2.grid(alpha=0.3)
            ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_compilation_overhead(self, output_path: Optional[Path] = None):
        """
        Genera un gráfico de análisis de sobrecarga de compilación.
        
        Args:
            output_path: Ruta donde guardar el gráfico. Si es None, se muestra en pantalla.
        """
        # Calcular mejora vs baseline para cada estrategia y problema
        baseline_df = self.df[self.df['strategy'] == 'NoCompilation']
        
        if baseline_df.empty:
            print("No se encontró estrategia baseline (NoCompilation)")
            return
        
        improvements = []
        for problem_type in self.df['problem_type'].unique():
            baseline_time = baseline_df[baseline_df['problem_type'] == problem_type]['total_time'].mean()
            
            for strategy in self.df['strategy'].unique():
                if strategy == 'NoCompilation':
                    continue
                
                strategy_df = self.df[(self.df['strategy'] == strategy) & (self.df['problem_type'] == problem_type)]
                strategy_time = strategy_df['total_time'].mean()
                
                if not np.isnan(baseline_time) and not np.isnan(strategy_time) and baseline_time > 0:
                    improvement = ((baseline_time - strategy_time) / baseline_time) * 100
                    improvements.append({
                        'strategy': strategy,
                        'problem_type': problem_type,
                        'improvement': improvement
                    })
        
        if not improvements:
            print("No se pudieron calcular mejoras")
            return
        
        improvements_df = pd.DataFrame(improvements)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Crear gráfico de barras agrupadas
        pivot_df = improvements_df.pivot(index='problem_type', columns='strategy', values='improvement')
        pivot_df.plot(kind='bar', ax=ax)
        
        ax.set_xlabel('Tipo de Problema')
        ax.set_ylabel('Mejora vs Baseline (%)')
        ax.set_title('Mejora de Rendimiento vs Estrategia Sin Compilación')
        ax.axhline(y=0, color='r', linestyle='--', label='Baseline')
        ax.legend(title='Estrategia', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, output_path: Path):
        """
        Genera un reporte completo de análisis de benchmarks en formato Markdown.
        
        Args:
            output_path: Ruta donde guardar el reporte.
        """
        stats = self.compute_statistics()
        
        report = []
        report.append("# Reporte de Análisis de Benchmarks del Compilador Multiescala\n")
        report.append(f"**Fecha de Generación**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Número Total de Benchmarks**: {len(self.df)}\n")
        report.append("\n---\n\n")
        
        # Resumen ejecutivo
        report.append("## Resumen Ejecutivo\n\n")
        report.append(f"Se ejecutaron **{len(self.df)}** benchmarks en total, evaluando **{len(self.df['strategy'].unique())}** estrategias de compilación ")
        report.append(f"en **{len(self.df['problem_type'].unique())}** tipos de problemas CSP diferentes.\n\n")
        
        # Estadísticas por estrategia
        report.append("## Estadísticas por Estrategia\n\n")
        report.append("| Estrategia | Tiempo Total (s) | Tiempo Compilación (s) | Tiempo Resolución (s) | Memoria (MB) | Tasa de Éxito (%) | Ratio Compresión |\n")
        report.append("|------------|------------------|------------------------|----------------------|--------------|-------------------|------------------|\n")
        
        for strategy, stats_data in stats['by_strategy'].items():
            report.append(f"| {strategy} | {stats_data['mean_total_time']:.4f} ± {stats_data['std_total_time']:.4f} | ")
            report.append(f"{stats_data['mean_compilation_time']:.4f} | {stats_data['mean_solving_time']:.4f} | ")
            report.append(f"{stats_data['mean_memory']:.2f} | {stats_data['success_rate']*100:.1f} | ")
            report.append(f"{stats_data['mean_compression_ratio']:.2f}x |\n")
        
        report.append("\n")
        
        # Análisis de sobrecarga de compilación
        if 'compilation_overhead' in stats:
            report.append("## Análisis de Sobrecarga de Compilación\n\n")
            report.append("Mejora promedio de cada estrategia de compilación vs estrategia sin compilación (NoCompilation):\n\n")
            report.append("| Estrategia | Mejora Promedio (%) | Mejora Mediana (%) | Desviación Estándar |\n")
            report.append("|------------|---------------------|--------------------|--------------------||\n")
            
            for strategy, overhead_data in stats['compilation_overhead'].items():
                report.append(f"| {strategy} | {overhead_data['mean_improvement']:+.2f} | ")
                report.append(f"{overhead_data['median_improvement']:+.2f} | {overhead_data['std_improvement']:.2f} |\n")
            
            report.append("\n**Nota**: Valores negativos indican que la estrategia es más lenta que el baseline.\n\n")
        
        # Estadísticas por tipo de problema
        report.append("## Estadísticas por Tipo de Problema\n\n")
        report.append("| Tipo de Problema | Tiempo Total (s) | Tasa de Éxito (%) | Número de Benchmarks |\n")
        report.append("|------------------|------------------|-------------------|---------------------|\n")
        
        for problem_type, stats_data in stats['by_problem'].items():
            report.append(f"| {problem_type} | {stats_data['mean_total_time']:.4f} | ")
            report.append(f"{stats_data['success_rate']*100:.1f} | {stats_data['num_benchmarks']} |\n")
        
        report.append("\n")
        
        # Observaciones y recomendaciones
        report.append("## Observaciones y Recomendaciones\n\n")
        
        # Identificar problemas de rendimiento
        if 'compilation_overhead' in stats:
            negative_improvements = [s for s, d in stats['compilation_overhead'].items() if d['mean_improvement'] < 0]
            if negative_improvements:
                report.append("### Rendimiento Negativo de la Compilación\n\n")
                report.append(f"Las siguientes estrategias muestran un rendimiento **peor** que la estrategia sin compilación:\n\n")
                for strategy in negative_improvements:
                    improvement = stats['compilation_overhead'][strategy]['mean_improvement']
                    report.append(f"- **{strategy}**: {improvement:+.2f}% (más lento que el baseline)\n")
                report.append("\n**Recomendación**: Investigar la sobrecarga de compilación en estos niveles y optimizar el proceso de compilación.\n\n")
        
        # Identificar problemas de resolución
        low_success_strategies = [(s, d) for s, d in stats['by_strategy'].items() if d['success_rate'] < 0.9]
        if low_success_strategies:
            report.append("### Baja Tasa de Éxito\n\n")
            report.append("Las siguientes estrategias tienen una tasa de éxito inferior al 90%:\n\n")
            for strategy, stats_data in low_success_strategies:
                report.append(f"- **{strategy}**: {stats_data['success_rate']*100:.1f}%\n")
            report.append("\n**Recomendación**: Investigar los problemas que no se resuelven y ajustar el solucionador o los generadores de problemas.\n\n")
        
        # Guardar reporte
        with open(output_path, 'w') as f:
            f.write(''.join(report))
        
        print(f"Reporte guardado en: {output_path}")


def main():
    """
    Función principal para ejecutar el análisis estadístico.
    """
    # Directorio de resultados
    results_dir = Path("/home/ubuntu/benchmark_results")
    
    # Crear analizador
    analyzer = BenchmarkAnalyzer(results_dir)
    
    # Cargar resultados
    analyzer.load_results()
    
    # Generar visualizaciones
    output_dir = Path("/home/ubuntu/lattice-weaver-repo/benchmark_analysis")
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerando visualizaciones...")
    analyzer.plot_performance_comparison(output_dir / "performance_comparison.png")
    analyzer.plot_scalability_analysis(output_dir / "scalability_analysis.png")
    analyzer.plot_compilation_overhead(output_dir / "compilation_overhead.png")
    
    # Generar reporte
    print("\nGenerando reporte...")
    analyzer.generate_report(output_dir / "benchmark_report.md")
    
    # Guardar estadísticas en JSON
    stats = analyzer.compute_statistics()
    with open(output_dir / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"\nAnálisis completo. Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    main()

