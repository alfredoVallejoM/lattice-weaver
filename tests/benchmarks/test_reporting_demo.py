"""
Test de Demostración del Sistema de Reportes

Genera reportes HTML de ejemplo con visualizaciones interactivas.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

import pytest
from pathlib import Path

from tests.benchmarks.visualizations import BenchmarkResult, BenchmarkVisualizer
from tests.benchmarks.report_generator import ReportGenerator


@pytest.mark.integration
def test_generate_sample_report(tmp_path):
    """
    Test: Generar reporte HTML de ejemplo.
    
    Crea un reporte completo con datos de ejemplo para demostración.
    """
    # Datos de ejemplo (N-Reinas n=4, n=6, n=8)
    results = {
        "BT_n4": BenchmarkResult(
            problem_name="N-Reinas n=4",
            algorithm_name="Backtracking",
            time_ms=0.09,
            memory_mb=2.5,
            nodes_explored=9,
            backtracks=4,
            success=True,
            problem_size=4
        ),
        "FC_n4": BenchmarkResult(
            problem_name="N-Reinas n=4",
            algorithm_name="Forward Checking",
            time_ms=0.24,
            memory_mb=2.8,
            nodes_explored=7,
            backtracks=2,
            success=True,
            problem_size=4
        ),
        "BT_n6": BenchmarkResult(
            problem_name="N-Reinas n=6",
            algorithm_name="Backtracking",
            time_ms=0.48,
            memory_mb=3.2,
            nodes_explored=32,
            backtracks=25,
            success=True,
            problem_size=6
        ),
        "FC_n6": BenchmarkResult(
            problem_name="N-Reinas n=6",
            algorithm_name="Forward Checking",
            time_ms=1.16,
            memory_mb=3.5,
            nodes_explored=20,
            backtracks=13,
            success=True,
            problem_size=6
        ),
        "BT_n8": BenchmarkResult(
            problem_name="N-Reinas n=8",
            algorithm_name="Backtracking",
            time_ms=3.09,
            memory_mb=4.1,
            nodes_explored=114,
            backtracks=105,
            success=True,
            problem_size=8
        ),
        "FC_n8": BenchmarkResult(
            problem_name="N-Reinas n=8",
            algorithm_name="Forward Checking",
            time_ms=4.62,
            memory_mb=4.5,
            nodes_explored=53,
            backtracks=44,
            success=True,
            problem_size=8
        ),
    }
    
    # Generar reporte
    generator = ReportGenerator(output_dir=str(tmp_path))
    report_path = generator.generate_benchmark_report(
        results,
        title="N-Reinas Benchmark Report"
    )
    
    # Verificar que se generó
    assert report_path.exists()
    assert report_path.suffix == ".html"
    
    # Verificar contenido
    content = report_path.read_text()
    assert "LatticeWeaver" in content
    assert "Benchmark Report" in content
    assert "Backtracking" in content
    assert "Forward Checking" in content
    
    print(f"\n✅ Reporte generado: {report_path}")
    print(f"   Tamaño: {report_path.stat().st_size / 1024:.1f} KB")
    
    return report_path


@pytest.mark.integration
def test_generate_comparison_report(tmp_path):
    """
    Test: Generar reporte de comparación.
    
    Crea un reporte con heatmap de speedups.
    """
    # Datos anidados por problema y algoritmo
    results = {
        "nqueens_4": {
            "Backtracking": BenchmarkResult(
                "N-Reinas n=4", "Backtracking",
                0.09, 2.5, 9, 4, True, 4
            ),
            "Forward Checking": BenchmarkResult(
                "N-Reinas n=4", "Forward Checking",
                0.24, 2.8, 7, 2, True, 4
            ),
        },
        "nqueens_6": {
            "Backtracking": BenchmarkResult(
                "N-Reinas n=6", "Backtracking",
                0.48, 3.2, 32, 25, True, 6
            ),
            "Forward Checking": BenchmarkResult(
                "N-Reinas n=6", "Forward Checking",
                1.16, 3.5, 20, 13, True, 6
            ),
        },
        "nqueens_8": {
            "Backtracking": BenchmarkResult(
                "N-Reinas n=8", "Backtracking",
                3.09, 4.1, 114, 105, True, 8
            ),
            "Forward Checking": BenchmarkResult(
                "N-Reinas n=8", "Forward Checking",
                4.62, 4.5, 53, 44, True, 8
            ),
        },
    }
    
    # Generar reporte de comparación
    generator = ReportGenerator(output_dir=str(tmp_path))
    report_path = generator.generate_comparison_report(
        results,
        baseline="Backtracking"
    )
    
    # Verificar
    assert report_path.exists()
    
    content = report_path.read_text()
    assert "Comparación" in content
    assert "Speedup" in content
    
    print(f"\n✅ Reporte de comparación generado: {report_path}")
    
    return report_path


@pytest.mark.integration
def test_generate_scalability_report(tmp_path):
    """
    Test: Generar reporte de escalabilidad.
    
    Crea un reporte con análisis de escalabilidad.
    """
    # Datos por tamaño
    results = {
        4: {
            "Backtracking": BenchmarkResult(
                "N-Reinas n=4", "Backtracking",
                0.09, 2.5, 9, 4, True, 4
            ),
            "Forward Checking": BenchmarkResult(
                "N-Reinas n=4", "Forward Checking",
                0.24, 2.8, 7, 2, True, 4
            ),
        },
        6: {
            "Backtracking": BenchmarkResult(
                "N-Reinas n=6", "Backtracking",
                0.48, 3.2, 32, 25, True, 6
            ),
            "Forward Checking": BenchmarkResult(
                "N-Reinas n=6", "Forward Checking",
                1.16, 3.5, 20, 13, True, 6
            ),
        },
        8: {
            "Backtracking": BenchmarkResult(
                "N-Reinas n=8", "Backtracking",
                3.09, 4.1, 114, 105, True, 8
            ),
            "Forward Checking": BenchmarkResult(
                "N-Reinas n=8", "Forward Checking",
                4.62, 4.5, 53, 44, True, 8
            ),
        },
    }
    
    # Generar reporte de escalabilidad
    generator = ReportGenerator(output_dir=str(tmp_path))
    report_path = generator.generate_scalability_report(results)
    
    # Verificar
    assert report_path.exists()
    
    content = report_path.read_text()
    assert "Escalabilidad" in content
    
    print(f"\n✅ Reporte de escalabilidad generado: {report_path}")
    
    return report_path


@pytest.mark.integration
def test_generate_all_reports():
    """
    Test: Generar todos los reportes en directorio reports/.
    
    Este test genera reportes reales que se pueden visualizar.
    """
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    
    # Datos de ejemplo
    results_flat = {
        "BT_n4": BenchmarkResult("N-Reinas n=4", "Backtracking", 0.09, 2.5, 9, 4, True, 4),
        "FC_n4": BenchmarkResult("N-Reinas n=4", "Forward Checking", 0.24, 2.8, 7, 2, True, 4),
        "BT_n6": BenchmarkResult("N-Reinas n=6", "Backtracking", 0.48, 3.2, 32, 25, True, 6),
        "FC_n6": BenchmarkResult("N-Reinas n=6", "Forward Checking", 1.16, 3.5, 20, 13, True, 6),
        "BT_n8": BenchmarkResult("N-Reinas n=8", "Backtracking", 3.09, 4.1, 114, 105, True, 8),
        "FC_n8": BenchmarkResult("N-Reinas n=8", "Forward Checking", 4.62, 4.5, 53, 44, True, 8),
    }
    
    results_nested = {
        "nqueens_4": {
            "Backtracking": results_flat["BT_n4"],
            "Forward Checking": results_flat["FC_n4"],
        },
        "nqueens_6": {
            "Backtracking": results_flat["BT_n6"],
            "Forward Checking": results_flat["FC_n6"],
        },
        "nqueens_8": {
            "Backtracking": results_flat["BT_n8"],
            "Forward Checking": results_flat["FC_n8"],
        },
    }
    
    results_by_size = {
        4: {
            "Backtracking": results_flat["BT_n4"],
            "Forward Checking": results_flat["FC_n4"],
        },
        6: {
            "Backtracking": results_flat["BT_n6"],
            "Forward Checking": results_flat["FC_n6"],
        },
        8: {
            "Backtracking": results_flat["BT_n8"],
            "Forward Checking": results_flat["FC_n8"],
        },
    }
    
    # Generar todos los reportes
    generator = ReportGenerator(output_dir=str(output_dir))
    
    report1 = generator.generate_benchmark_report(results_flat)
    report2 = generator.generate_comparison_report(results_nested)
    report3 = generator.generate_scalability_report(results_by_size)
    
    print("\n" + "="*60)
    print("✅ REPORTES GENERADOS")
    print("="*60)
    print(f"\n📄 Benchmark Report: {report1}")
    print(f"📄 Comparison Report: {report2}")
    print(f"📄 Scalability Report: {report3}")
    print(f"\n💡 Abre los archivos HTML en tu navegador para verlos")
    print("="*60)
    
    assert report1.exists()
    assert report2.exists()
    assert report3.exists()


if __name__ == "__main__":
    # Ejecutar generación de reportes
    test_generate_all_reports()

