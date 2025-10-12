"""
Script de prueba para la API REST del visualizador.

Este script demuestra cómo interactuar con la API desde un cliente Python.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

import requests
import json


def test_api():
    """Prueba los endpoints de la API."""
    base_url = "http://localhost:5000"
    
    # Ruta del trace de ejemplo
    trace_path = "/home/ubuntu/lattice-weaver/examples/nqueens_6_trace.csv"
    
    print("=== Prueba de API REST del Visualizador ===")
    print()
    
    # 1. Health check
    print("1. Health check...")
    response = requests.get(f"{base_url}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    print()
    
    # 2. Obtener estadísticas
    print("2. Obtener estadísticas...")
    response = requests.post(
        f"{base_url}/api/v1/statistics",
        json={"trace_path": trace_path}
    )
    print(f"   Status: {response.status_code}")
    stats = response.json()['statistics']
    print(f"   Nodos explorados: {stats['nodes_explored']}")
    print(f"   Backtracks: {stats['backtracks']}")
    print(f"   Duración: {stats['duration']:.4f}s")
    print()
    
    # 3. Generar visualización del árbol
    print("3. Generar árbol de búsqueda...")
    response = requests.post(
        f"{base_url}/api/v1/visualize/tree",
        json={"trace_path": trace_path, "max_nodes": 100}
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   ✓ Figura generada correctamente")
    print()
    
    # 4. Generar timeline
    print("4. Generar timeline...")
    response = requests.post(
        f"{base_url}/api/v1/visualize/timeline",
        json={"trace_path": trace_path}
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   ✓ Timeline generado correctamente")
    print()
    
    # 5. Descargar reporte
    print("5. Generar y descargar reporte...")
    response = requests.post(
        f"{base_url}/api/v1/report",
        json={
            "trace_path": trace_path,
            "title": "Reporte de Prueba API",
            "advanced": True
        }
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        with open("/home/ubuntu/lattice-weaver/examples/api_report.html", "wb") as f:
            f.write(response.content)
        print(f"   ✓ Reporte guardado en api_report.html")
    print()
    
    print("=== Prueba completada ===")


if __name__ == '__main__':
    print("NOTA: Este script requiere que el servidor API esté ejecutándose.")
    print("Ejecutar primero: python -m lattice_weaver.visualization.api")
    print()
    
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("ERROR: No se pudo conectar al servidor API.")
        print("Asegúrate de que el servidor esté ejecutándose en http://localhost:5000")

