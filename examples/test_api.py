
import requests
import json
import pytest
import subprocess
import time
import os
import tempfile

@pytest.fixture(scope="module")
def flask_server():
    """Inicia y detiene el servidor Flask para los tests de la API."""
    server_process = None
    log_file_path = None
    try:
        # Crear un archivo temporal para capturar la salida del servidor Flask
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as log_file_obj:
            log_file_path = log_file_obj.name

        # Iniciar el servidor Flask en un proceso separado, redirigiendo stdout y stderr al archivo temporal
        server_command = [
            "python", "-m", "lattice_weaver.visualization.api"
        ]
        with open(log_file_path, 'w') as f:
            server_process = subprocess.Popen(server_command, cwd="/home/ubuntu/lattice-weaver",
                                              stdout=f,
                                              stderr=f)
        
        # Esperar a que el servidor se inicie.
        time.sleep(5) # Dar tiempo al servidor para arrancar
        
        # Verificar que el servidor está escuchando
        base_url = "http://localhost:5000"
        for _ in range(10): # Intentar varias veces
            try:
                response = requests.get(f"{base_url}/health", timeout=1)
                if response.status_code == 200:
                    print("\nServidor Flask iniciado y accesible.")
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            # Si el servidor no se inició, leer el log para depurar
            with open(log_file_path, 'r') as f:
                server_output = f.read()
            raise RuntimeError(f"No se pudo iniciar el servidor Flask para los tests. Log:\n{server_output}")

        yield base_url

    finally:
        if server_process:
            print("\nDeteniendo servidor Flask...")
            server_process.terminate()
            server_process.wait(timeout=5) # Esperar a que termine
            if server_process.poll() is None: # Si aún no ha terminado
                server_process.kill()
            print("Servidor Flask detenido.")
        if log_file_path and os.path.exists(log_file_path):
            # Imprimir el log del servidor si hubo un error en el test
            # Para simplificar, siempre imprimiremos el log si el test falla.
            with open(log_file_path, 'r') as f:
                server_output = f.read()
            print(f"\n--- Log del Servidor Flask ---\n{server_output}\n------------------------------")
            os.remove(log_file_path)

def test_api(flask_server):
    """Prueba los endpoints de la API."""
    base_url = flask_server
    
    # Ruta del trace de ejemplo
    trace_path = os.path.join(os.path.dirname(__file__), "nqueens_4_trace.csv")
    
    print("\n=== Prueba de API REST del Visualizador ===")
    
    # 1. Health check
    print("1. Health check...")
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # 2. Obtener estadísticas
    print("2. Obtener estadísticas...")
    response = requests.post(
        f"{base_url}/api/v1/statistics",
        json={
            "trace_path": trace_path,
            "output_dir": os.path.dirname(__file__)
        }
    )
    assert response.status_code == 200
    stats = response.json()["statistics"]
    assert "nodes_explored" in stats
    assert "backtracks" in stats
    assert "duration" in stats
    print(f"   Status: {response.status_code}")
    print(f"   Nodos explorados: {stats['nodes_explored']}")
    print(f"   Backtracks: {stats['backtracks']}")
    print(f"   Duración: {stats['duration']:.4f}s")
    
    # 3. Generar visualización del árbol
    print("3. Generar árbol de búsqueda...")
    response = requests.post(
        f"{base_url}/api/v1/visualize/tree",
        json={
            "trace_path": trace_path,
            "max_nodes": 100,
            "output_dir": os.path.dirname(__file__)
        }
    )
    assert response.status_code == 200
    assert "figure" in response.json()
    print(f"   Status: {response.status_code}")
    print(f"   ✓ Figura generada correctamente (JSON)")
    
    # 4. Generar timeline
    print("4. Generar timeline...")
    response = requests.post(
        f"{base_url}/api/v1/visualize/timeline",
        json={
            "trace_path": trace_path,
            "output_dir": os.path.dirname(__file__)
        }
    )
    assert response.status_code == 200
    assert "figure" in response.json()
    print(f"   Status: {response.status_code}")
    print(f"   ✓ Timeline generado correctamente (JSON)")
    
    # 5. Generar y descargar reporte
    print("5. Generar y descargar reporte...")
    response = requests.post(
        f"{base_url}/api/v1/report",
        json={
            "trace_path": trace_path,
            "title": "Reporte de Prueba API",
            "advanced": True,
            "output_dir": os.path.dirname(__file__)
        }
    )
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/html; charset=utf-8"
    assert "attachment" in response.headers["Content-Disposition"]
    print(f"   Status: {response.status_code}")
    print(f"   ✓ Reporte HTML generado y recibido correctamente")
    
    print("\n=== Prueba completada ===")


