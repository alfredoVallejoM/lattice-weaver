import time
import os
import shutil

from lattice_weaver.paging.page import Page
from lattice_weaver.paging.page_manager import PageManager
from lattice_weaver.validation.paging_validator import PagingValidator
from lattice_weaver.paging.paging_monitor import PagingMonitor

def run_example_monitor():
    print("--- Iniciando ejemplo de PagingMonitor ---")

    # 1. Configurar el directorio de almacenamiento L3
    l3_storage_dir = "./monitor_page_storage"
    if os.path.exists(l3_storage_dir):
        shutil.rmtree(l3_storage_dir)
    os.makedirs(l3_storage_dir, exist_ok=True)

    # 2. Inicializar PageManager y PagingValidator
    pm = PageManager(l1_capacity=10, l2_capacity=20, l3_capacity=50, l3_storage_dir=l3_storage_dir)
    validator = PagingValidator(pm)

    # 3. Inicializar PagingMonitor
    # Monitorear cada 2 segundos
    monitor = PagingMonitor(pm, validator, interval=2)

    # 4. Iniciar el monitoreo
    monitor.start_monitoring()

    # 5. Realizar algunas operaciones de paginación para generar estadísticas
    print("Realizando operaciones de paginación...")
    page_ids = [f"monitor_page_{i}" for i in range(100)]
    original_data_template = {"data": "monitor_test_data"}

    # Poner algunas páginas
    for i in range(20):
        pm.put_page(Page(page_ids[i], original_data_template, page_type="monitor_type", abstraction_level=1))

    # Generar aciertos
    for i in range(10):
        pm.get_page(page_ids[i])

    # Generar fallos (páginas que no están en caché o están en L3)
    for i in range(20, 40):
        pm.get_page(page_ids[i])

    # Esperar un poco para que el monitor registre algunas métricas
    print("Esperando para que el monitor registre métricas...")
    time.sleep(5) 

    # 6. Detener el monitoreo
    monitor.stop_monitoring()

    # 7. Mostrar los resultados del monitoreo
    print("--- Resultados del monitoreo ---")
    results = monitor.get_monitoring_results()
    if results:
        for i, res in enumerate(results):
            print(f"Resultado {i+1} (Timestamp: {res['timestamp']}):")
            for cache_name, stats in res['metrics']['cache_stats'].items():
                print(f"  {cache_name}: Hits={stats['hits']}, Misses={stats['misses']}, Hit Rate={stats['hit_rate']:.2f}, Size={stats['size']}/{stats['capacity']}")
    else:
        print("No se registraron resultados de monitoreo.")

    print("--- Ejemplo de PagingMonitor finalizado ---")

if __name__ == "__main__":
    # Añadir el directorio raíz del proyecto al PYTHONPATH
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "lattice-weaver")))
    run_example_monitor()

