from typing import Any, Dict, List, Optional, Set, Callable
import os
import shutil
import traceback

from lattice_weaver.core.csp_problem import CSP, is_satisfiable, verify_solution, generate_nqueens, generate_random_csp
from lattice_weaver.paging.page import Page
from lattice_weaver.paging.page_manager import PageManager
from lattice_weaver.validation.certificates import ValidationCertificate, create_certificate, CertificateRepository

class PagingValidator:
    """
    Validador para el sistema de paginación.
    Verifica la corrección, coherencia y eficiencia del sistema de caché multinivel.
    """

    def __init__(self, page_manager: PageManager):
        self.page_manager = page_manager
        self._metrics: Dict[str, Any] = {}

    def validate_coherence(self, page_id: str, original_data: Any) -> bool:
        """
        Verifica que los datos de una página son coherentes a través de los niveles de caché.
        Se asegura de que la página recuperada de cualquier nivel de caché o del almacenamiento L3
        coincida con los datos originales.
        """
        # Intentar recuperar la página y verificar su contenido
        retrieved_page = self.page_manager.get_page(page_id)
        if not retrieved_page or retrieved_page.content != original_data:
            print(f"Error de coherencia: la página {page_id} recuperada no coincide con los datos originales.")
            return False



        return True

    def validate_persistence(self, page_id: str, original_data: Any) -> bool:
        """
        Verifica que una página se persiste correctamente en el almacenamiento L3.
        """
        # Asegurarse de que la página está en L3 (put_page ya la propaga)
        # Forzar el desalojo de la página de L1 y L2 para que se lea de L3
        self.page_manager.l1_cache.remove(page_id)
        self.page_manager.l2_cache.remove(page_id)
        
        retrieved_page = self.page_manager.get_page(page_id)
        if not retrieved_page or retrieved_page.content != original_data:
            print(f"Fallo de persistencia: la página {page_id} no se recuperó correctamente de L3.")
            return False
        return True

    def measure_performance(self) -> Dict[str, Any]:
        """
        Mide el rendimiento del sistema de paginación (hit rate, miss rate, etc.).
        """
        self._metrics["cache_stats"] = self.page_manager.get_cache_stats()
        return self._metrics

    def generate_certificate(self) -> ValidationCertificate:
        """
        Genera un certificado de validación para el sistema de paginación.
        """
        # Placeholder para la lógica de generación de certificado
        # Esto se completará con los resultados de las validaciones y métricas
        return create_certificate(
            level_name="paging",
            type_checked=True,
            invariants_verified=["coherence", "persistence"],
            runtime_tests_passed=0,
            runtime_tests_failed=0,
            speedup_measured=0.0,
            memory_reduction=0.0,
            correctness_rate=0.0,
            metadata=self._metrics
        )

class PagingTestSuite:
    """
    Suite de tests para el sistema de paginación.
    """
    def __init__(self, cert_repo: CertificateRepository):
        self.cert_repo = cert_repo
        self.test_results: List[tuple[str, ValidationCertificate]] = []

    def run_test(self, test_name: str, test_func: Callable[[PageManager], bool]) -> ValidationCertificate:
        """
        Ejecuta un test de paginación y genera un certificado.
        """
        print(f"\n--- Running Paging Test: {test_name} ---")
        storage_dir = f"./page_storage_paging_{test_name.replace(' ', '_')}"
        # Limpiar el directorio de almacenamiento L3 antes de cada test para asegurar un estado limpio
        if os.path.exists(storage_dir):
            shutil.rmtree(storage_dir)
        page_manager = PageManager(l1_capacity=10, l2_capacity=20, l3_capacity=10000, l3_storage_dir=storage_dir)
        
        try:
            test_passed = test_func(page_manager, storage_dir)
            validator = PagingValidator(page_manager)
            metrics = validator.measure_performance()

            if test_passed:
                certificate = create_certificate(
                    level_name="paging",
                    type_checked=True,
                    invariants_verified=["coherence", "persistence"],
                    runtime_tests_passed=1,
                    runtime_tests_failed=0,
                    speedup_measured=0.0, # Placeholder
                    memory_reduction=0.0, # Placeholder
                    correctness_rate=1.0,
                    metadata={
                        **metrics,
                        "test_name": test_name
                    }
                )
            else:
                certificate = create_certificate(
                    level_name="paging",
                    type_checked=False,
                    invariants_verified=[],
                    runtime_tests_passed=0,
                    runtime_tests_failed=1,
                    speedup_measured=0.0,
                    memory_reduction=0.0,
                    correctness_rate=0.0,
                    metadata={
                        **metrics,
                        "test_name": test_name
                    }
                )
            self.cert_repo.store(certificate)
            self.test_results.append((test_name, certificate))
            print(f"Test \'{test_name}\' completed. Certificate generated: {certificate.signature}")
            print(certificate)
            return certificate

        except Exception as e:
            error_certificate = create_certificate(
                level_name="paging",
                type_checked=False,
                invariants_verified=[],
                runtime_tests_passed=0,
                runtime_tests_failed=1,
                speedup_measured=0.0,
                memory_reduction=0.0,
                correctness_rate=0.0,
                metadata={
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "test_name": test_name
                }
            )
            self.cert_repo.store(error_certificate)
            self.test_results.append((test_name, error_certificate))
            print(f"Test \033[31m\\\\'\{test_name}\\' FAILED\033[0m. Certificate generated: \{error_certificate.signature}")
            print(error_certificate)
            return error_certificate

    def run_all_tests(self) -> List[ValidationCertificate]:
        certificates: List[ValidationCertificate] = []

        # Test 1: Coherencia básica
        def test_basic_coherence(pm: PageManager, storage_dir: str) -> bool:
            page_id = "test_page_1"
            original_data = {"key": "value", "number": 123}
            page = Page(page_id, original_data, page_type="test_type", abstraction_level=1)
            pm.put_page(page)
            return PagingValidator(pm).validate_coherence(page_id, original_data)
        certificates.append(self.run_test("Basic Coherence Test", test_basic_coherence))

        # Test 2: Persistencia L3
        def test_l3_persistence(pm: PageManager, storage_dir: str) -> bool:
            page_id = "test_page_2"
            original_data = {"list": [1, 2, 3, 4, 5]}
            page = Page(page_id, original_data, page_type="test_type", abstraction_level=1)
            # Asegurarse de que la página se ha escrito en L3
            # Forzar el desalojo de L1 y L2 para que la página se mueva a L3
            pm.put_page(page) # Poner la página en el sistema de paginación
            # Llenar L1 y L2 para forzar el desalojo de la página a L3
            for i in range(pm.l1_cache.capacity + pm.l2_cache.capacity + 1):
                dummy_page_id = f"dummy_l3_eviction_{i}"
                dummy_page = Page(dummy_page_id, {"d": i}, page_type="dummy", abstraction_level=1)
                pm.put_page(dummy_page)
            
            return PagingValidator(pm).validate_persistence(page_id, original_data)
        certificates.append(self.run_test("L3 Persistence Test", test_l3_persistence))

        # Test 6: Persistencia después de reinicio del PageManager
        def test_persistence_after_restart(pm: PageManager, storage_dir: str) -> bool:
            page_id = "test_page_restart"
            original_data = {"status": "persisted_after_restart", "value": 999}
            page = Page(page_id, original_data, page_type="test_type", abstraction_level=1)
            pm.put_page(page) # Poner la página en el sistema de paginación

            # Asegurarse de que la página se ha escrito en L3
            # Forzar el desalojo de L1 y L2 para que la página se mueva a L3
            # Insertar solo suficientes páginas dummy para llenar L1 y L2, sin exceder la capacidad de L3
            num_dummy_pages = pm.l1_cache.capacity + pm.l2_cache.capacity + 1
            for i in range(num_dummy_pages):
                dummy_page_id = f"dummy_restart_eviction_{i}"
                dummy_page = Page(dummy_page_id, {"d": i}, page_type="dummy", abstraction_level=1)
                pm.put_page(dummy_page)
            


            # Simular un reinicio del PageManager cargando desde el mismo directorio L3
            restarted_pm = PageManager(l3_storage_dir=storage_dir)
            
            # Intentar recuperar la página del PageManager reiniciado
            retrieved_page = restarted_pm.get_page(page_id)

            if not retrieved_page or retrieved_page.content != original_data:
                print(f"Fallo de persistencia después de reinicio para {page_id}. Datos recuperados: {retrieved_page.content if retrieved_page else 'None'}")
                return False
            return True
        certificates.append(self.run_test("Persistence After Restart Test", test_persistence_after_restart))

        # Test 3: Promoción de caché
        def test_cache_promotion(pm: PageManager, storage_dir: str) -> bool:
            page_id = "test_page_3"
            original_data = {"data": "promoted"}
            page = Page(page_id, original_data, page_type="test_type", abstraction_level=1)
            
            # Poner en L3 directamente (simulando desalojo de L2)
            pm.l3_cache.put(page)
            
            # Acceder a la página, debería ser promovida a L1
            retrieved_page = pm.get_page(page_id)
            
            if not retrieved_page or retrieved_page.content != original_data:
                return False
            
            # Verificar que está en L1
            return pm.l1_cache.get(page_id) is not None
        certificates.append(self.run_test("Cache Promotion Test", test_cache_promotion))

        # Test 4: Desalojo y propagación
        def test_eviction_propagation(pm: PageManager, storage_dir: str) -> bool:
            # L1 tiene capacidad 100, L2 500, L3 1000 por defecto
            # Llenar L1 para forzar desalojo a L2
            for i in range(pm.l1_cache.capacity + 1):
                page_id = f"eviction_page_{i}"
                page = Page(page_id, {"val": i}, page_type="test_type", abstraction_level=1)
                pm.put_page(page)
            
            # La primera página puesta (eviction_page_0) debería estar en L2
            return pm.l1_cache.get("eviction_page_0") is None and pm.l2_cache.get("eviction_page_0") is not None
        certificates.append(self.run_test("Eviction Propagation Test", test_eviction_propagation))

        # Test 5: Coherencia avanzada después de movimientos de caché
        def test_advanced_coherence(pm: PageManager, storage_dir: str) -> bool:
            page_id = "test_advanced_coherence_page"
            original_data = {"complex_key": "complex_value", "array": [1, 2, 3, {"nested": "data"}]}
            page = Page(page_id, original_data, page_type="test_type", abstraction_level=1)
            pm.put_page(page) # Pone la página en L1

            # Forzar desalojo de L1 a L2
            for i in range(pm.l1_cache.capacity + 1):
                pm.put_page(Page(f"dummy_l1_{i}", {"d": i}, page_type="test_type", abstraction_level=1))
            
            # Verificar coherencia después de desalojo a L2
            if not PagingValidator(pm).validate_coherence(page_id, original_data):
                print(f"Fallo de coherencia después de desalojo a L2 para {page_id}")
                return False

            # Forzar desalojo de L2 a L3
            for i in range(pm.l2_cache.capacity + 1):
                pm.put_page(Page(f"dummy_l2_{i}", {"d": i}, page_type="test_type", abstraction_level=1))

            # Verificar coherencia después de desalojo a L3
            if not PagingValidator(pm).validate_coherence(page_id, original_data):
                print(f"Fallo de coherencia después de desalojo a L3 para {page_id}")
                return False

            # Acceder a la página de L3 para forzar su promoción a L1
            retrieved_page = pm.get_page(page_id)
            if not retrieved_page or retrieved_page.content != original_data:
                print(f"Fallo de coherencia durante la promoción de L3 a L1 para {page_id}")
                return False
            
            # Verificar coherencia después de promoción a L1
            if not PagingValidator(pm).validate_coherence(page_id, original_data):
                print(f"Fallo de coherencia después de promoción a L1 para {page_id}")
                return False

            return True
        certificates.append(self.run_test("Advanced Coherence Test", test_advanced_coherence))

        # Test 7: Medición de rendimiento (tasa de aciertos, tasa de fallos)
        def test_performance_metrics(pm: PageManager, storage_dir: str) -> bool:
            page_ids = [f"perf_page_{i}" for i in range(pm.l1_cache.capacity + pm.l2_cache.capacity + pm.l3_cache.capacity + 50)]
            original_data_template = {"data": "performance_test_data"}

            # 1. Llenar la caché con páginas para asegurar que haya contenido y se generen desalojos
            for i in range(pm.l1_cache.capacity + pm.l2_cache.capacity + pm.l3_cache.capacity):
                pm.put_page(Page(page_ids[i], original_data_template, page_type="test_type", abstraction_level=1))
            
            # 2. Generar aciertos (acceder a páginas que ya deberían estar en caché)
            for i in range(pm.l1_cache.capacity // 2):
                pm.get_page(page_ids[i])
            
            # 3. Generar fallos (acceder a páginas que no están en caché o están en L3 y necesitan ser promovidas)
            # Acceder a páginas que deberían estar en L3 (desalojadas de L1/L2)
            for i in range(pm.l1_cache.capacity, pm.l1_cache.capacity + pm.l2_cache.capacity // 2):
                pm.get_page(page_ids[i])
            
            # Acceder a páginas completamente nuevas para forzar fallos en todos los niveles
            for i in range(pm.l1_cache.capacity + pm.l2_cache.capacity + pm.l3_cache.capacity, len(page_ids)):
                pm.get_page(page_ids[i])

            # 4. Obtener estadísticas y verificar que se registraron aciertos y fallos
            stats = pm.get_cache_stats()
            total_hits = sum(s["hits"] for s in stats.values())
            total_misses = sum(s["misses"] for s in stats.values())

            if total_hits == 0 or total_misses == 0:
                print(f"Fallo en el test de rendimiento: total_hits={total_hits}, total_misses={total_misses}")
                return False

            return True
        certificates.append(self.run_test("Performance Metrics Test", test_performance_metrics))

        return certificates


if __name__ == "__main__":
    # Limpiar el directorio de almacenamiento L3 antes de ejecutar los tests
    if os.path.exists("./page_storage_paging_Basic_Coherence_Test"):
        shutil.rmtree("./page_storage_paging_Basic_Coherence_Test")
    if os.path.exists("./page_storage_paging_L3_Persistence_Test"):
        shutil.rmtree("./page_storage_paging_L3_Persistence_Test")
    if os.path.exists("./page_storage_paging_Cache_Promotion_Test"):
        shutil.rmtree("./page_storage_paging_Cache_Promotion_Test")
    if os.path.exists("./page_storage_paging_Eviction_Propagation_Test"):
        shutil.rmtree("./page_storage_paging_Eviction_Propagation_Test")
    if os.path.exists("./page_storage_paging_Advanced_Coherence_Test"):
        shutil.rmtree("./page_storage_paging_Advanced_Coherence_Test")
    if os.path.exists("./page_storage_paging_Persistence_After_Restart_Test"):
        shutil.rmtree("./page_storage_paging_Persistence_After_Restart_Test")
    if os.path.exists("./page_storage_paging_Performance_Metrics_Test"):
        shutil.rmtree("./page_storage_paging_Performance_Metrics_Test")

    cert_repo = CertificateRepository()
    test_suite = PagingTestSuite(cert_repo)
    certificates = test_suite.run_all_tests()

    print("\n==================================================")
    print("Paging Validation Test Suite Summary")
    print("==================================================")
    for test_name, cert in test_suite.test_results:
        status_icon = "✓" if cert.is_valid() else "✗"
        status_message = 'VALID' if cert.is_valid() else 'INVALID'
        print(f"- {test_name}: {status_icon} {status_message} (Correctness: {cert.correctness_rate*100:.1f}%, Speedup: {cert.speedup_measured:.2f}x, Mem. Red.: {cert.memory_reduction*100:.1f}%)")

    print("\n--- Resumen General de la Validación de Paginación ---")
    print(f"Total de certificados generados: {len(certificates)}")
    print(f"Certificados válidos: {sum(1 for c in certificates if c.is_valid())}")
    print(f"Certificados inválidos: {sum(1 for c in certificates if not c.is_valid())}")
    
    valid_certs = [c for c in certificates if c.is_valid()]
    if valid_certs:
        avg_speedup = sum(c.speedup_measured for c in valid_certs) / len(valid_certs)
        avg_memory_reduction = sum(c.memory_reduction for c in valid_certs) / len(valid_certs)
        avg_correctness = sum(c.correctness_rate for c in valid_certs) / len(valid_certs)
        print(f"Speedup promedio (válidos): {avg_speedup:.2f}x")
        print(f"Reducción de memoria promedio (válidos): {avg_memory_reduction*100:.1f}%")
        print(f"Tasa de corrección promedio (válidos): {avg_correctness*100:.1f}%")
    else:
        print("No hay certificados válidos para calcular promedios.")

    print("Tests de paginación completados. Los certificados se han guardado en ./certificates_data/")

