import os
import shutil
from lattice_weaver.validation.certificates import CertificateRepository
from lattice_weaver.validation.paging_validator import PagingTestSuite

if __name__ == "__main__":
    # Limpiar el directorio de almacenamiento L3 antes de ejecutar los tests
    # Asegurarse de que los directorios se limpian correctamente para cada test
    test_names = [
        "Basic Coherence Test",
        "L3 Persistence Test",
        "Cache Promotion Test",
        "Eviction Propagation Test"
    ]
    for name in test_names:
        dir_path = f"./page_storage_paging_{name.replace(' ', '_')}"
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    cert_repo = CertificateRepository()
    test_suite = PagingTestSuite(cert_repo)
    certificates = test_suite.run_all_tests()

    print("\n==================================================")
    print("Paging Validation Test Suite Summary")
    print("==================================================")
    for test_name, cert in test_suite.test_results:
        status_icon = "✓" if cert.is_valid() else "✗"
        print(f"- {test_name}: {status_icon} {cert.status_message} (Correctness: {cert.correctness_rate*100:.1f}%, Speedup: {cert.speedup_measured:.2f}x, Mem. Red.: {cert.memory_reduction*100:.1f}%)")

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

