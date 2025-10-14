import sys
import os

# Añadir el directorio raíz del proyecto al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from lattice_weaver.validation.certificates import CertificateRepository
from lattice_weaver.validation.renormalization_validator import RenormalizationTestSuite

def main():
    print("Iniciando la ejecución de los tests de renormalización...")
    
    # Inicializar el repositorio de certificados
    cert_repo = CertificateRepository(storage_path="./certificates_data")
    cert_repo.clear() # Limpiar certificados anteriores para una ejecución limpia

    # Inicializar la suite de tests de renormalización
    test_suite = RenormalizationTestSuite()

    # Ejecutar todos los tests
    certificates = test_suite.run_all_tests()
    for cert in certificates:
        cert_repo.store(cert)

    print("\n--- Resumen General de la Validación de Renormalización ---")
    report = cert_repo.generate_report()
    print(f"Total de certificados generados: {report['total_certificates']}")
    print(f"Certificados válidos: {report['valid_certificates']}")
    print(f"Certificados inválidos: {report['invalid_certificates']}")
    print(f"Speedup promedio (válidos): {report['avg_speedup']:.2f}x")
    print(f"Reducción de memoria promedio (válidos): {report['avg_memory_reduction'] * 100:.1f}%")
    print(f"Tasa de corrección promedio (válidos): {report['avg_correctness'] * 100:.1f}%")

    # Opcional: Verificar la cadena de certificados si hay una secuencia lógica
    # Por ahora, solo verificamos individualmente.

    print("\nTests de renormalización completados. Los certificados se han guardado en ./certificates_data/")

if __name__ == "__main__":
    main()

