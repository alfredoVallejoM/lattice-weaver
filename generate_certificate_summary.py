import os
import json
from pathlib import Path
import shutil

from lattice_weaver.validation.certificates import ValidationCertificate, CertificateRepository
from lattice_weaver.validation.paging_validator import PagingTestSuite
from lattice_weaver.paging.page_manager import PageManager

def generate_summary(certificates_dir: str = './certificates_data') -> dict:
    # Limpiar el directorio de certificados antes de generar el resumen
    if Path(certificates_dir).exists():
        shutil.rmtree(certificates_dir)
    Path(certificates_dir).mkdir(parents=True, exist_ok=True)

    # Crear una instancia de CertificateRepository para que PagingTestSuite la use
    repo = CertificateRepository(storage_path=certificates_dir)
    
    # Crear una instancia de PagingTestSuite
    test_suite = PagingTestSuite(repo)
    
    # Ejecutar todos los tests y obtener los certificados
    generated_certificates = test_suite.run_all_tests()

    # El repositorio ya almacena los certificados durante la ejecución de los tests
    # Ahora podemos generar el informe directamente desde el repositorio
    summary = repo.generate_report()
    
    # Filtrar solo los certificados relacionados con el PagingValidator (que ya son todos en este caso)
    paging_certs = [cert for cert in repo.certificates.values() if cert.level_name == "paging"]
    
    # Generar un resumen más específico para los tests de paginación
    paging_summary = {
        "total_paging_certificates": len(paging_certs),
        "valid_paging_certificates": len([c for c in paging_certs if c.is_valid()]),
        "invalid_paging_certificates": len([c for c in paging_certs if not c.is_valid()]),
        "details": []
    }

    for cert in paging_certs:
        paging_summary["details"].append({
            "test_name": cert.metadata.get("test_name", "N/A"),
            "status": "VALID" if cert.is_valid() else "INVALID",
            "correctness_rate": cert.correctness_rate,
            "speedup_measured": cert.speedup_measured,
            "memory_reduction": cert.memory_reduction,
            "timestamp": cert.timestamp,
            "signature": cert.signature
        })

    return paging_summary

if __name__ == "__main__":
    # Añadir el directorio raíz del proyecto al PYTHONPATH
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "lattice-weaver")))

    print("--- Generando resumen de certificados de validación para PagingValidator ---")
    summary_report = generate_summary()
    print(json.dumps(summary_report, indent=2))
    print("--- Resumen de certificados generado ---")

