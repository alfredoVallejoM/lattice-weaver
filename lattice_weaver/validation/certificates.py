# lattice_weaver/validation/certificates.py

"""
Sistema de Certificados de Validación

Este módulo implementa el sistema de certificados que atestiguan la corrección
de cada nivel del compilador multiescala. Cada nivel debe culminar con un
certificado válido antes de proceder al siguiente.

Principios de diseño:
- Validación intrínseca: Cada nivel se auto-valida
- Composicionalidad: Certificados se componen para validación end-to-end
- Trazabilidad: Firma criptográfica para auditoría
"""

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import time
import hashlib
import json
import numpy as np


@dataclass
class ValidationCertificate:
    """
    Certificado de validación de un nivel.
    
    Un certificado atestigua que un nivel ha sido validado y cumple con
    los invariantes requeridos. Contiene información sobre:
    - Validación estática (type checking, invariantes)
    - Validación dinámica (tests en runtime)
    - Validación empírica (benchmarks, métricas)
    
    Attributes:
        level_name: Nombre del nivel validado
        timestamp: Momento de generación del certificado
        type_checked: Si el nivel pasó type checking estático
        invariants_verified: Lista de invariantes verificados
        runtime_tests_passed: Número de tests dinámicos que pasaron
        runtime_tests_failed: Número de tests dinámicos que fallaron
        speedup_measured: Speedup medido empíricamente
        memory_reduction: Reducción de memoria medida (0.0 a 1.0)
        correctness_rate: Tasa de corrección (0.0 a 1.0)
        signature: Firma criptográfica SHA-256
        metadata: Información adicional específica del nivel
    """
    
    level_name: str
    timestamp: float
    
    # Validación estática
    type_checked: bool
    invariants_verified: List[str]
    
    # Validación dinámica
    runtime_tests_passed: int
    runtime_tests_failed: int
    
    # Validación empírica
    speedup_measured: float
    memory_reduction: float
    correctness_rate: float
    
    # Trazabilidad
    signature: str
    
    # Metadata adicional
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self, min_correctness: float = 0.99) -> bool:
        """
        Verifica si el certificado es válido.
        
        Un certificado es válido si:
        - Pasó type checking
        - No hay tests fallidos
        - Tasa de corrección >= min_correctness
        
        Args:
            min_correctness: Tasa mínima de corrección requerida
        
        Returns:
            True si el certificado es válido
        """
        return (
            self.type_checked and
            self.runtime_tests_failed == 0 and
            self.correctness_rate >= min_correctness
        )
    
    def to_dict(self) -> dict:
        """Convierte certificado a diccionario."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serializa certificado a JSON."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ValidationCertificate':
        """Crea certificado desde diccionario."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ValidationCertificate':
        """Deserializa certificado desde JSON."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        """Representación legible del certificado."""
        status = "✓ VALID" if self.is_valid() else "✗ INVALID"
        return (
            f"ValidationCertificate({status})\n"
            f"  Level: {self.level_name}\n"
            f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}\n"
            f"  Tests: {self.runtime_tests_passed} passed, {self.runtime_tests_failed} failed\n"
            f"  Speedup: {self.speedup_measured:.2f}x\n"
            f"  Memory reduction: {self.memory_reduction*100:.1f}%\n"
            f"  Correctness: {self.correctness_rate*100:.1f}%\n"
            f"  Signature: {self.signature[:16]}..."
        )


def compose_certificates(cert_lower: ValidationCertificate,
                        cert_upper: ValidationCertificate) -> ValidationCertificate:
    """
    Compone dos certificados de niveles consecutivos.
    
    La composición verifica que los niveles son compatibles y combina
    sus métricas de forma apropiada:
    - Speedup: multiplicativo (speedup total = speedup_L0 × speedup_L1)
    - Memory: mínimo (el cuello de botella)
    - Correctness: multiplicativo (probabilidad conjunta)
    
    Args:
        cert_lower: Certificado del nivel inferior
        cert_upper: Certificado del nivel superior
    
    Returns:
        Certificado compuesto
    
    Raises:
        ValueError: Si los certificados no son componibles
    """
    # Verificar que ambos son válidos
    if not cert_lower.is_valid():
        raise ValueError(f"Lower certificate is invalid: {cert_lower.level_name}")
    
    if not cert_upper.is_valid():
        raise ValueError(f"Upper certificate is invalid: {cert_upper.level_name}")
    
    # Componer
    composed = ValidationCertificate(
        level_name=f"{cert_lower.level_name}_to_{cert_upper.level_name}",
        timestamp=time.time(),
        type_checked=cert_lower.type_checked and cert_upper.type_checked,
        invariants_verified=cert_lower.invariants_verified + cert_upper.invariants_verified,
        runtime_tests_passed=cert_lower.runtime_tests_passed + cert_upper.runtime_tests_passed,
        runtime_tests_failed=cert_lower.runtime_tests_failed + cert_upper.runtime_tests_failed,
        speedup_measured=cert_lower.speedup_measured * cert_upper.speedup_measured,
        memory_reduction=min(cert_lower.memory_reduction, cert_upper.memory_reduction),
        correctness_rate=cert_lower.correctness_rate * cert_upper.correctness_rate,
        signature=_hash_certificates(cert_lower, cert_upper),
        metadata={
            'composed_from': [cert_lower.level_name, cert_upper.level_name],
            'lower_cert_signature': cert_lower.signature,
            'upper_cert_signature': cert_upper.signature
        }
    )
    
    return composed


def _hash_certificates(cert_lower: ValidationCertificate,
                      cert_upper: ValidationCertificate) -> str:
    """
    Computa hash criptográfico de dos certificados.
    
    Args:
        cert_lower: Certificado inferior
        cert_upper: Certificado superior
    
    Returns:
        Hash SHA-256 hexadecimal
    """
    data = f"{cert_lower.signature}{cert_upper.signature}{time.time()}"
    return hashlib.sha256(data.encode()).hexdigest()


class CertificateRepository:
    """
    Repositorio centralizado de certificados de validación.
    
    Almacena certificados en memoria y disco, permitiendo:
    - Almacenar y recuperar certificados
    - Verificar cadenas de certificados
    - Generar reportes de validación
    
    Attributes:
        storage_path: Directorio donde se persisten certificados
        certificates: Diccionario de certificados en memoria (signature → cert)
    """
    
    def __init__(self, storage_path: str = './certificates'):
        """
        Inicializa repositorio.
        
        Args:
            storage_path: Ruta donde almacenar certificados
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.certificates: Dict[str, ValidationCertificate] = {}
        
        # Cargar certificados existentes
        self._load_existing_certificates()
    
    def store(self, cert: ValidationCertificate) -> None:
        """
        Almacena certificado en memoria y disco.
        
        Args:
            cert: Certificado a almacenar
        """
        # Guardar en memoria
        self.certificates[cert.signature] = cert
        
        # Persistir a disco
        cert_file = self.storage_path / f"{cert.level_name}_{int(cert.timestamp)}.json"
        with open(cert_file, 'w') as f:
            f.write(cert.to_json())
    
    def retrieve(self, signature: str) -> Optional[ValidationCertificate]:
        """
        Recupera certificado por firma.
        
        Args:
            signature: Firma SHA-256 del certificado
        
        Returns:
            Certificado o None si no existe
        """
        return self.certificates.get(signature)
    
    def get_all_for_level(self, level_name: str) -> List[ValidationCertificate]:
        """
        Obtiene todos los certificados de un nivel.
        
        Args:
            level_name: Nombre del nivel
        
        Returns:
            Lista de certificados del nivel
        """
        return [
            cert for cert in self.certificates.values()
            if cert.level_name == level_name
        ]
    
    def get_latest_for_level(self, level_name: str) -> Optional[ValidationCertificate]:
        """
        Obtiene el certificado más reciente de un nivel.
        
        Args:
            level_name: Nombre del nivel
        
        Returns:
            Certificado más reciente o None
        """
        certs = self.get_all_for_level(level_name)
        if not certs:
            return None
        
        return max(certs, key=lambda c: c.timestamp)
    
    def verify_chain(self, signatures: List[str]) -> bool:
        """
        Verifica cadena de certificados.
        
        Una cadena es válida si:
        - Todos los certificados existen
        - Todos los certificados son válidos
        - Los niveles son consecutivos (si aplica)
        
        Args:
            signatures: Lista de firmas en orden
        
        Returns:
            True si la cadena es válida
        """
        certs = [self.retrieve(sig) for sig in signatures]
        
        # Verificar que todos existen
        if any(cert is None for cert in certs):
            return False
        
        # Verificar que todos son válidos
        if any(not cert.is_valid() for cert in certs):
            return False
        
        return True
    
    def generate_report(self) -> dict:
        """
        Genera reporte de todos los certificados.
        
        Returns:
            Diccionario con estadísticas
        """
        if not self.certificates:
            return {
                'total_certificates': 0,
                'valid_certificates': 0,
                'invalid_certificates': 0
            }
        
        valid_certs = [c for c in self.certificates.values() if c.is_valid()]
        
        return {
            'total_certificates': len(self.certificates),
            'valid_certificates': len(valid_certs),
            'invalid_certificates': len(self.certificates) - len(valid_certs),
            'avg_speedup': np.mean([c.speedup_measured for c in valid_certs]) if valid_certs else 0.0,
            'avg_memory_reduction': np.mean([c.memory_reduction for c in valid_certs]) if valid_certs else 0.0,
            'avg_correctness': np.mean([c.correctness_rate for c in valid_certs]) if valid_certs else 0.0,
            'levels': list(set(c.level_name for c in self.certificates.values()))
        }
    
    def _load_existing_certificates(self) -> None:
        """Carga certificados existentes desde disco."""
        for cert_file in self.storage_path.glob('*.json'):
            try:
                with open(cert_file) as f:
                    cert = ValidationCertificate.from_json(f.read())
                    self.certificates[cert.signature] = cert
            except Exception as e:
                print(f"Warning: Could not load certificate {cert_file}: {e}")
    
    def clear(self) -> None:
        """Limpia todos los certificados (útil para testing)."""
        self.certificates.clear()
        
        # Opcional: eliminar archivos del disco
        # for cert_file in self.storage_path.glob('*.json'):
        #     cert_file.unlink()


def create_certificate(level_name: str,
                      type_checked: bool,
                      invariants_verified: List[str],
                      runtime_tests_passed: int,
                      runtime_tests_failed: int,
                      speedup_measured: float,
                      memory_reduction: float,
                      correctness_rate: float,
                      metadata: Optional[Dict[str, Any]] = None) -> ValidationCertificate:
    """
    Función helper para crear certificados.
    
    Args:
        level_name: Nombre del nivel
        type_checked: Si pasó type checking
        invariants_verified: Lista de invariantes verificados
        runtime_tests_passed: Tests pasados
        runtime_tests_failed: Tests fallidos
        speedup_measured: Speedup medido
        memory_reduction: Reducción de memoria
        correctness_rate: Tasa de corrección
        metadata: Metadata adicional
    
    Returns:
        Certificado creado
    """
    # Computar firma
    data = f"{level_name}{time.time()}{invariants_verified}{speedup_measured}"
    signature = hashlib.sha256(data.encode()).hexdigest()
    
    return ValidationCertificate(
        level_name=level_name,
        timestamp=time.time(),
        type_checked=type_checked,
        invariants_verified=invariants_verified,
        runtime_tests_passed=runtime_tests_passed,
        runtime_tests_failed=runtime_tests_failed,
        speedup_measured=speedup_measured,
        memory_reduction=memory_reduction,
        correctness_rate=correctness_rate,
        signature=signature,
        metadata=metadata or {}
    )

