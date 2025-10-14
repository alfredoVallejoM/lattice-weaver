from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import hashlib
import json
import time

@dataclass
class Page:
    """
    Representa una página semántica en el sistema de paginación.
    
    Una página es una unidad lógica de datos que encapsula una estructura
    significativa del CSP o del compilador multiescala.
    
    Attributes:
        id: Identificador único de la página (hash del contenido y metadatos clave).
        content: El contenido real de la página (e.g., un CSP, un conjunto de soluciones).
        page_type: Tipo de contenido semántico (e.g., 'cluster_csp', 'solutions_set').
        abstraction_level: Nivel de abstracción al que pertenece esta página.
        metadata: Diccionario para metadatos adicionales (timestamp, tamaño, etc.).
    """
    id: str
    content: Any
    page_type: str
    abstraction_level: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
        if 'timestamp' not in self.metadata:
            self.metadata['timestamp'] = time.time()
        if 'size_bytes' not in self.metadata:
            self.metadata['size_bytes'] = self._estimate_size()

    def _generate_id(self) -> str:
        # Genera un ID basado en un hash del contenido y metadatos clave
        # Esto es un placeholder; la serialización real se hará en Serializer/Deserializer
        data_to_hash = {
            "content_hash": hashlib.sha256(str(self.content).encode()).hexdigest(),
            "page_type": self.page_type,
            "abstraction_level": self.abstraction_level
        }
        return hashlib.sha256(json.dumps(data_to_hash, sort_keys=True).encode()).hexdigest()

    def _estimate_size(self) -> int:
        # Estima el tamaño de la página en bytes (placeholder)
        # Una implementación real serializaría el contenido para obtener el tamaño exacto
        return len(str(self.content).encode())

    def serialize(self) -> bytes:
        """Serializa la página a bytes. Placeholder."""
        # Esto será manejado por el Serializer/Deserializer real
        return json.dumps(self.to_dict()).encode()

    @classmethod
    def deserialize(cls, data: bytes) -> 'Page':
        """Deserializa bytes a una página. Placeholder."""
        # Esto será manejado por el Serializer/Deserializer real
        data_dict = json.loads(data.decode())
        deserialized_content = json.loads(data_dict["content"]) if isinstance(data_dict["content"], str) and (data_dict["content"].startswith("{") or data_dict["content"].startswith("[")) else data_dict["content"]
        return cls(id=data_dict["id"], 
                   content=deserialized_content, 
                   page_type=data_dict["page_type"], 
                   abstraction_level=data_dict["abstraction_level"],
                   metadata=data_dict["metadata"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": json.dumps(self.content) if isinstance(self.content, (dict, list)) else self.content,
            "page_type": self.page_type,
            "abstraction_level": self.abstraction_level,
            "metadata": self.metadata
        }

    def __repr__(self) -> str:
        return f"Page(id=\'{self.id[:8]}...\', type=\'{self.page_type}\' level={self.abstraction_level})"
