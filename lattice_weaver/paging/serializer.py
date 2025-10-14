from typing import Any, Dict
import json
import zlib # Para compresión

from lattice_weaver.paging.page import Page

class PageSerializer:
    """
    Clase para serializar y deserializar objetos Page.
    """
    @staticmethod
    def serialize(page: Page) -> bytes:
        """
        Serializa un objeto Page a bytes, incluyendo compresión.
        """
        # Convertir el contenido a un formato serializable (ej. JSON)
        # Para CSPs, esto implicaría serializar variables, dominios y restricciones
        # Por ahora, usamos una representación simple.
        serializable_content = page.to_dict()
        json_data = json.dumps(serializable_content, sort_keys=True).encode("utf-8")
        return zlib.compress(json_data)

    @staticmethod
    def deserialize(data: bytes) -> Page:
        """
        Deserializa bytes a un objeto Page, incluyendo descompresión.
        """
        decompressed_data = zlib.decompress(data)
        data_dict = json.loads(decompressed_data.decode("utf-8"))
        
        # Reconstruir el objeto Page. El contenido real necesitará un deserializador específico
        # para CSPs u otras estructuras complejas.
        # Por ahora, el contenido se almacena como un string en to_dict
        return Page(id=data_dict["id"],
                    content=json.loads(data_dict["content"]) if isinstance(data_dict["content"], str) and (data_dict["content"].startswith("{") or data_dict["content"].startswith("[")) else data_dict["content"], # Deserializar de JSON si es string y parece JSON
                    page_type=data_dict["page_type"],
                    abstraction_level=data_dict["abstraction_level"],
                    metadata=data_dict["metadata"])

# Nota: Para una implementación completa, el campo 'content' de la Page
# necesitaría un serializador/deserializador específico para cada 'page_type'.
# Por ejemplo, si page_type es 'cluster_csp', necesitaríamos una función
# para serializar/deserializar un objeto CSP.
