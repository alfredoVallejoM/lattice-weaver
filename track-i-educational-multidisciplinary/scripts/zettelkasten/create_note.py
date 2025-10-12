#!/usr/bin/env python3
"""
Script para crear nuevas notas en el Zettelkasten del Track I.

Asiste en la creación de notas asignando IDs automáticamente,
usando templates apropiados y sugiriendo conexiones.

Uso:
    python create_note.py --type fenomeno
    python create_note.py --type isomorfismo --interactive
"""

import os
import sys
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Añadir el directorio raíz al path
SCRIPT_DIR = Path(__file__).parent
TRACK_DIR = SCRIPT_DIR.parent.parent
ZETTEL_DIR = TRACK_DIR / "zettelkasten"
TEMPLATES_DIR = TRACK_DIR / "templates"

# Mapeo de tipos a prefijos y directorios
TYPE_CONFIG = {
    "fenomeno": {"prefix": "F", "dir": "fenomenos", "template": "template_fenomeno.md"},
    "categoria": {"prefix": "C", "dir": "categorias", "template": "template_categoria.md"},
    "isomorfismo": {"prefix": "I", "dir": "isomorfismos", "template": "template_isomorfismo.md"},
    "tecnica": {"prefix": "T", "dir": "tecnicas", "template": "template_tecnica.md"},
    "dominio": {"prefix": "D", "dir": "dominios", "template": None},
    "concepto": {"prefix": "K", "dir": "conceptos", "template": None},
    "mapeo": {"prefix": "M", "dir": "mapeos", "template": None},
}


def get_next_id(note_type: str) -> str:
    """
    Obtiene el siguiente ID disponible para un tipo de nota.
    
    Args:
        note_type: Tipo de nota (fenomeno, categoria, etc.)
    
    Returns:
        ID en formato PREFIX### (ej. F001, C003)
    """
    config = TYPE_CONFIG[note_type]
    prefix = config["prefix"]
    notes_dir = ZETTEL_DIR / config["dir"]
    
    # Buscar todos los archivos con el prefijo
    existing_ids = []
    if notes_dir.exists():
        for file in notes_dir.glob(f"{prefix}*.md"):
            match = re.match(rf"{prefix}(\d+)_", file.name)
            if match:
                existing_ids.append(int(match.group(1)))
    
    # Obtener el siguiente número
    next_num = max(existing_ids, default=0) + 1
    return f"{prefix}{next_num:03d}"


def load_template(note_type: str) -> Optional[str]:
    """
    Carga el template para un tipo de nota.
    
    Args:
        note_type: Tipo de nota
    
    Returns:
        Contenido del template o None si no existe
    """
    config = TYPE_CONFIG[note_type]
    template_name = config.get("template")
    
    if not template_name:
        return None
    
    template_path = TEMPLATES_DIR / template_name
    if not template_path.exists():
        return None
    
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def create_note_interactive(note_type: str) -> Dict[str, str]:
    """
    Crea una nota de forma interactiva pidiendo información al usuario.
    
    Args:
        note_type: Tipo de nota a crear
    
    Returns:
        Diccionario con la información de la nota
    """
    print(f"\n=== Creando nueva nota de tipo: {note_type.upper()} ===\n")
    
    # Obtener ID automáticamente
    note_id = get_next_id(note_type)
    print(f"ID asignado: {note_id}")
    
    # Pedir información básica
    titulo = input("\nTítulo de la nota: ").strip()
    
    # Información específica por tipo
    info = {
        "id": note_id,
        "tipo": note_type,
        "titulo": titulo,
        "fecha_creacion": datetime.now().strftime("%Y-%m-%d"),
        "fecha_modificacion": datetime.now().strftime("%Y-%m-%d"),
        "estado": "borrador",
    }
    
    if note_type == "fenomeno":
        dominios_str = input("Dominios (separados por coma): ").strip()
        info["dominios"] = [d.strip() for d in dominios_str.split(",") if d.strip()]
        
        categorias_str = input("Categorías (IDs separados por coma, ej. C001,C002): ").strip()
        info["categorias"] = [c.strip() for c in categorias_str.split(",") if c.strip()]
        
        tags_str = input("Tags (separados por coma): ").strip()
        info["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]
        
        prioridad = input("Prioridad (maxima/alta/media/baja) [media]: ").strip() or "media"
        info["prioridad"] = prioridad
    
    elif note_type == "isomorfismo":
        nivel = input("Nivel (exacto/fuerte/analogia) [fuerte]: ").strip() or "fuerte"
        info["nivel"] = nivel
        
        fenomenos_str = input("Fenómenos relacionados (IDs separados por coma, ej. F001,F002): ").strip()
        info["fenomenos"] = [f.strip() for f in fenomenos_str.split(",") if f.strip()]
        
        dominios_str = input("Dominios (separados por coma): ").strip()
        info["dominios"] = [d.strip() for d in dominios_str.split(",") if d.strip()]
        
        categorias_str = input("Categorías (IDs separados por coma): ").strip()
        info["categorias"] = [c.strip() for c in categorias_str.split(",") if c.strip()]
        
        tags_str = input("Tags (separados por coma): ").strip()
        info["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]
        
        info["validacion"] = "pendiente"
    
    elif note_type == "categoria":
        tags_str = input("Tags (separados por coma): ").strip()
        info["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]
        
        info["fenomenos_count"] = 0
        info["dominios_count"] = 0
    
    elif note_type == "tecnica":
        dominio_origen = input("Dominio de origen: ").strip()
        info["dominio_origen"] = dominio_origen
        
        categorias_str = input("Categorías aplicables (IDs separados por coma): ").strip()
        info["categorias_aplicables"] = [c.strip() for c in categorias_str.split(",") if c.strip()]
        
        tags_str = input("Tags (separados por coma): ").strip()
        info["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]
        
        info["implementado"] = False
    
    return info


def fill_template(template: str, info: Dict[str, str]) -> str:
    """
    Rellena un template con la información proporcionada.
    
    Args:
        template: Contenido del template
        info: Diccionario con información de la nota
    
    Returns:
        Template rellenado
    """
    # Reemplazar ID
    template = template.replace("id: F###", f"id: {info['id']}")
    template = template.replace("id: C###", f"id: {info['id']}")
    template = template.replace("id: I###", f"id: {info['id']}")
    template = template.replace("id: T###", f"id: {info['id']}")
    
    # Reemplazar tipo
    template = template.replace(f"tipo: {info['tipo']}", f"tipo: {info['tipo']}")
    
    # Reemplazar título
    template = template.replace("titulo: [Nombre del Fenómeno]", f"titulo: {info['titulo']}")
    template = template.replace("titulo: [Fenómeno A] ≅ [Fenómeno B]", f"titulo: {info['titulo']}")
    template = template.replace("titulo: [Nombre de la Categoría Estructural]", f"titulo: {info['titulo']}")
    template = template.replace("titulo: [Nombre de la Técnica/Algoritmo]", f"titulo: {info['titulo']}")
    
    # Reemplazar en el cuerpo
    template = template.replace("# [Nombre del Fenómeno]", f"# {info['titulo']}")
    template = template.replace("# Isomorfismo: [Fenómeno A] ≅ [Fenómeno B]", f"# Isomorfismo: {info['titulo']}")
    template = template.replace("# Categoría: [Nombre de la Categoría Estructural]", f"# Categoría: {info['titulo']}")
    template = template.replace("# Técnica: [Nombre de la Técnica/Algoritmo]", f"# Técnica: {info['titulo']}")
    
    # Reemplazar fechas
    template = template.replace("fecha_creacion: YYYY-MM-DD", f"fecha_creacion: {info['fecha_creacion']}")
    template = template.replace("fecha_modificacion: YYYY-MM-DD", f"fecha_modificacion: {info['fecha_modificacion']}")
    
    # Reemplazar campos específicos
    if "dominios" in info:
        dominios_yaml = "[" + ", ".join(info["dominios"]) + "]"
        template = re.sub(r"dominios: \[.*?\]", f"dominios: {dominios_yaml}", template)
    
    if "categorias" in info:
        categorias_yaml = "[" + ", ".join(info["categorias"]) + "]"
        template = re.sub(r"categorias: \[.*?\]", f"categorias: {categorias_yaml}", template)
    
    if "tags" in info:
        tags_yaml = "[" + ", ".join(info["tags"]) + "]"
        template = re.sub(r"tags: \[.*?\]", f"tags: {tags_yaml}", template)
    
    if "prioridad" in info:
        template = re.sub(r"prioridad: \w+", f"prioridad: {info['prioridad']}", template)
    
    if "nivel" in info:
        template = re.sub(r"nivel: \w+", f"nivel: {info['nivel']}", template)
    
    if "fenomenos" in info:
        fenomenos_yaml = "[" + ", ".join(info["fenomenos"]) + "]"
        template = re.sub(r"fenomenos: \[.*?\]", f"fenomenos: {fenomenos_yaml}", template)
    
    if "dominio_origen" in info:
        template = template.replace("dominio_origen: [dominio]", f"dominio_origen: {info['dominio_origen']}")
    
    if "categorias_aplicables" in info:
        categorias_yaml = "[" + ", ".join(info["categorias_aplicables"]) + "]"
        template = re.sub(r"categorias_aplicables: \[.*?\]", f"categorias_aplicables: {categorias_yaml}", template)
    
    # Reemplazar última actualización al final
    template = re.sub(
        r"\*\*Última actualización:\*\* YYYY-MM-DD",
        f"**Última actualización:** {info['fecha_creacion']}",
        template
    )
    
    return template


def save_note(note_type: str, info: Dict[str, str], content: str) -> Path:
    """
    Guarda la nota en el directorio apropiado.
    
    Args:
        note_type: Tipo de nota
        info: Información de la nota
        content: Contenido de la nota
    
    Returns:
        Path al archivo creado
    """
    config = TYPE_CONFIG[note_type]
    notes_dir = ZETTEL_DIR / config["dir"]
    notes_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear nombre de archivo
    titulo_slug = re.sub(r'[^a-z0-9]+', '_', info['titulo'].lower()).strip('_')
    filename = f"{info['id']}_{titulo_slug}.md"
    filepath = notes_dir / filename
    
    # Guardar archivo
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Crear nueva nota en el Zettelkasten del Track I"
    )
    parser.add_argument(
        "--type",
        choices=list(TYPE_CONFIG.keys()),
        required=True,
        help="Tipo de nota a crear"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Modo interactivo (pide información al usuario)"
    )
    parser.add_argument(
        "--titulo",
        help="Título de la nota (modo no interactivo)"
    )
    
    args = parser.parse_args()
    
    # Verificar que los directorios existen
    if not ZETTEL_DIR.exists():
        print(f"Error: Directorio zettelkasten no encontrado en {ZETTEL_DIR}")
        sys.exit(1)
    
    if not TEMPLATES_DIR.exists():
        print(f"Error: Directorio templates no encontrado en {TEMPLATES_DIR}")
        sys.exit(1)
    
    # Modo interactivo
    if args.interactive:
        info = create_note_interactive(args.type)
    else:
        # Modo no interactivo (básico)
        if not args.titulo:
            print("Error: --titulo es requerido en modo no interactivo")
            sys.exit(1)
        
        note_id = get_next_id(args.type)
        info = {
            "id": note_id,
            "tipo": args.type,
            "titulo": args.titulo,
            "fecha_creacion": datetime.now().strftime("%Y-%m-%d"),
            "fecha_modificacion": datetime.now().strftime("%Y-%m-%d"),
            "estado": "borrador",
        }
    
    # Cargar template
    template = load_template(args.type)
    if not template:
        print(f"Advertencia: No hay template para tipo '{args.type}'")
        print("Creando nota básica...")
        template = f"""---
id: {info['id']}
tipo: {info['tipo']}
titulo: {info['titulo']}
fecha_creacion: {info['fecha_creacion']}
fecha_modificacion: {info['fecha_modificacion']}
estado: borrador
---

# {info['titulo']}

## Descripción

[Agregar descripción aquí]

## Conexiones

[Agregar conexiones aquí]

---

**Última actualización:** {info['fecha_creacion']}
"""
    else:
        # Rellenar template
        template = fill_template(template, info)
    
    # Guardar nota
    filepath = save_note(args.type, info, template)
    
    print(f"\n✅ Nota creada exitosamente:")
    print(f"   ID: {info['id']}")
    print(f"   Archivo: {filepath}")
    print(f"\nPróximos pasos:")
    print(f"1. Editar el archivo para completar el contenido")
    print(f"2. Crear enlaces bidireccionales con otras notas")
    print(f"3. Ejecutar: python scripts/zettelkasten/update_catalog.py")
    print(f"4. Ejecutar: python scripts/zettelkasten/validate_zettelkasten.py")


if __name__ == "__main__":
    main()

