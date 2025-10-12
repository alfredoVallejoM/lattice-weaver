#!/usr/bin/env python3
"""
Script para validar la consistencia del Zettelkasten.

Verifica:
- Enlaces bidireccionales
- IDs duplicados
- Formato de metadatos
- Notas huérfanas
- Enlaces rotos

Uso:
    python validate_zettelkasten.py
    python validate_zettelkasten.py --fix  # Intentar arreglar problemas automáticamente
"""

import os
import sys
import argparse
import re
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Añadir el directorio raíz al path
SCRIPT_DIR = Path(__file__).parent
TRACK_DIR = SCRIPT_DIR.parent.parent
ZETTEL_DIR = TRACK_DIR / "zettelkasten"


class ValidationError:
    """Representa un error de validación."""
    
    def __init__(self, severity: str, category: str, message: str, filepath: str = None):
        self.severity = severity  # 'error', 'warning', 'info'
        self.category = category
        self.message = message
        self.filepath = filepath
    
    def __str__(self):
        severity_emoji = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}
        emoji = severity_emoji.get(self.severity, '•')
        
        if self.filepath:
            return f"{emoji} [{self.category}] {self.message}\n   Archivo: {self.filepath}"
        else:
            return f"{emoji} [{self.category}] {self.message}"


def extract_metadata(filepath: Path) -> Tuple[Dict, List[ValidationError]]:
    """
    Extrae y valida los metadatos YAML de una nota.
    
    Args:
        filepath: Path al archivo
    
    Returns:
        Tupla (metadata, errores)
    """
    errors = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        errors.append(ValidationError(
            'error', 'FILE_READ',
            f"No se pudo leer el archivo: {e}",
            str(filepath)
        ))
        return {}, errors
    
    # Buscar el bloque YAML front matter
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not match:
        errors.append(ValidationError(
            'error', 'YAML_MISSING',
            "No se encontró bloque YAML front matter",
            str(filepath)
        ))
        return {}, errors
    
    yaml_content = match.group(1)
    try:
        metadata = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        errors.append(ValidationError(
            'error', 'YAML_INVALID',
            f"YAML inválido: {e}",
            str(filepath)
        ))
        return {}, errors
    
    # Validar campos requeridos
    required_fields = ['id', 'tipo', 'titulo', 'fecha_creacion', 'estado']
    for field in required_fields:
        if field not in metadata:
            errors.append(ValidationError(
                'error', 'FIELD_MISSING',
                f"Campo requerido '{field}' no encontrado",
                str(filepath)
            ))
    
    # Validar formato de ID
    if 'id' in metadata:
        note_id = metadata['id']
        if not re.match(r'^[A-Z]\d{3}$', note_id):
            errors.append(ValidationError(
                'error', 'ID_FORMAT',
                f"ID '{note_id}' no tiene formato válido (debe ser LNNN, ej. F001)",
                str(filepath)
            ))
    
    # Validar tipo
    valid_types = ['fenomeno', 'categoria', 'isomorfismo', 'tecnica', 'dominio', 'concepto', 'mapeo']
    if 'tipo' in metadata:
        if metadata['tipo'] not in valid_types:
            errors.append(ValidationError(
                'warning', 'TYPE_INVALID',
                f"Tipo '{metadata['tipo']}' no es uno de los tipos válidos: {valid_types}",
                str(filepath)
            ))
    
    # Validar estado
    valid_states = ['borrador', 'en_revision', 'completo']
    if 'estado' in metadata:
        if metadata['estado'] not in valid_states:
            errors.append(ValidationError(
                'warning', 'STATE_INVALID',
                f"Estado '{metadata['estado']}' no es uno de los estados válidos: {valid_states}",
                str(filepath)
            ))
    
    # Validar formato de fecha
    if 'fecha_creacion' in metadata:
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', str(metadata['fecha_creacion'])):
            errors.append(ValidationError(
                'warning', 'DATE_FORMAT',
                f"Fecha de creación '{metadata['fecha_creacion']}' no tiene formato YYYY-MM-DD",
                str(filepath)
            ))
    
    metadata['_filepath'] = filepath
    return metadata, errors


def extract_links(filepath: Path) -> List[str]:
    """
    Extrae todos los enlaces [[ID]] de una nota.
    
    Args:
        filepath: Path al archivo
    
    Returns:
        Lista de IDs enlazados
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar todos los enlaces [[ID]] o [[ID|texto]]
    links = re.findall(r'\[\[([A-Z]\d{3})', content)
    return links  # No eliminar duplicados para detectar enlaces múltiples


def collect_all_notes() -> Tuple[Dict[str, Dict], List[ValidationError]]:
    """
    Recolecta todas las notas del Zettelkasten.
    
    Returns:
        Tupla (diccionario {ID: metadata}, lista de errores)
    """
    notes = {}
    errors = []
    
    for subdir in ZETTEL_DIR.iterdir():
        if not subdir.is_dir():
            continue
        
        for filepath in subdir.glob("*.md"):
            metadata, file_errors = extract_metadata(filepath)
            errors.extend(file_errors)
            
            if metadata and 'id' in metadata:
                note_id = metadata['id']
                
                # Verificar IDs duplicados
                if note_id in notes:
                    errors.append(ValidationError(
                        'error', 'ID_DUPLICATE',
                        f"ID duplicado '{note_id}' encontrado en múltiples archivos",
                        str(filepath)
                    ))
                    errors.append(ValidationError(
                        'error', 'ID_DUPLICATE',
                        f"ID duplicado '{note_id}' (primera ocurrencia)",
                        str(notes[note_id]['_filepath'])
                    ))
                
                metadata['_links'] = extract_links(filepath)
                notes[note_id] = metadata
    
    return notes, errors


def validate_bidirectional_links(notes: Dict[str, Dict]) -> List[ValidationError]:
    """
    Valida que los enlaces sean bidireccionales.
    
    Args:
        notes: Diccionario de notas
    
    Returns:
        Lista de errores
    """
    errors = []
    
    for note_id, metadata in notes.items():
        links = metadata.get('_links', [])
        
        for linked_id in set(links):  # Usar set para evitar duplicados
            # Verificar que el enlace existe
            if linked_id not in notes:
                errors.append(ValidationError(
                    'error', 'LINK_BROKEN',
                    f"Enlace roto: [[{note_id}]] → [[{linked_id}]] (destino no existe)",
                    str(metadata['_filepath'])
                ))
                continue
            
            # Verificar que el enlace es bidireccional
            reverse_links = notes[linked_id].get('_links', [])
            if note_id not in reverse_links:
                errors.append(ValidationError(
                    'warning', 'LINK_UNIDIRECTIONAL',
                    f"Enlace unidireccional: [[{note_id}]] → [[{linked_id}]] (falta enlace inverso)",
                    str(metadata['_filepath'])
                ))
    
    return errors


def validate_orphan_notes(notes: Dict[str, Dict]) -> List[ValidationError]:
    """
    Detecta notas huérfanas (sin conexiones).
    
    Args:
        notes: Diccionario de notas
    
    Returns:
        Lista de advertencias
    """
    errors = []
    
    for note_id, metadata in notes.items():
        links = metadata.get('_links', [])
        
        # Contar enlaces entrantes
        incoming_links = sum(1 for other_meta in notes.values() 
                            if note_id in other_meta.get('_links', []))
        
        if len(links) == 0 and incoming_links == 0:
            errors.append(ValidationError(
                'info', 'NOTE_ORPHAN',
                f"Nota huérfana (sin conexiones): [[{note_id}]] - {metadata.get('titulo', 'Sin título')}",
                str(metadata['_filepath'])
            ))
    
    return errors


def validate_filename_consistency(notes: Dict[str, Dict]) -> List[ValidationError]:
    """
    Valida que los nombres de archivo sean consistentes con los IDs.
    
    Args:
        notes: Diccionario de notas
    
    Returns:
        Lista de errores
    """
    errors = []
    
    for note_id, metadata in notes.items():
        filepath = metadata['_filepath']
        filename = Path(filepath).name
        
        # El nombre de archivo debe comenzar con el ID
        if not filename.startswith(note_id):
            errors.append(ValidationError(
                'warning', 'FILENAME_INCONSISTENT',
                f"Nombre de archivo '{filename}' no comienza con ID '{note_id}'",
                str(filepath)
            ))
    
    return errors


def validate_category_references(notes: Dict[str, Dict]) -> List[ValidationError]:
    """
    Valida que las referencias a categorías existan.
    
    Args:
        notes: Diccionario de notas
    
    Returns:
        Lista de errores
    """
    errors = []
    
    # Recolectar todas las categorías existentes
    existing_categories = {nid for nid, meta in notes.items() 
                          if meta.get('tipo') == 'categoria'}
    
    for note_id, metadata in notes.items():
        categorias = metadata.get('categorias', [])
        if not isinstance(categorias, list):
            continue
        
        for cat_id in categorias:
            if cat_id not in existing_categories:
                errors.append(ValidationError(
                    'warning', 'CATEGORY_MISSING',
                    f"Referencia a categoría inexistente: {cat_id}",
                    str(metadata['_filepath'])
                ))
    
    return errors


def print_validation_report(errors: List[ValidationError]):
    """
    Imprime un reporte de validación.
    
    Args:
        errors: Lista de errores
    """
    # Agrupar por severidad
    by_severity = defaultdict(list)
    for error in errors:
        by_severity[error.severity].append(error)
    
    # Contar por categoría
    by_category = defaultdict(int)
    for error in errors:
        by_category[error.category] += 1
    
    print("\n" + "="*80)
    print("REPORTE DE VALIDACIÓN DEL ZETTELKASTEN")
    print("="*80 + "\n")
    
    # Resumen
    print("📊 RESUMEN\n")
    print(f"Total de problemas encontrados: {len(errors)}")
    print(f"  - Errores: {len(by_severity['error'])}")
    print(f"  - Advertencias: {len(by_severity['warning'])}")
    print(f"  - Información: {len(by_severity['info'])}")
    print()
    
    # Problemas por categoría
    if by_category:
        print("Por categoría:")
        for category, count in sorted(by_category.items(), key=lambda x: -x[1]):
            print(f"  - {category}: {count}")
        print()
    
    # Detalles de errores
    if by_severity['error']:
        print("="*80)
        print("ERRORES (requieren corrección)")
        print("="*80 + "\n")
        for error in by_severity['error']:
            print(str(error))
            print()
    
    # Detalles de advertencias
    if by_severity['warning']:
        print("="*80)
        print("ADVERTENCIAS (recomendable corregir)")
        print("="*80 + "\n")
        for error in by_severity['warning']:
            print(str(error))
            print()
    
    # Información
    if by_severity['info']:
        print("="*80)
        print("INFORMACIÓN")
        print("="*80 + "\n")
        for error in by_severity['info']:
            print(str(error))
            print()
    
    # Conclusión
    print("="*80)
    if not errors:
        print("✅ ¡Zettelkasten válido! No se encontraron problemas.")
    elif not by_severity['error']:
        print("✅ No se encontraron errores críticos.")
        print("⚠️  Hay advertencias que deberían revisarse.")
    else:
        print("❌ Se encontraron errores que deben corregirse.")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validar la consistencia del Zettelkasten"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Intentar arreglar problemas automáticamente (no implementado aún)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar información detallada"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print("🔍 Recolectando notas del Zettelkasten...")
    
    notes, collection_errors = collect_all_notes()
    
    if args.verbose:
        print(f"   Encontradas {len(notes)} notas")
        print("🔍 Validando enlaces bidireccionales...")
    
    link_errors = validate_bidirectional_links(notes)
    
    if args.verbose:
        print("🔍 Detectando notas huérfanas...")
    
    orphan_errors = validate_orphan_notes(notes)
    
    if args.verbose:
        print("🔍 Validando nombres de archivo...")
    
    filename_errors = validate_filename_consistency(notes)
    
    if args.verbose:
        print("🔍 Validando referencias a categorías...")
    
    category_errors = validate_category_references(notes)
    
    # Combinar todos los errores
    all_errors = (collection_errors + link_errors + orphan_errors + 
                  filename_errors + category_errors)
    
    # Imprimir reporte
    print_validation_report(all_errors)
    
    # Código de salida
    has_errors = any(e.severity == 'error' for e in all_errors)
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()

