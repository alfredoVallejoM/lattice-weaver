# Índice de Documentación de LatticeWeaver

Este documento sirve como un punto de entrada centralizado para la documentación del proyecto LatticeWeaver, facilitando la navegación a través de sus componentes y principios.

---

## Documentos Clave del Proyecto

*   [**README.md**](../README.md): Visión general de alto nivel del proyecto, sus objetivos y estado actual.
*   [**Visión General del Proyecto**](PROJECT_OVERVIEW.md): Resumen ejecutivo, arquitectura modular, hoja de ruta estratégica y protocolo de colaboración.
*   [**Protocolo de Agentes**](../PROTOCOLO_AGENTES_LATTICEWEAVER.md): Guía detallada para el desarrollo, incluyendo ciclo de vida de tareas, diseño, implementación, pruebas y actualización del repositorio.
*   [**Principios de Diseño Maestro**](../MASTER_DESIGN_PRINCIPLES.md): Documento que establece los meta-principios de diseño fundamentales que rigen todo el desarrollo en LatticeWeaver.

---

## Documentación por Módulos

Aquí se listan los módulos principales del proyecto, con enlaces a sus archivos `__init__.py` (que a menudo contienen docstrings de alto nivel) o a sus archivos `core.py` para una visión general de su funcionalidad.

*   **`core`**: [lattice_weaver/core/__init__.py](../lattice_weaver/core/__init__.py)
    *   [csp_problem.py](../lattice_weaver/core/csp_problem.py)
*   **`formal`**: [lattice_weaver/formal/__init__.py](../lattice_weaver/formal/__init__.py)
    *   [cubical_engine.py](../lattice_weaver/formal/cubical_engine.py)
    *   [csp_cubical_bridge.py](../lattice_weaver/formal/csp_cubical_bridge.py)
*   **`renormalization`**: [lattice_weaver/renormalization/__init__.py](../lattice_weaver/renormalization/__init__.py)
    *   [core.py](../lattice_weaver/renormalization/core.py)
    *   [partition.py](../lattice_weaver/renormalization/partition.py)
*   **`paging`**: [lattice_weaver/paging/__init__.py](../lattice_weaver/paging/__init__.py)
    *   [page_manager.py](../lattice_weaver/paging/page_manager.py)
*   **`fibration`**: [lattice_weaver/fibration/__init__.py](../lattice_weaver/fibration/__init__.py)
    *   [energy_landscape_optimized.py](../lattice_weaver/fibration/energy_landscape_optimized.py)
*   **`ml`**: [lattice_weaver/ml/__init__.py](../lattice_weaver/ml/__init__.py)
    *   [mini_nets/renormalization.py](../lattice_weaver/ml/mini_nets/renormalization.py)
*   **`compiler_multiescala`**: [lattice_weaver/compiler_multiescala/__init__.py](../lattice_weaver/compiler_multiescala/__init__.py)
*   **`validation`**: [lattice_weaver/validation/__init__.py](../lattice_weaver/validation/__init__.py)

---

## Tracks de Investigación y Desarrollo

*   [**Track D: Inference Engine Design**](TRACK_D_INFERENCE_ENGINE_DESIGN.md)
*   [**Track I: Educational Multidisciplinary**](../track-i-educational-multidisciplinary/README.md)

---

**Nota:** Esta documentación se actualizará continuamente a medida que el proyecto evolucione.
