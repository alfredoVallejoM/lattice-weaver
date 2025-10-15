# Protocolo de Desarrollo para Agentes de LatticeWeaver

**Versión:** 3.0
**Fecha:** 15 de Octubre, 2025
**Propósito:** Establecer un protocolo de desarrollo riguroso para agentes autónomos que garantice la creación de código fiable, robusto y eficiente desde el principio, alineado con los meta-principios de diseño de LatticeWeaver.

---

## 🎯 Objetivo

Este protocolo tiene como objetivo guiar a los agentes en un ciclo de desarrollo que prioriza la planificación y el diseño en profundidad, la implementación de código de alta calidad, el análisis riguroso de errores y la optimización continua, asegurando la coherencia y la integridad del repositorio principal.

---

## 📋 Ciclo de Desarrollo para Agentes

### Fase 1: Planificación y Diseño en Profundidad

1.  **Planificación Detallada de la Tarea**: Antes de escribir cualquier código, el agente debe realizar una planificación en profundidad de la tarea, descomponiéndola en subtareas manejables y estimando el esfuerzo requerido.
2.  **Diseño Acorde a Principios**: El agente debe diseñar una solución que se alinee con los **Meta-Principios de Diseño de LatticeWeaver** (`LatticeWeaver_Meta_Principios_Diseño.md`). Esto incluye la creación de un documento de diseño que justifique las decisiones tomadas y cómo se respetan los principios de eficiencia, modularidad, no redundancia, etc.

### Fase 2: Implementación y Pruebas

1.  **Implementación de Código Funcional**: El agente debe implementar el código de acuerdo con el diseño, asegurando que sea legible, bien documentado y robusto.
2.  **Pruebas Rigurosas**: Se deben desarrollar tests unitarios y de integración con una alta cobertura (>90%) para validar la funcionalidad y prevenir regresiones.
3.  **Revisión de Librerías y Compatibilidad**: Antes de escribir nuevo código, es mandatorio revisar la documentación y el código de las librerías y módulos existentes con los que se va a interactuar. Esto asegura la compatibilidad, el uso correcto de las funciones y APIs (respetando el nombre exacto de las funciones), y previene la duplicación de funcionalidades.

### Fase 3: Análisis de Errores y Refinamiento

1.  **Análisis en Profundidad de Errores**: Si surgen errores durante las pruebas, y después de 2-3 intentos de corrección, el agente debe detenerse y realizar un análisis en profundidad de la causa raíz del problema. Se debe evitar cambiar el código arbitrariamente solo para que los tests pasen, ya que esto puede introducir problemas de dependencias.
2.  **Refinamiento del Código**: Una vez que el código es funcional y pasa los tests, se debe realizar un análisis de los algoritmos para asegurar su eficiencia y proponer mejoras. Se debe evaluar cómo se podría integrar con el resto de la estructura existente de la manera más óptima.

### Política de Resolución de Errores en Testing

Cuando se encuentren errores durante la fase de pruebas, se debe seguir la siguiente política para evitar la introducción de errores catastróficos y mantener la integridad del código base:

1.  **Priorizar el ajuste de los casos de prueba:** Antes de realizar cualquier cambio en el código fuente, se debe analizar si el error reside en el caso de prueba mismo. Es común que los tests no reflejen correctamente la lógica esperada o no estén actualizados. En estos casos, se debe corregir el test para que se alinee con el comportamiento correcto del código existente.

2.  **Modificar el código solo cuando sea necesario:** Solo se debe proceder a modificar el código fuente cuando se haya verificado que los casos de prueba son correctos y que el error revela un problema real en la lógica de la implementación. Esta medida previene cambios innecesarios que puedan desestabilizar otras partes del sistema.

3.  **Análisis de impacto:** Antes de aplicar cualquier corrección en el código, se debe realizar un análisis de impacto para entender cómo el cambio puede afectar a otras librerías o módulos. El objetivo es evitar introducir regresiones o efectos secundarios no deseados.

### Fase 4: Documentación y Actualización del Repositorio

1.  **Documentación Exhaustiva**: El código funcional debe ser debidamente documentado, incluyendo docstrings, comentarios y cualquier otra información relevante para su comprensión y mantenimiento.
2.  **Actualización del Repositorio Principal**: El agente debe actualizar el repositorio principal, asegurando que no se creen problemas de compatibilidad. Esto incluye la actualización de toda la documentación relevante (READMEs, guías, etc.) que se vea afectada por los cambios.

---

## 🚀 Flujo de Trabajo Detallado

```
┌─────────────────────────────────────────────────────────────┐
│ INICIO DE TAREA                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. PLANIFICACIÓN Y DISEÑO EN PROFUNDIDAD                    │
│    - Descomponer tarea en subtareas                         │
│    - Diseñar solución acorde a principios de diseño         │
│    - Crear documento de diseño                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. IMPLEMENTACIÓN Y PRUEBAS                                 │
│    - Revisar librerías existentes                           │
│    - Implementar código funcional                           │
│    - Desarrollar tests con alta cobertura (>90%)            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. EJECUTAR TESTS                                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────┴───────┐
                    │ ¿Tests pasan? │
                    └───────┬───────┘
                            │
                ┌───────────┴───────────┐
                │                       │
               SÍ                      NO
                │                       │
                ▼                       ▼
    ┌─────────────────────┐  ┌──────────────────────────┐
    │ 4A. ANÁLISIS DE       │  │ 4B. ANÁLISIS DE ERRORES    │
    │ ALGORITMOS Y MEJORAS  │  │ (SEGÚN POLÍTICA)         │
    └─────────────────────┘  └──────────────────────────┘
                │                       │
                ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. DOCUMENTACIÓN Y ACTUALIZACIÓN                            │
│    - Documentar código y decisiones de diseño             │
│    - Actualizar repositorio principal y documentación       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ FIN DE TAREA                                                │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ Checklist de Finalización de Tarea

Antes de dar por finalizada una tarea, el agente debe verificar que se han cumplido los siguientes puntos:

*   [ ] Se ha realizado una planificación y diseño en profundidad.
*   [ ] Se ha realizado una revisión de las librerías existentes para asegurar la compatibilidad.
*   [ ] El código implementado es funcional, robusto y está bien documentado.
*   [ ] Se han desarrollado tests con alta cobertura y todos pasan.
*   [ ] Se ha seguido la política de resolución de errores en testing, priorizando el ajuste de los tests.
*   [ ] Se ha realizado un análisis de eficiencia de los algoritmos y se han propuesto mejoras.
*   [ ] Se ha evaluado la integración con el resto de la estructura existente.
*   [ ] Se ha actualizado el repositorio principal y toda la documentación relevante sin crear conflictos.

---

**Este protocolo es de obligado cumplimiento para todos los agentes que contribuyan al desarrollo de LatticeWeaver.**

