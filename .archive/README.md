# Archivo de Código Obsoleto

Este directorio contiene código y documentación que ha sido archivado tras la transición a la arquitectura unificada v8.0.

## Contenido

### old-tracks/
Directorios de tracks individuales del sistema antiguo de desarrollo por tracks separados:
- track-a-core/
- track-b-locales/
- track-c-families/
- track-d-inference/
- track-e-web/
- track-f-desktop/
- track-g-editing/
- track-h-formal-math/
- Archivos .tar.gz de distribución

### old-docs/
Documentación obsoleta del sistema de coordinación de tracks:
- COORDINACION_TRACKS_V3.md
- Analisis_Dependencias_Tracks.md
- TRACK_B_ENTREGABLE_README.md

## Razón del Archivo

Con la arquitectura v8.0, LatticeWeaver ha migrado a una estructura modular unificada donde:

1. **Todos los tracks están integrados** en una única estructura de directorios coherente
2. **Las estrategias son inyectables** mediante interfaces claras
3. **El desarrollo es compatible por diseño** sin necesidad de coordinación manual
4. **La arquitectura es extensible** sin modificar el núcleo

Los directorios de tracks separados ya no son necesarios porque todo el código está ahora organizado por funcionalidad en `lattice_weaver/`.

## Política de Retención

Este código se mantiene archivado por:
- **Referencia histórica**: Para entender la evolución del proyecto
- **Recuperación**: Por si se necesita recuperar algún componente específico
- **Documentación**: Como ejemplo del proceso de refactorización

**No se recomienda** usar este código para desarrollo nuevo. Usar la estructura en `lattice_weaver/` en su lugar.

---

**Fecha de archivo**: 15 de Octubre, 2025  
**Versión archivada**: 7.0  
**Versión actual**: 8.0-alpha
