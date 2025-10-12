---
id: T###
tipo: tecnica
titulo: [Nombre de la Técnica/Algoritmo]
dominio_origen: [dominio]
categorias_aplicables: [C###, C###]
tags: [tag1, tag2, tag3]
fecha_creacion: YYYY-MM-DD
fecha_modificacion: YYYY-MM-DD
estado: borrador  # borrador | en_revision | completo
implementado: false  # true | false
---

# Técnica: [Nombre de la Técnica/Algoritmo]

## Descripción

[Descripción concisa de la técnica en 1-2 párrafos. Explicar qué hace, para qué sirve y su propósito general.]

## Origen

**Dominio de origen:** [[D###]] - [Nombre del dominio]  
**Año de desarrollo:** [Año]  
**Desarrolladores:** [Nombres de los desarrolladores originales]  
**Contexto:** [Descripción del problema original que motivó el desarrollo de esta técnica]

## Formulación

### Entrada

[Descripción de qué tipo de datos/estructuras recibe la técnica]

- **Tipo 1:** [Descripción]
- **Tipo 2:** [Descripción]

### Salida

[Descripción de qué produce la técnica]

- **Tipo 1:** [Descripción]
- **Tipo 2:** [Descripción]

### Parámetros

[Lista de parámetros configurables de la técnica]

| Parámetro | Tipo | Rango | Descripción | Valor por defecto |
|-----------|------|-------|-------------|-------------------|
| [param1] | [tipo] | [rango] | [desc] | [default] |
| [param2] | [tipo] | [rango] | [desc] | [default] |

## Algoritmo

### Pseudocódigo

```
ALGORITMO [NombreTecnica](entrada, parámetros)
    ENTRADA: [descripción]
    SALIDA: [descripción]
    
    1. [Paso 1]
    2. [Paso 2]
    3. MIENTRAS [condición] HACER
        3.1. [Subpaso]
        3.2. [Subpaso]
    4. RETORNAR [resultado]
FIN ALGORITMO
```

### Descripción Paso a Paso

1. **[Paso 1]:** [Explicación detallada de qué hace este paso y por qué]
2. **[Paso 2]:** [Repetir]
3. **[Paso N]:** [Repetir]

### Invariantes

[Propiedades que se mantienen durante la ejecución del algoritmo]

1. **[Invariante 1]:** [Descripción]
2. **[Invariante 2]:** [Repetir]

## Análisis

### Complejidad Temporal

- **Mejor caso:** O([complejidad])
- **Caso promedio:** O([complejidad])
- **Peor caso:** O([complejidad])

**Justificación:** [Explicación del análisis de complejidad]

### Complejidad Espacial

- **Espacio auxiliar:** O([complejidad])
- **Espacio total:** O([complejidad])

**Justificación:** [Explicación]

### Corrección

[Argumento de por qué el algoritmo es correcto]

**Teorema:** [Enunciado formal de corrección]  
**Demostración:** [Sketch de la demostración o referencia]

### Optimalidad

[¿Es este algoritmo óptimo? ¿Hay cotas inferiores conocidas?]

## Aplicabilidad

### Categorías Estructurales Aplicables

[Lista de categorías donde esta técnica es aplicable]

1. [[C###]] - [Nombre de la categoría]
   - **Por qué funciona:** [Explicación]
   - **Limitaciones:** [Cuándo no funciona bien]

2. [[C###]] - [Repetir]

### Fenómenos Donde Se Ha Aplicado

#### En Dominio Original

- [[F###]] - [Nombre del fenómeno]
  - **Resultado:** [Descripción del resultado de aplicar la técnica]
  - **Referencias:** [Papers relevantes]

#### Transferencias a Otros Dominios

- [[F###]] - [Nombre del fenómeno en otro dominio]
  - **Adaptaciones necesarias:** [Descripción]
  - **Resultado:** [Descripción del resultado]
  - **Referencias:** [Papers relevantes]

- [[F###]] - [Repetir]

### Prerequisitos

[Qué debe cumplir un problema para que esta técnica sea aplicable]

1. **[Prerequisito 1]:** [Descripción]
2. **[Prerequisito 2]:** [Repetir]

### Contraindicaciones

[Situaciones donde esta técnica NO debe usarse]

1. **[Contraindicación 1]:** [Descripción]
2. **[Contraindicación 2]:** [Repetir]

## Variantes

### Variante 1: [Nombre de la Variante]

**Modificación:** [Qué cambia respecto al algoritmo base]  
**Ventaja:** [Qué mejora]  
**Desventaja:** [Qué empeora]  
**Cuándo usar:** [Situaciones apropiadas]

### Variante 2: [Nombre de la Variante]
[Repetir estructura]

## Comparación con Técnicas Alternativas

### Técnica Alternativa 1: [[T###]] - [Nombre]

| Criterio | Esta Técnica | Técnica Alternativa |
|----------|--------------|---------------------|
| Complejidad temporal | [valor] | [valor] |
| Complejidad espacial | [valor] | [valor] |
| Facilidad de implementación | [valoración] | [valoración] |
| Calidad de solución | [valoración] | [valoración] |
| Aplicabilidad | [descripción] | [descripción] |

**Cuándo preferir esta técnica:** [Descripción]  
**Cuándo preferir la alternativa:** [Descripción]

### Técnica Alternativa 2: [[T###]] - [Nombre]
[Repetir estructura]

## Ejemplos de Uso

### Ejemplo 1: [Nombre del Ejemplo]

**Contexto:** [Descripción del problema específico]

**Entrada:**
```
[Datos de entrada específicos]
```

**Ejecución:**
```
[Traza de la ejecución paso a paso]
```

**Salida:**
```
[Resultado obtenido]
```

**Análisis:** [Explicación de por qué la técnica funcionó bien/mal en este caso]

### Ejemplo 2: [Nombre del Ejemplo]
[Repetir estructura]

## Implementación

### En LatticeWeaver

**Módulo:** `lattice_weaver/algorithms/[nombre_tecnica]/`

**Interfaz:**
```python
def [nombre_tecnica](
    input_data: [Tipo],
    param1: [Tipo] = [default],
    param2: [Tipo] = [default],
    **kwargs
) -> [TipoRetorno]:
    """
    [Docstring describiendo la función]
    
    Args:
        input_data: [Descripción]
        param1: [Descripción]
        param2: [Descripción]
        **kwargs: Parámetros adicionales
    
    Returns:
        [Descripción del retorno]
    
    Raises:
        [Excepciones que puede lanzar]
    
    Examples:
        >>> [ejemplo de uso]
    """
    pass
```

### Dependencias

[Librerías o módulos necesarios]

- `[libreria1]` - [Propósito]
- `[libreria2]` - [Propósito]

### Tests

**Ubicación:** `tests/algorithms/test_[nombre_tecnica].py`

**Casos de test:**
1. Test de corrección con entrada simple
2. Test de casos borde
3. Test de complejidad (performance)
4. Test de robustez (entradas inválidas)

## Visualización

### Visualización de la Ejecución

[Descripción de cómo visualizar la ejecución del algoritmo]

**Tipo de visualización:** [Animación | Diagrama estático | Gráfico | ...]

**Componentes:**
- [Componente 1 a visualizar]
- [Componente 2 a visualizar]

### Visualización de Resultados

[Descripción de cómo visualizar los resultados]

## Recursos

### Literatura Clave

#### Paper Original
[Autor(es)]. ([Año]). *[Título]*. [Journal/Editorial].

#### Análisis y Mejoras
1. [Referencia de paper que analiza o mejora la técnica]
2. [Repetir]

#### Aplicaciones
1. [Referencia de paper que aplica la técnica a un dominio específico]
2. [Repetir]

### Implementaciones Existentes

- **[Librería/Proyecto]:** [URL]
  - **Lenguaje:** [lenguaje]
  - **Licencia:** [licencia]
  - **Notas:** [Observaciones sobre la implementación]

- **[Repetir]**

### Tutoriales y Recursos Educativos

- **[Título del recurso]:** [URL] - [Descripción breve]
- **[Repetir]**

## Conexiones

### Técnicas Relacionadas

- [[T###]] - [Nombre de técnica relacionada]
  - **Relación:** [Cómo se relacionan: generalización, especialización, complementaria, etc.]

- [[T###]] - [Repetir]

### Conceptos Fundamentales

- [[K###]] - [Concepto en el que se basa la técnica]
- [[K###]] - [Repetir]

### Fenómenos Aplicables

- [[F###]] - [Fenómeno donde se aplica]
- [[F###]] - [Repetir]

## Historia y Evolución

### Desarrollo Histórico

[Cronología del desarrollo de la técnica]

- **[Año]:** [Evento o mejora]
- **[Año]:** [Evento o mejora]

### Impacto

[Descripción del impacto de esta técnica en su campo y en otros campos]

**Citaciones:** [Número aproximado de citaciones del paper original]  
**Adopción:** [Descripción de cuán ampliamente se usa]

## Estado de Implementación

- [ ] Pseudocódigo documentado
- [ ] Análisis de complejidad completado
- [ ] Implementación en Python
- [ ] Tests unitarios
- [ ] Tests de performance
- [ ] Documentación de API
- [ ] Ejemplos de uso
- [ ] Visualización de ejecución
- [ ] Tutorial

## Notas Adicionales

### Ideas para Mejora

- [Idea 1]
- [Idea 2]

### Preguntas Abiertas

- [Pregunta 1]
- [Pregunta 2]

### Observaciones

[Cualquier observación relevante]

---

**Última actualización:** YYYY-MM-DD  
**Responsable:** [Nombre del agente/persona]

