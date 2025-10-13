---
id: K006
tipo: concepto
titulo: Lógica Proposicional
dominios: [logica, filosofia, informatica]
categorias: [C006] # Satisfacibilidad Lógica
tags: [proposiciones, conectivas, tablas_de_verdad, validez]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo
prioridad: alta
---

# Lógica Proposicional

## Descripción

La **Lógica Proposicional** (o Lógica de Enunciados) es la rama más básica de la lógica formal que estudia las relaciones entre proposiciones sin considerar su estructura interna. Utiliza variables proposicionales (ej., P, Q, R) para representar enunciados y conectivas lógicas (ej., AND, OR, NOT, IMPLICA, EQUIVALENCIA) para combinarlas y formar fórmulas más complejas. Su principal herramienta son las tablas de verdad para determinar la validez de los argumentos y la satisfacibilidad de las fórmulas [15].

## Componentes Clave

-   **Proposiciones Atómicas:** Enunciados simples que pueden ser verdaderos o falsos.
-   **Conectivas Lógicas:** Operadores que combinan proposiciones (¬, ∧, ∨, →, ↔).
-   **Tablas de Verdad:** Método para evaluar el valor de verdad de fórmulas complejas.

## Mapeo a Formalismos

### CSP

-   **Variables:** Cada proposición atómica puede ser una variable en un CSP.
-   **Dominios:** `{Verdadero, Falso}` para cada variable.
-   **Restricciones:** Las conectivas lógicas se traducen directamente en restricciones. Por ejemplo, `P AND Q` se convierte en una restricción que requiere que `P` y `Q` sean ambos `Verdadero` para que la conjunción sea `Verdadero`. El `AdaptiveConsistencyEngine` de LatticeWeaver es ideal para resolver la satisfacibilidad de fórmulas proposicionales.

## Conexiones

### Instancia de
- [[F011]] - Lógica y Argumentación (Filosofía)

### Conceptos Relacionados
- [[K007]] - Lógica de Predicados
- [[K009]] - Validez Lógica

## Recursos

### Literatura Clave
1.  Copi, I. M., & Cohen, C. (2002). *Introducción a la lógica*. Limusa.

## Estado de Implementación

- [x] Investigación completada
- [ ] Diseño de mapeo
- [ ] Implementación
- [ ] Visualización
- [ ] Documentación
- [ ] Tutorial

## Notas Adicionales

La lógica proposicional es la base para entender sistemas lógicos más complejos y es fundamental en la informática teórica.


### Instancia de
- [[F011]] - Lógica y Argumentación (Filosofía)



### Isomorfismos Inversos
- [[I008]] - Isomorfismo entre Marcos de Argumentación Abstractos (AFs) y CSP



### Categorías Inversas
- [[C006]] - Satisfacibilidad Lógica



### Isomorfismos Inversos
- [[I008]] - Isomorfismo entre Marcos de Argumentación Abstractos (AFs) y CSP



### Conceptos Inversos
- [[K016]] - Lógica Paraconsistente
- [[K017]] - Lógica Relevante
- [[K018]] - Lógica Dialógica y Juegos de Diálogo
- [[K008]] - Inferencia



### Conceptos Inversos
- [[K012]] - Lógica Modal



### Conceptos Inversos
- [[K016]] - Lógica Paraconsistente



- [[K012]] - Lógica Modal



### Conceptos Inversos
- [[K012]] - Lógica Modal



### Conceptos Inversos
- [[K014]] - Lógica Temporal



### Conceptos Inversos
- [[K015]] - Lógica Difusa (Fuzzy Logic)



### Fenómenos Inversos
- [[F013]] - Marcos de Argumentación Basados en Lógica



### Fenómenos Inversos
- [[F015]] - Marcos de Argumentación Basados en Supuestos (ABAs)



### Conceptos Inversos
- [[K014]] - Lógica Temporal



### Conceptos Inversos
- [[K015]] - Lógica Difusa (Fuzzy Logic)

