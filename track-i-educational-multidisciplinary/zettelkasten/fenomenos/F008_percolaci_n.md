---
id: F008
tipo: fenomeno
titulo: Percolación
dominios: [fisica_estadistica, ciencia_materiales, ecologia, epidemiologia, redes]
categorias: [C001, C004]
tags: [transiciones_de_fase, conectividad, redes_aleatorias, criticalidad, robustez, vulnerabilidad]
fecha_creacion: 2025-10-12
fecha_modificacion: 2025-10-13
estado: completo  # borrador | en_revision | completo
prioridad: media  # maxima | alta | media | baja
---

# Percolación

## Descripción

La **Teoría de Percolación** es un modelo matemático que describe el comportamiento de la conectividad en sistemas aleatorios. Fue introducida por Broadbent y Hammersley en 1957 para modelar el flujo de fluidos a través de medios porosos. En su forma más simple, considera una red (o un retículo) donde los nodos o las aristas se activan (o se eliminan) con una cierta probabilidad. El fenómeno central de la percolación es la aparición de un "clúster percolante" o "componente gigante" que se extiende a través de todo el sistema cuando la probabilidad de activación supera un umbral crítico. Este umbral, conocido como **umbral de percolación**, marca una transición de fase.

La percolación es un concepto fundamental en la física estadística y la ciencia de redes, ya que proporciona un marco para entender cómo la conectividad y la robustez de un sistema cambian drásticamente con pequeñas variaciones en sus componentes. Sus aplicaciones abarcan desde la conductividad eléctrica en materiales compuestos y la propagación de enfermedades en poblaciones, hasta la estabilidad de ecosistemas y la vulnerabilidad de infraestructuras críticas. Es un excelente ejemplo de cómo un modelo simple puede capturar la esencia de fenómenos complejos de gran escala.

## Componentes Clave

### Variables
-   **Red/Retículo (L):** Un conjunto de sitios (nodos) y/o enlaces (aristas).
-   **Probabilidad (p):** La probabilidad de que un sitio o un enlace esté "abierto" o "activo".
-   **Sitios/Enlaces Abiertos/Cerrados:** El estado binario de cada elemento de la red.
-   **Clúster:** Un conjunto de sitios/enlaces abiertos conectados entre sí.
-   **Tamaño del Clúster (S):** El número de sitios/enlaces en un clúster.
-   **Fracción del Clúster Gigante (P_inf):** La proporción de sitios/enlaces que pertenecen al clúster percolante.

### Dominios
-   **Dominio de p:** [0, 1] (número real).
-   **Dominio de Sitios/Enlaces:** {Abierto, Cerrado} o {1, 0}.
-   **Dominio de S:** Enteros no negativos.
-   **Dominio de P_inf:** [0, 1] (número real).

### Restricciones/Relaciones
-   **Conectividad Aleatoria:** Los sitios/enlaces se activan de forma independiente con probabilidad `p`.
-   **Definición de Clúster:** Dos sitios/enlaces abiertos pertenecen al mismo clúster si existe un camino de sitios/enlaces abiertos entre ellos.

### Función Objetivo (si aplica)
-   No hay una función objetivo a optimizar en el sentido tradicional. El interés radica en estudiar las **propiedades emergentes** del sistema (ej. tamaño del clúster gigante, conductividad) en función de `p`.

## Mapeo a Formalismos

### CSP (Constraint Satisfaction Problem)
-   **Variables:** El estado binario (abierto/cerrado) de cada sitio/enlace.
-   **Dominios:** {Abierto, Cerrado} para cada sitio/enlace.
-   **Restricciones:** Las restricciones no son fijas, sino que emergen de la conectividad aleatoria. Por ejemplo, si se busca un camino entre dos puntos, la existencia de ese camino es una restricción que debe satisfacerse por la configuración de sitios/enlaces abiertos.
-   **Tipo:** Satisfacción (determinar si existe un camino percolante entre dos regiones, o si un clúster gigante existe).

### Sistemas Dinámicos (Estocásticos)
-   **Espacio de Estados:** El conjunto de todas las posibles configuraciones de sitios/enlaces abiertos/cerrados.
-   **Dinámica:** La "dinámica" no es temporal, sino que se refiere a cómo las propiedades del sistema cambian a medida que `p` varía (parámetro de control). La aleatoriedad introduce un componente estocástico.
-   **Transición de Fase:** El umbral de percolación `p_c` es un punto crítico donde el sistema sufre una transición de fase de un estado no percolante a uno percolante.

## Ejemplos Concretos

### Ejemplo 1: Conductividad Eléctrica en Materiales Compuestos
**Contexto:** Un material compuesto está formado por partículas conductoras dispersas en una matriz aislante. ¿Cuándo el material se vuelve conductor?

**Mapeo:**
-   Sitios/Enlaces = Partículas conductoras o conexiones entre ellas.
-   Probabilidad `p` = Fracción de partículas conductoras o probabilidad de que una conexión sea conductora.

**Solución esperada:** Cuando `p` supera el umbral de percolación, se forma un camino conductor a través del material, permitiendo el flujo de corriente.

**Referencias:** Stauffer, D., & Aharony, A. (1994). *Introduction to Percolation Theory* (2nd ed.). Taylor & Francis.

### Ejemplo 2: Propagación de Epidemias
**Contexto:** Modelar la propagación de una enfermedad en una población donde los individuos tienen contactos aleatorios.

**Mapeo:**
-   Nodos = Individuos.
-   Aristas = Contactos entre individuos.
-   Probabilidad `p` = Probabilidad de transmisión de la enfermedad a través de un contacto, o probabilidad de que un individuo sea susceptible/infectado.

**Solución esperada:** Si la probabilidad de transmisión `p` supera un umbral crítico, la enfermedad puede percolar a través de la población, causando una epidemia a gran escala.

**Referencias:** Newman, M. E. J. (2002). "Spread of epidemic disease on networks". *Physical Review E*, 66(1), 016128.

### Ejemplo 3: Robustez de Redes de Infraestructura
**Contexto:** Evaluar la vulnerabilidad de redes como Internet, redes eléctricas o sistemas de transporte ante fallos aleatorios o ataques dirigidos.

**Mapeo:**
-   Nodos/Enlaces = Componentes de la infraestructura (routers, centrales eléctricas, carreteras).
-   Probabilidad `p` = Probabilidad de que un componente funcione correctamente (1-p es la probabilidad de fallo).

**Solución esperada:** Si `p` cae por debajo de un umbral crítico, la red puede fragmentarse y perder su funcionalidad global, incluso si muchos componentes individuales siguen funcionando.

**Referencias:** Albert, R., Barabási, A. L., & Jeong, H. (2000). "Error and attack tolerance of complex networks". *Nature*, 406(6794), 378-382.

## Conexiones

#- [[F008]] - Conexión inversa con Fenómeno.
- [[F008]] - Conexión inversa con Fenómeno.
- [[F008]] - Conexión inversa con Fenómeno.
- [[F008]] - Conexión inversa con Fenómeno.
- [[F008]] - Conexión inversa con Fenómeno.
- [[F008]] - Conexión inversa con Fenómeno.
- [[F008]] - Conexión inversa con Fenómeno.
- [[F008]] - Conexión inversa con Fenómeno.
- [[F008]] - Conexión inversa con Fenómeno.
## Categoría Estructural
-   [[C001]] - Redes de Interacción: La percolación es un estudio de la conectividad y robustez de las redes.
-   [[C004]] - Sistemas Dinámicos: La transición de fase en percolación es un ejemplo de comportamiento crítico en sistemas complejos.

### Conexiones Inversas
-   [[C001]] - Redes de Interacción (instancia)
-   [[C004]] - Sistemas Dinámicos (instancia)

#- [[F008]] - Conexión inversa con Fenómeno.
## Isomorfismos
-   [[I###]] - Percolación ≅ Transiciones de Fase de Segundo Orden (en física estadística, comparten exponentes críticos universales).
-   [[I###]] - Percolación ≅ Problema de Conectividad en Grafos Aleatorios (Erdos-Renyi).

### Instancias en Otros Dominios
-   [[F003]] - Modelo de Ising 2D: Ambos exhiben transiciones de fase y fenómenos críticos.
-   [[F009]] - Modelo de votantes: La propagación de opiniones puede verse como un proceso de percolación en una red social.

### Técnicas Aplicables
-   [[T###]] - Simulación de Monte Carlo (para estudiar el comportamiento de percolación en redes grandes).
-   [[T###]] - Teoría de Grafos Aleatorios (para analizar propiedades de conectividad).
-   [[T###]] - Renormalización (para estudiar el comportamiento crítico cerca del umbral de percolación).

### Conceptos Fundamentales
-   [[K###]] - Transiciones de Fase
-   [[K###]] - Fenómenos Críticos
-   [[K###]] - Exponentes Críticos
-   [[K###]] - Redes Aleatorias

### Prerequisitos
-   [[K###]] - Teoría de Probabilidad Básica
-   [[K###]] - Teoría de Grafos Básica
-   [[K###]] - Conceptos de Física Estadística

## Propiedades Matemáticas

### Complejidad Computacional
-   **Simulación:** La simulación de percolación en retículos o redes grandes puede ser computacionalmente intensiva, especialmente cerca del umbral crítico.
-   **Determinación del Umbral:** Para retículos regulares, el umbral de percolación `p_c` puede ser conocido analíticamente (ej. `p_c = 0.5` para percolación de enlaces en un retículo cuadrado 2D). Para redes complejas, a menudo se determina mediante simulación.

### Propiedades Estructurales
-   **Umbral de Percolación (p_c):** El valor crítico de `p` donde la probabilidad de que exista un clúster percolante se vuelve no nula.
-   **Exponenciales Críticos:** Cerca de `p_c`, varias propiedades del sistema (ej. tamaño del clúster gigante, longitud de correlación) exhiben un comportamiento de ley de potencias, caracterizado por exponentes críticos universales.

### Teoremas Relevantes
-   **Teorema de Kesten (1982):** Para percolación de enlaces en un retículo cuadrado 2D, el umbral de percolación es exactamente `p_c = 1/2`.

## Visualización

### Tipos de Visualización Aplicables
1.  **Retículos/Redes Coloreadas:** Mostrar los sitios/enlaces abiertos y cerrados, y resaltar los clústeres, especialmente el clúster percolante.
2.  **Gráficos de P_inf vs p:** Mostrar la curva de la fracción del clúster gigante en función de la probabilidad `p`, revelando la transición de fase.
3.  **Animaciones:** Simular el proceso de adición aleatoria de sitios/enlaces y la formación de clústeres.

### Componentes Reutilizables
-   Componentes de visualización de grafos/retículos.
-   Generadores de números aleatorios.
-   Funciones para encontrar componentes conectados.

## Recursos

### Literatura Clave
1.  Stauffer, D., & Aharony, A. (1994). *Introduction to Percolation Theory* (2nd ed.). Taylor & Francis.
2.  Sahimi, M. (1994). *Applications of Percolation Theory*. CRC Press.
3.  Newman, M. E. J. (2010). *Networks: An Introduction*. Oxford University Press.

### Datasets
-   **Retículos 2D/3D:** Datos de simulaciones de percolación en diferentes geometrías.
-   **Redes complejas:** Datos de redes sociales, biológicas o tecnológicas para estudiar su robustez.

### Implementaciones Existentes
-   **SciPy (Python):** Funciones para encontrar componentes conectados en grafos.
-   **NetworkX (Python):** Para la manipulación y análisis de redes.

### Código en LatticeWeaver
-   **Módulo:** `lattice_weaver/phenomena/percolation/`
-   **Tests:** `tests/phenomena/test_percolation.py`
-   **Documentación:** `docs/phenomena/percolation.md`

## Estado de Implementación

### Fase 1: Investigación
-   [x] Revisión bibliográfica completada
-   [x] Ejemplos concretos identificados
-   [x] Datasets recopilados (referenciados)
-   [ ] Documento de investigación creado (integrado aquí)

### Fase 2: Diseño
-   [x] Mapeo a CSP diseñado
-   [x] Mapeo a otros formalismos (Sistemas Dinámicos Estocásticos)
-   [ ] Arquitectura de código planificada
-   [ ] Visualizaciones diseñadas

### Fase 3: Implementación
-   [ ] Clases base implementadas
-   [ ] Algoritmos implementados
-   [ ] Tests unitarios escritos
-   [ ] Tests de integración escritos

### Fase 4: Visualización
-   [ ] Componentes de visualización implementados
-   [ ] Visualizaciones interactivas creadas
-   [ ] Exportación de visualizaciones

### Fase 5: Documentación
-   [ ] Documentación de API
-   [ ] Tutorial paso a paso
-   [ ] Ejemplos de uso
-   [ ] Casos de estudio

### Fase 6: Validación
-   [ ] Revisión por pares
-   [ ] Validación con expertos del dominio
-   [ ] Refinamiento basado en feedback

## Estimaciones

-   **Tiempo de investigación:** 20 horas
-   **Tiempo de diseño:** 10 horas
-   **Tiempo de implementación:** 30 horas
-   **Tiempo de visualización:** 15 horas
-   **Tiempo de documentación:** 10 horas
-   **TOTAL:** 85 horas

## Notas Adicionales

### Ideas para Expansión
-   Estudiar percolación en redes complejas (ej. redes libres de escala).
-   Explorar la relación entre percolación y robustez de redes.
-   Aplicaciones en la formación de clusters en datos.

### Preguntas Abiertas
-   ¿Cómo se puede predecir el umbral de percolación en redes arbitrarias?
-   ¿Qué papel juega la percolación en la conciencia o la emergencia de propiedades en sistemas biológicos?

### Observaciones
-   La percolación es un modelo simple que revela la complejidad y la universalidad de las transiciones de fase en sistemas desordenados.

---

**Última actualización:** 2025-10-13
**Responsable:** Agente Autónomo de Análisis
- [[C001]]
- [[C004]]
- [[F009]]
- [[I006]]
- [[I008]]
- [[T003]]
- [[K006]]
- [[K007]]
- [[F003]]
