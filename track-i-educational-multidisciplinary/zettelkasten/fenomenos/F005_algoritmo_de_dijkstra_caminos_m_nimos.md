---
id: F005
tipo: fenomeno
titulo: Algoritmo de Dijkstra / Caminos mínimos
dominios: [informatica, matematicas, logistica, transporte, redes]
categorias: [C001, C003]
tags: [grafos, algoritmos, optimizacion, caminos_cortos, greedy, redes_de_transporte]
fecha_creacion: 2025-10-12
fecha_modificacion: 2025-10-13
estado: completo  # borrador | en_revision | completo
prioridad: media  # maxima | alta | media | baja
---

# Algoritmo de Dijkstra / Caminos mínimos

## Descripción

El **Algoritmo de Dijkstra** es un algoritmo de búsqueda de ruta única que resuelve el problema del camino más corto para un grafo con pesos de arista no negativos. Fue concebido por Edsger W. Dijkstra en 1956 y publicado en 1959. Su objetivo es encontrar el camino de menor costo (o distancia) desde un nodo fuente dado a todos los demás nodos en el grafo. Es un algoritmo "greedy" que construye el camino más corto paso a paso, seleccionando en cada iteración el nodo no visitado con la distancia más pequeña conocida desde la fuente.

Este algoritmo es fundamental en ciencias de la computación, matemáticas aplicadas y logística, siendo la base para numerosas aplicaciones prácticas. Su importancia radica en su eficiencia y en la garantía de encontrar la solución óptima para su clase de problemas. Aunque existen variantes más rápidas para casos específicos (como el algoritmo A* que incorpora heurísticas), Dijkstra sigue siendo un pilar para entender la optimización en redes.

## Componentes Clave

### Variables
-   **Grafo (G):** Un conjunto de nodos (V) y aristas (E).
-   **Nodos (v ∈ V):** Puntos en el grafo, representando ubicaciones, estados, etc.
-   **Aristas (e ∈ E):** Conexiones entre nodos, representando rutas, transiciones, etc.
-   **Peso de Arista (w(e)):** Un valor no negativo asociado a cada arista, representando costo, distancia, tiempo, etc.
-   **Nodo Fuente (s):** El nodo desde el cual se calculan los caminos más cortos.
-   **Distancia (d[v]):** La distancia más corta conocida desde el nodo fuente `s` hasta el nodo `v`.
-   **Predecesor (π[v]):** El nodo que precede a `v` en el camino más corto desde `s`.
-   **Conjunto de Nodos Visitados (S):** Nodos para los cuales ya se ha determinado la distancia más corta final.
-   **Cola de Prioridad (Q):** Estructura de datos que almacena los nodos no visitados, priorizados por su distancia actual `d[v]`.

### Dominios
-   **Dominio de Nodos:** Conjunto finito de identificadores únicos.
-   **Dominio de Pesos de Arista:** Números reales no negativos (w(e) ≥ 0).
-   **Dominio de Distancia:** Números reales no negativos.

### Restricciones/Relaciones
-   **Pesos de Arista No Negativos:** `w(e) ≥ 0` para todas las aristas. Esta es una restricción crucial para la validez de Dijkstra.
-   **Conectividad:** El algoritmo asume que el grafo es conexo o que se buscan caminos dentro de componentes conexos.
-   **Relajación de Aristas:** La operación fundamental donde se actualiza la distancia a un nodo si se encuentra un camino más corto a través de un nodo adyacente.
    -   `if d[u] + w(u,v) < d[v]: d[v] = d[u] + w(u,v)`

### Función Objetivo
-   **Minimizar la suma de los pesos de las aristas** a lo largo de un camino desde el nodo fuente a cualquier otro nodo.

## Mapeo a Formalismos

### CSP (Constraint Satisfaction Problem)
-   **Variables:** Para cada nodo `v`, una variable `d_v` que representa su distancia desde la fuente, y una variable `p_v` que representa su predecesor.
-   **Dominios:** `d_v` tiene como dominio los números reales no negativos; `p_v` tiene como dominio los nodos del grafo.
-   **Restricciones:**
    -   `d_s = 0` (distancia a la fuente es cero).
    -   Para cada arista `(u,v)` con peso `w(u,v)`, `d_v ≤ d_u + w(u,v)`.
    -   Si `p_v = u`, entonces `d_v = d_u + w(u,v)`.
-   **Tipo:** Optimización (encontrar la asignación de `d_v` que minimice las distancias).

### Sistemas Dinámicos (Discretos)
-   **Espacio de Estados:** El conjunto de todas las posibles asignaciones de distancias `d[v]` y predecesores `π[v]` en cada iteración.
-   **Dinámica:** Cada paso del algoritmo (selección de nodo, relajación de aristas) es una transición de estado discreta.
-   **Atractor:** El estado final donde todas las distancias son las más cortas y no se pueden realizar más relajaciones es el atractor del sistema.

## Ejemplos Concretos

### Ejemplo 1: Rutas de Navegación GPS
**Contexto:** Encontrar la ruta más corta (en distancia o tiempo) entre dos puntos en un mapa de carreteras.

**Mapeo:**
-   Nodos = Intersecciones de carreteras o puntos de interés.
-   Aristas = Segmentos de carretera.
-   Pesos de Arista = Distancia física, tiempo de viaje (considerando tráfico), o costo de peaje.

**Solución esperada:** El algoritmo de Dijkstra (o sus variantes como A*) calcula la secuencia óptima de segmentos de carretera para llegar al destino.

**Referencias:** Aplicación estándar en sistemas de navegación.

### Ejemplo 2: Enrutamiento de Paquetes en Redes de Computadoras
**Contexto:** Determinar el camino más eficiente para que los paquetes de datos viajen desde un origen a un destino a través de una red de routers.

**Mapeo:**
-   Nodos = Routers.
-   Aristas = Enlaces de red.
-   Pesos de Arista = Latencia del enlace, ancho de banda, costo administrativo.

**Solución esperada:** Dijkstra se utiliza en protocolos de enrutamiento (ej. OSPF) para construir tablas de enrutamiento que dirigen el tráfico por los caminos más cortos.

**Referencias:** Tanenbaum, A. S., & Wetherall, D. J. (2011). *Computer Networks* (5th ed.). Pearson Education.

### Ejemplo 3: Planificación de Proyectos (CPM/PERT)
**Contexto:** Encontrar el camino crítico en un diagrama de red de actividades de un proyecto para determinar la duración mínima del proyecto.

**Mapeo:**
-   Nodos = Eventos o hitos del proyecto.
-   Aristas = Actividades del proyecto.
-   Pesos de Arista = Duración de cada actividad.

**Solución esperada:** Aunque CPM/PERT usa variantes, la lógica subyacente de encontrar el camino más largo (o el camino crítico) es análoga a encontrar el camino más corto en un grafo modificado (negando pesos o invirtiendo el problema).

**Referencias:** Moder, J. J., Phillips, C. R., & Davis, E. W. (1983). *Project Management with CPM, PERT and Precedence Diagramming*. Van Nostrand Reinhold.

## Conexiones

#- [[F005]] - Conexión inversa con Fenómeno.
- [[F005]] - Conexión inversa con Fenómeno.
- [[F005]] - Conexión inversa con Fenómeno.
- [[F005]] - Conexión inversa con Fenómeno.
- [[F005]] - Conexión inversa con Fenómeno.
- [[F005]] - Conexión inversa con Fenómeno.
- [[F005]] - Conexión inversa con Fenómeno.
- [[F005]] - Conexión inversa con Fenómeno.
- [[F005]] - Conexión inversa con Fenómeno.
- [[F005]] - Conexión inversa con Fenómeno.
- [[F005]] - Conexión inversa con Fenómeno.
- [[F005]] - Conexión inversa con Fenómeno.
- [[F005]] - Conexión inversa con Fenómeno.
- [[F005]] - Conexión inversa con Fenómeno.
- [[F005]] - Conexión inversa con Fenómeno.
## Categoría Estructural
-   [[C001]] - Redes de Interacción: El algoritmo opera directamente sobre estructuras de red (grafos).
-   [[C003]] - Optimización con Restricciones: Busca la solución óptima (camino más corto) bajo la restricción de los pesos de las aristas y la estructura del grafo.

### Conexiones Inversas
-   [[C001]] - Redes de Interacción (instancia)
-   [[C003]] - Optimización con Restricciones (instancia)

#- [[F005]] - Conexión inversa con Fenómeno.
## Isomorfismos
-   [[I###]] - Bellman-Ford ≅ Dijkstra (para grafos con pesos negativos, pero similar lógica de relajación).
-   [[I###]] - Algoritmo A* ≅ Dijkstra (A* es una extensión de Dijkstra que usa heurísticas).

### Instancias en Otros Dominios
-   [[F006]] - Coloreo de grafos (problemas de optimización en grafos).
-   [[F007]] - Satisfacibilidad booleana (SAT) (problemas de búsqueda en espacios discretos).

### Técnicas Aplicables
-   [[T###]] - Colas de Prioridad (estructura de datos esencial para la eficiencia de Dijkstra).
-   [[T###]] - Programación Dinámica (Dijkstra puede verse como una aplicación de programación dinámica).

### Conceptos Fundamentales
-   [[K###]] - Grafos
-   [[K###]] - Algoritmos Greedy
-   [[K###]] - Caminos Más Cortos
-   [[K###]] - Relajación de Aristas

### Prerequisitos
-   [[K###]] - Estructuras de Datos (listas, colas de prioridad)
-   [[K###]] - Teoría de Grafos Básica

## Propiedades Matemáticas

### Complejidad Computacional
-   **Tiempo:** `O(E log V)` o `O(E + V log V)` con una cola de prioridad eficiente (ej. Fibonacci heap). `O(V^2)` con un array simple.
-   **Espacio:** `O(V + E)` para almacenar el grafo y las distancias/predecesores.

### Propiedades Estructurales
-   **Optimalidad:** Dijkstra garantiza encontrar el camino más corto si todos los pesos de las aristas son no negativos.
-   **Subestructura Óptima:** Cualquier subcamino de un camino más corto es también un camino más corto. Esta propiedad es clave para la aplicación de algoritmos greedy y de programación dinámica.

### Teoremas Relevantes
-   **Teorema de Dijkstra:** El algoritmo de Dijkstra, cuando se aplica a un grafo con pesos de arista no negativos, encuentra el camino más corto desde el nodo fuente a todos los demás nodos.

## Visualización

### Tipos de Visualización Aplicables
1.  **Animación del Algoritmo:** Mostrar cómo el algoritmo explora el grafo, actualiza distancias y selecciona el siguiente nodo (resaltando nodos visitados, distancias actuales, camino más corto parcial).
2.  **Representación de Grafo:** Visualizar el grafo con nodos y aristas, y el camino más corto final resaltado.
3.  **Tabla de Distancias/Predecesores:** Mostrar la evolución de las distancias `d[v]` y predecesores `π[v]` en cada paso.

### Componentes Reutilizables
-   Componentes de visualización de grafos (nodos, aristas, pesos).
-   Componentes para resaltar caminos o nodos específicos.
-   Controles de reproducción/pausa para animaciones de algoritmos.

## Recursos

### Literatura Clave
1.  Dijkstra, E. W. (1959). "A note on two problems in connexion with graphs". *Numerische Mathematik*, 1(1), 269-271.
2.  Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.
3.  Skiena, S. S. (2008). *The Algorithm Design Manual* (2nd ed.). Springer.

### Datasets
-   **Mapas de ciudades:** Datos de OpenStreetMap o Google Maps (simplificados).
-   **Redes de transporte:** Datos de vuelos, trenes, etc.
-   **Grafos sintéticos:** Generados aleatoriamente con diferentes densidades y tamaños.

### Implementaciones Existentes
-   **NetworkX (Python):** Librería popular para manipulación y algoritmos de grafos.
-   **Boost Graph Library (C++):** Librería de grafos de alto rendimiento.

### Código en LatticeWeaver
-   **Módulo:** `lattice_weaver/phenomena/dijkstra_shortest_path/`
-   **Tests:** `tests/phenomena/test_dijkstra_shortest_path.py`
-   **Documentación:** `docs/phenomena/dijkstra_shortest_path.md`

## Estado de Implementación

### Fase 1: Investigación
-   [x] Revisión bibliográfica completada
-   [x] Ejemplos concretos identificados
-   [x] Datasets recopilados (referenciados)
-   [ ] Documento de investigación creado (integrado aquí)

### Fase 2: Diseño
-   [x] Mapeo a CSP diseñado
-   [x] Mapeo a otros formalismos (Sistemas Dinámicos)
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

-   **Tiempo de investigación:** 15 horas
-   **Tiempo de diseño:** 8 horas
-   **Tiempo de implementación:** 30 horas
-   **Tiempo de visualización:** 15 horas
-   **Tiempo de documentación:** 8 horas
-   **TOTAL:** 76 horas

## Notas Adicionales

### Ideas para Expansión
-   Implementar variantes como A* o Bellman-Ford para comparar rendimiento y aplicabilidad.
-   Explorar aplicaciones en grafos dinámicos (donde los pesos o la topología cambian).
-   Conexión con problemas de flujo máximo/mínimo.

### Preguntas Abiertas
-   ¿Cómo se adapta Dijkstra a grafos muy grandes o distribuidos?
-   ¿Qué impacto tienen los pesos negativos en la elección del algoritmo?

### Observaciones
-   La simplicidad y elegancia del algoritmo de Dijkstra lo convierten en un excelente punto de partida para enseñar algoritmos de grafos y optimización.

---

**Última actualización:** 2025-10-13
**Responsable:** Agente Autónomo de Análisis
- [[C001]]
- [[C003]]
- [[F006]]
- [[F007]]
- [[T002]]
- [[K003]]
- [[K006]]
- [[K008]]
