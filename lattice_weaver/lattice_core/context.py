"""
FormalContext: Contexto formal para Formal Concept Analysis (FCA).

Este módulo implementa el concepto de contexto formal, que es la estructura
básica de FCA, y los operadores de Galois asociados.

Autor: Manus AI
Fecha: 11 de Octubre de 2025
"""

from typing import Set, FrozenSet, Tuple, List


class FormalContext:
    """
    Contexto formal para FCA.
    
    Un contexto formal es una tripleta K = (G, M, I) donde:
    - G: Conjunto de objetos
    - M: Conjunto de atributos
    - I ⊆ G × M: Relación de incidencia
    
    Attributes:
        objects: Conjunto de objetos
        attributes: Conjunto de atributos
        incidence: Relación de incidencia (pares objeto-atributo)
    """
    
    def __init__(self):
        """Inicializa un contexto formal vacío."""
        self.objects: Set = set()
        self.attributes: Set = set()
        self.incidence: Set[Tuple] = set()
    
    def add_object(self, obj):
        """
        Añade un objeto al contexto.
        
        Args:
            obj: Objeto a añadir
        """
        self.objects.add(obj)
    
    def add_attribute(self, attr):
        """
        Añade un atributo al contexto.
        
        Args:
            attr: Atributo a añadir
        """
        self.attributes.add(attr)
    
    def add_incidence(self, obj, attr):
        """
        Añade una relación de incidencia.
        
        Args:
            obj: Objeto
            attr: Atributo
        """
        if obj in self.objects and attr in self.attributes:
            self.incidence.add((obj, attr))
    
    def prime_objects(self, objects: Set) -> Set:
        """
        Calcula A' (operador de Galois sobre objetos).
        
        Retorna el conjunto de atributos comunes a todos los objetos en A.
        
        Args:
            objects: Conjunto de objetos
            
        Returns:
            Conjunto de atributos comunes
        """
        if not objects:
            return self.attributes.copy()
        
        common_attrs = None
        
        for obj in objects:
            obj_attrs = {attr for (o, attr) in self.incidence if o == obj}
            
            if common_attrs is None:
                common_attrs = obj_attrs
            else:
                common_attrs &= obj_attrs
        
        return common_attrs if common_attrs is not None else set()
    
    def prime_attributes(self, attributes: Set) -> Set:
        """
        Calcula B' (operador de Galois sobre atributos).
        
        Retorna el conjunto de objetos que tienen todos los atributos en B.
        
        Args:
            attributes: Conjunto de atributos
            
        Returns:
            Conjunto de objetos con todos los atributos
        """
        if not attributes:
            return self.objects.copy()
        
        objects_with_all = self.objects.copy()
        
        for attr in attributes:
            objects_with_attr = {obj for (obj, a) in self.incidence if a == attr}
            objects_with_all &= objects_with_attr
        
        return objects_with_all
    
    def get_object_attributes(self, obj) -> Set:
        """
        Obtiene todos los atributos de un objeto.
        
        Args:
            obj: Objeto
            
        Returns:
            Conjunto de atributos del objeto
        """
        return {attr for (o, attr) in self.incidence if o == obj}
    
    def get_attribute_objects(self, attr) -> Set:
        """
        Obtiene todos los objetos que tienen un atributo.
        
        Args:
            attr: Atributo
            
        Returns:
            Conjunto de objetos con el atributo
        """
        return {obj for (obj, a) in self.incidence if a == attr}
    
    def to_concepts_format(self) -> str:
        """
        Convierte el contexto al formato de la librería `concepts`.
        
        Returns:
            String en formato de concepts
        """
        # Ordenar objetos y atributos
        objects_list = sorted(self.objects, key=str)
        attributes_list = sorted(self.attributes, key=str)
        
        # Construir encabezado
        header = "           |" + "|".join(str(a) for a in attributes_list) + "|\n"
        
        # Construir filas
        rows = []
        for obj in objects_list:
            row = f"    {obj:<7}|"
            for attr in attributes_list:
                if (obj, attr) in self.incidence:
                    row += "X|"
                else:
                    row += " |"
            rows.append(row)
        
        return header + "\n".join(rows)
    
    def get_statistics(self) -> dict:
        """
        Obtiene estadísticas del contexto.
        
        Returns:
            Diccionario con estadísticas
        """
        density = len(self.incidence) / (len(self.objects) * len(self.attributes)) if self.objects and self.attributes else 0
        
        return {
            'num_objects': len(self.objects),
            'num_attributes': len(self.attributes),
            'num_incidences': len(self.incidence),
            'density': round(density, 4)
        }
    
    @classmethod
    def from_arc_engine(cls, arc_engine) -> 'FormalContext':
        """
        Crea un contexto formal desde un ArcEngine.
        
        Los objetos son las variables y los atributos son los valores posibles.
        
        Args:
            arc_engine: Instancia de ArcEngine
            
        Returns:
            FormalContext construido
        """
        context = cls()
        
        # Añadir objetos (variables)
        for var_name in arc_engine.variables.keys():
            context.add_object(var_name)
        
        # Añadir atributos (valores únicos)
        all_values = set()
        for domain in arc_engine.variables.values():
            all_values.update(domain.get_values())
        
        for value in all_values:
            context.add_attribute(value)
        
        # Añadir incidencias (variable puede tener valor)
        for var_name, domain in arc_engine.variables.items():
            for value in domain.get_values():
                context.add_incidence(var_name, value)
        
        return context
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FormalContext':
        """
        Crea un contexto desde un diccionario.
        
        Args:
            data: Diccionario con objetos, atributos e incidencias
            
        Returns:
            FormalContext construido
        """
        context = cls()
        
        context.objects = set(data.get('objects', []))
        context.attributes = set(data.get('attributes', []))
        context.incidence = {tuple(inc) for inc in data.get('incidence', [])}
        
        return context
    
    def to_dict(self) -> dict:
        """
        Serializa el contexto a un diccionario.
        
        Returns:
            Diccionario con el contexto
        """
        return {
            'objects': list(self.objects),
            'attributes': list(self.attributes),
            'incidence': [list(inc) for inc in self.incidence]
        }

