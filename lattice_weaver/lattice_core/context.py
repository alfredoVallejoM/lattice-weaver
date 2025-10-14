from typing import Set, FrozenSet, Dict, Any, List


class FormalContext:
    """
    Representa un contexto formal en Formal Concept Analysis (FCA).

    Un contexto formal se define como una terna (G, M, I), donde G es un conjunto
    de objetos, M es un conjunto de atributos, e I es una relación binaria
    entre G y M (I ⊆ G × M).

    Attributes:
        objects: Conjunto de objetos.
        attributes: Conjunto de atributos.
        incidences: Conjunto de pares (objeto, atributo) que representan la relación I.
        _obj_to_attrs: Diccionario que mapea cada objeto a un conjunto de sus atributos.
        _attr_to_objs: Diccionario que mapea cada atributo a un conjunto de los objetos que lo poseen.
        _objects_list: Lista ordenada de objetos para indexación.
        _object_to_index: Mapeo de objeto a su índice en _objects_list.
    """

    def __init__(self, objects: Set[Any] = None, attributes: Set[Any] = None, incidences: Set[tuple] = None):
        self.objects = objects if objects is not None else set()
        self.attributes = attributes if attributes is not None else set()
        self.incidences = incidences if incidences is not None else set()

        self._obj_to_attrs: Dict[Any, Set[Any]] = {obj: set() for obj in self.objects}
        self._attr_to_objs: Dict[Any, Set[Any]] = {attr: set() for attr in self.attributes}
        self._objects_list: List[Any] = sorted(list(self.objects))
        self._object_to_index: Dict[Any, int] = {obj: i for i, obj in enumerate(self._objects_list)}

        for obj, attr in self.incidences:
            self._obj_to_attrs.setdefault(obj, set()).add(attr)
            self._attr_to_objs.setdefault(attr, set()).add(obj)

    def add_object(self, obj: Any):
        if obj not in self.objects:
            self.objects.add(obj)
            self._obj_to_attrs[obj] = set()
            # Mantener _objects_list y _object_to_index actualizados y ordenados
            self._objects_list.append(obj)
            self._objects_list.sort()
            self._object_to_index = {o: i for i, o in enumerate(self._objects_list)}

    def add_attribute(self, attr: Any):
        if attr not in self.attributes:
            self.attributes.add(attr)
            self._attr_to_objs[attr] = set()

    def add_incidence(self, obj: Any, attr: Any):
        self.add_object(obj)
        self.add_attribute(attr)
        if (obj, attr) not in self.incidences:
            self.incidences.add((obj, attr))
            self._obj_to_attrs[obj].add(attr)
            self._attr_to_objs[attr].add(obj)

    def prime_objects(self, objects_subset: Set[Any]) -> FrozenSet[Any]:
        """
        Calcula la intensión de un subconjunto de objetos (A’).
        
        Args:
            objects_subset: Subconjunto de objetos.
            
        Returns:
            Conjunto de atributos que todos los objetos en objects_subset poseen.
        """
        if not objects_subset:
            return frozenset(self.attributes)

        common_attributes = None
        for obj in objects_subset:
            if obj in self._obj_to_attrs:
                if common_attributes is None:
                    common_attributes = self._obj_to_attrs[obj].copy()
                else:
                    common_attributes.intersection_update(self._obj_to_attrs[obj])
            else:
                return frozenset() # Objeto no existe en el contexto
        return frozenset(common_attributes) if common_attributes is not None else frozenset()

    def prime_attributes(self, attributes_subset: Set[Any]) -> FrozenSet[Any]:
        """
        Calcula la extensión de un subconjunto de atributos (B’).
        
        Args:
            attributes_subset: Subconjunto de atributos.
            
        Returns:
            Conjunto de objetos que poseen todos los atributos en attributes_subset.
        """
        if not attributes_subset:
            return frozenset(self.objects)

        common_objects = None
        for attr in attributes_subset:
            if attr in self._attr_to_objs:
                if common_objects is None:
                    common_objects = self._attr_to_objs[attr].copy()
                else:
                    common_objects.intersection_update(self._attr_to_objs[attr])
            else:
                return frozenset() # Atributo no existe en el contexto
        return frozenset(common_objects) if common_objects is not None else frozenset()

    def get_object_index(self, obj: Any) -> int:
        """
        Devuelve el índice de un objeto en la lista ordenada de objetos.
        
        Args:
            obj: Objeto a buscar.
            
        Returns:
            Índice del objeto.
            
        Raises:
            ValueError: Si el objeto no se encuentra en el contexto.
        """
        if obj not in self._object_to_index:
            raise ValueError(f"Object {obj} not found in context.")
        return self._object_to_index[obj]

    def to_concepts_format(self) -> str:
        """
        Convierte el contexto formal a un formato de cadena compatible con la librería `concepts`.
        
        Returns:
            Cadena representando el contexto formal.
        """
        header = ' '.join(sorted(list(self.attributes)))
        rows = []
        for obj in sorted(list(self.objects)):
            row = [obj] + ['X' if (obj, attr) in self.incidences else '' for attr in sorted(list(self.attributes))]
            rows.append(' '.join(map(str, row)))
        return f" {header}\n" + '\n'.join(rows)

    @classmethod
    def from_arc_engine(cls, arc_engine) -> "FormalContext":
        """
        Crea un LatticeBuilder desde un ArcEngine.
        
        Args:
            arc_engine: Instancia de ArcEngine
            
        Returns:
            LatticeBuilder con el contexto construido
        """
        objects = set(arc_engine.variables.keys())
        attributes = set()
        incidences = set()

        for var_name, var_obj in arc_engine.variables.items():
            for domain_val in var_obj.domain:
                attr_name = f"{var_name}={domain_val}"
                attributes.add(attr_name)
                incidences.add((var_name, attr_name))

        return cls(objects, attributes, incidences)

