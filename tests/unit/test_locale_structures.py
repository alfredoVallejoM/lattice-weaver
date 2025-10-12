"""
Tests Unitarios para Estructuras de Locales y Frames

Tests exhaustivos para:
- PartialOrder
- CompleteLattice
- Frame
- Locale

Autor: LatticeWeaver Team (Track B)
Fecha: 12 de Octubre de 2025
"""

import pytest
from lattice_weaver.topology_new.locale import (
    PartialOrder,
    CompleteLattice,
    Frame,
    Locale,
    LatticeBuilder,
    FrameBuilder,
    LocaleBuilder,
    FrozenDict
)


# ============================================================================
# Tests de PartialOrder
# ============================================================================

class TestPartialOrder:
    """Tests para PartialOrder."""
    
    def test_construction_simple(self):
        """Test construcción de poset simple."""
        elements = frozenset({1, 2, 3})
        leq = frozenset({
            (1, 1), (2, 2), (3, 3),  # Reflexividad
            (1, 2), (2, 3), (1, 3)   # Orden
        })
        
        poset = PartialOrder(elements, leq)
        
        assert len(poset) == 3
        assert 1 in poset
        assert 4 not in poset
    
    def test_reflexivity_verification(self):
        """Test verificación de reflexividad."""
        elements = frozenset({1, 2})
        leq = frozenset({(1, 1)})  # Falta (2, 2)
        
        with pytest.raises(ValueError, match="Reflexividad violada"):
            PartialOrder(elements, leq)
    
    def test_antisymmetry_verification(self):
        """Test verificación de antisimetría."""
        elements = frozenset({1, 2})
        leq = frozenset({
            (1, 1), (2, 2),
            (1, 2), (2, 1)  # Violación: 1 ≤ 2 y 2 ≤ 1 pero 1 ≠ 2
        })
        
        with pytest.raises(ValueError, match="Antisimetría violada"):
            PartialOrder(elements, leq)
    
    def test_transitivity_verification(self):
        """Test verificación de transitividad."""
        elements = frozenset({1, 2, 3})
        leq = frozenset({
            (1, 1), (2, 2), (3, 3),
            (1, 2), (2, 3)  # Falta (1, 3) para transitividad
        })
        
        with pytest.raises(ValueError, match="Transitividad violada"):
            PartialOrder(elements, leq)
    
    def test_is_leq(self):
        """Test verificación de orden."""
        poset = self._create_divisors_poset(12)
        
        assert poset.is_leq(1, 12)
        assert poset.is_leq(2, 6)
        assert poset.is_leq(3, 3)
        assert not poset.is_leq(3, 4)
    
    def test_is_less(self):
        """Test verificación de orden estricto."""
        poset = self._create_divisors_poset(12)
        
        assert poset.is_less(1, 12)
        assert poset.is_less(2, 6)
        assert not poset.is_less(3, 3)  # No estricto
        assert not poset.is_less(3, 4)
    
    def test_upper_bounds(self):
        """Test cálculo de cotas superiores."""
        poset = self._create_divisors_poset(12)
        
        # Cotas superiores de {2, 3}
        upper = poset.upper_bounds({2, 3})
        assert upper == {6, 12}  # mcm(2,3) = 6, y 12
    
    def test_lower_bounds(self):
        """Test cálculo de cotas inferiores."""
        poset = self._create_divisors_poset(12)
        
        # Cotas inferiores de {6, 12}
        lower = poset.lower_bounds({6, 12})
        assert lower == {1, 2, 3, 6}  # Divisores comunes
    
    def test_minimal_elements(self):
        """Test cálculo de elementos minimales."""
        poset = self._create_divisors_poset(12)
        
        minimal = poset.minimal_elements()
        assert minimal == {1}
    
    def test_maximal_elements(self):
        """Test cálculo de elementos maximales."""
        poset = self._create_divisors_poset(12)
        
        maximal = poset.maximal_elements()
        assert maximal == {12}
    
    def test_comparable(self):
        """Test verificación de comparabilidad."""
        poset = self._create_divisors_poset(12)
        
        assert poset.comparable(2, 6)  # 2 | 6
        assert poset.comparable(6, 2)  # Mismo par
        assert not poset.comparable(2, 3)  # Incomparables
    
    def test_is_chain(self):
        """Test verificación de cadena."""
        poset = self._create_divisors_poset(12)
        
        assert poset.is_chain({1, 2, 4, 12})  # Cadena
        assert not poset.is_chain({2, 3})  # No cadena
    
    def test_is_antichain(self):
        """Test verificación de anticadena."""
        poset = self._create_divisors_poset(12)
        
        assert poset.is_antichain({2, 3})  # Anticadena
        assert not poset.is_antichain({2, 4})  # No anticadena
    
    def test_hasse_diagram(self):
        """Test cálculo de diagrama de Hasse."""
        poset = self._create_divisors_poset(12)
        
        hasse = poset.hasse_diagram_edges()
        
        # Verificar que contiene relaciones de cobertura
        assert (1, 2) in hasse
        assert (1, 3) in hasse
        assert (2, 4) in hasse
        assert (2, 6) in hasse
        
        # Verificar que NO contiene relaciones transitivas
        assert (1, 12) not in hasse  # Transitiva
    
    # Helpers
    
    def _create_divisors_poset(self, n: int) -> PartialOrder:
        """Crea poset de divisores de n."""
        divisors = set()
        for i in range(1, n + 1):
            if n % i == 0:
                divisors.add(i)
        
        divisors = frozenset(divisors)
        
        leq = frozenset(
            (a, b)
            for a in divisors
            for b in divisors
            if b % a == 0
        )
        
        return PartialOrder(divisors, leq)


# ============================================================================
# Tests de CompleteLattice
# ============================================================================

class TestCompleteLattice:
    """Tests para CompleteLattice."""
    
    def test_construction_from_divisors(self):
        """Test construcción desde divisores."""
        lattice = LatticeBuilder.from_divisors(12)
        
        assert lattice.top == 12
        assert lattice.bottom == 1
        assert len(lattice.poset.elements) == 6  # {1, 2, 3, 4, 6, 12}
    
    def test_join_binary(self):
        """Test supremo binario."""
        lattice = LatticeBuilder.from_divisors(12)
        
        # mcm(2, 3) = 6
        assert lattice.join({2, 3}) == 6
        
        # mcm(4, 6) = 12
        assert lattice.join({4, 6}) == 12
    
    def test_meet_binary(self):
        """Test ínfimo binario."""
        lattice = LatticeBuilder.from_divisors(12)
        
        # mcd(6, 12) = 6
        assert lattice.meet({6, 12}) == 6
        
        # mcd(4, 6) = 2
        assert lattice.meet({4, 6}) == 2
    
    def test_join_empty_set(self):
        """Test supremo de conjunto vacío."""
        lattice = LatticeBuilder.from_divisors(12)
        
        # ⋁ ∅ = ⊥
        assert lattice.join(set()) == lattice.bottom
    
    def test_meet_empty_set(self):
        """Test ínfimo de conjunto vacío."""
        lattice = LatticeBuilder.from_divisors(12)
        
        # ⋀ ∅ = ⊤
        assert lattice.meet(set()) == lattice.top
    
    def test_join_singleton(self):
        """Test supremo de singleton."""
        lattice = LatticeBuilder.from_divisors(12)
        
        # ⋁ {a} = a
        assert lattice.join({6}) == 6
    
    def test_meet_singleton(self):
        """Test ínfimo de singleton."""
        lattice = LatticeBuilder.from_divisors(12)
        
        # ⋀ {a} = a
        assert lattice.meet({6}) == 6
    
    def test_join_idempotent(self):
        """Test idempotencia del supremo."""
        lattice = LatticeBuilder.from_divisors(12)
        
        # a ∨ a = a
        assert lattice.join_binary(6, 6) == 6
    
    def test_meet_idempotent(self):
        """Test idempotencia del ínfimo."""
        lattice = LatticeBuilder.from_divisors(12)
        
        # a ∧ a = a
        assert lattice.meet_binary(6, 6) == 6
    
    def test_join_commutative(self):
        """Test conmutatividad del supremo."""
        lattice = LatticeBuilder.from_divisors(12)
        
        # a ∨ b = b ∨ a
        assert lattice.join_binary(2, 3) == lattice.join_binary(3, 2)
    
    def test_meet_commutative(self):
        """Test conmutatividad del ínfimo."""
        lattice = LatticeBuilder.from_divisors(12)
        
        # a ∧ b = b ∧ a
        assert lattice.meet_binary(6, 12) == lattice.meet_binary(12, 6)
    
    def test_join_associative(self):
        """Test asociatividad del supremo."""
        lattice = LatticeBuilder.from_divisors(12)
        
        # (a ∨ b) ∨ c = a ∨ (b ∨ c)
        lhs = lattice.join_binary(lattice.join_binary(2, 3), 4)
        rhs = lattice.join_binary(2, lattice.join_binary(3, 4))
        assert lhs == rhs
    
    def test_meet_associative(self):
        """Test asociatividad del ínfimo."""
        lattice = LatticeBuilder.from_divisors(12)
        
        # (a ∧ b) ∧ c = a ∧ (b ∧ c)
        lhs = lattice.meet_binary(lattice.meet_binary(12, 6), 4)
        rhs = lattice.meet_binary(12, lattice.meet_binary(6, 4))
        assert lhs == rhs
    
    def test_absorption_laws(self):
        """Test leyes de absorción."""
        lattice = LatticeBuilder.from_divisors(12)
        
        # a ∨ (a ∧ b) = a
        a, b = 6, 12
        assert lattice.join_binary(a, lattice.meet_binary(a, b)) == a
        
        # a ∧ (a ∨ b) = a
        assert lattice.meet_binary(a, lattice.join_binary(a, b)) == a
    
    def test_powerset_lattice(self):
        """Test retículo de powerset."""
        lattice = LatticeBuilder.from_powerset({1, 2, 3})
        
        # Top = {1, 2, 3}
        assert lattice.top == frozenset({1, 2, 3})
        
        # Bottom = ∅
        assert lattice.bottom == frozenset()
        
        # Unión
        a = frozenset({1})
        b = frozenset({2})
        assert lattice.join({a, b}) == frozenset({1, 2})
        
        # Intersección
        c = frozenset({1, 2})
        d = frozenset({2, 3})
        assert lattice.meet({c, d}) == frozenset({2})


# ============================================================================
# Tests de Frame
# ============================================================================

class TestFrame:
    """Tests para Frame."""
    
    def test_construction_from_powerset(self):
        """Test construcción de Frame desde powerset."""
        frame = FrameBuilder.from_powerset({1, 2, 3})
        
        assert frame.top == frozenset({1, 2, 3})
        assert frame.bottom == frozenset()
        assert len(frame.poset.elements) == 8  # 2^3
    
    def test_infinite_distributivity_powerset(self):
        """Test ley distributiva infinita en powerset."""
        frame = FrameBuilder.from_powerset({1, 2, 3})
        
        # a ∧ (⋁ S) = ⋁ {a ∧ s | s ∈ S}
        a = frozenset({1, 2})
        s1 = frozenset({1})
        s2 = frozenset({2, 3})
        S = {s1, s2}
        
        # LHS: a ∧ (⋁ S)
        join_s = frame.join(S)
        lhs = frame.meet_binary(a, join_s)
        
        # RHS: ⋁ {a ∧ s | s ∈ S}
        meets = {frame.meet_binary(a, s) for s in S}
        rhs = frame.join(meets)
        
        assert lhs == rhs
    
    def test_heyting_implication_powerset(self):
        """Test implicación de Heyting en powerset."""
        frame = FrameBuilder.from_powerset({1, 2, 3})
        
        # En powerset: A → B = ¬A ∪ B = (X - A) ∪ B
        a = frozenset({1})
        b = frozenset({1, 2})
        
        impl = frame.heyting_implication(a, b)
        
        # ¬{1} ∪ {1,2} = {2,3} ∪ {1,2} = {1,2,3}
        assert impl == frozenset({1, 2, 3})
    
    def test_heyting_negation_powerset(self):
        """Test negación de Heyting en powerset."""
        frame = FrameBuilder.from_powerset({1, 2, 3})
        
        # En powerset: ¬A = X - A (complemento)
        a = frozenset({1})
        neg_a = frame.heyting_negation(a)
        
        assert neg_a == frozenset({2, 3})
    
    def test_implication_when_leq(self):
        """Test a → b = ⊤ cuando a ≤ b."""
        frame = FrameBuilder.from_powerset({1, 2, 3})
        
        a = frozenset({1})
        b = frozenset({1, 2})
        
        # a ⊆ b → a → b = ⊤
        impl = frame.heyting_implication(a, b)
        assert impl == frame.top
    
    def test_double_negation(self):
        """Test doble negación."""
        frame = FrameBuilder.from_powerset({1, 2, 3})
        
        a = frozenset({1})
        neg_neg_a = frame.heyting_negation(frame.heyting_negation(a))
        
        # En álgebra de Boole (powerset), ¬¬a = a
        assert neg_neg_a == a
    
    def test_regular_elements_powerset(self):
        """Test elementos regulares en powerset."""
        frame = FrameBuilder.from_powerset({1, 2})
        
        # En powerset, todos los elementos son regulares
        regular = frame.regular_elements()
        assert len(regular) == len(frame.poset.elements)
    
    def test_negation_of_top_and_bottom(self):
        """Test negación de top y bottom."""
        frame = FrameBuilder.from_powerset({1, 2})
        
        # ¬⊤ = ⊥
        assert frame.heyting_negation(frame.top) == frame.bottom
        
        # ¬⊥ = ⊤
        assert frame.heyting_negation(frame.bottom) == frame.top


# ============================================================================
# Tests de Locale
# ============================================================================

class TestLocale:
    """Tests para Locale."""
    
    def test_construction_discrete(self):
        """Test construcción de Locale discreto."""
        locale = LocaleBuilder.discrete_locale({1, 2, 3})
        
        assert len(locale.opens()) == 8  # 2^3 abiertos
        assert locale.is_open(frozenset({1}))
        assert locale.is_open(frozenset())
    
    def test_construction_trivial(self):
        """Test construcción de Locale trivial."""
        locale = LocaleBuilder.trivial_locale({1, 2, 3})
        
        assert len(locale.opens()) == 2  # Solo ∅ y X
        assert locale.is_open(frozenset())
        assert locale.is_open(frozenset({1, 2, 3}))
        assert not locale.is_open(frozenset({1}))
    
    def test_union_of_opens(self):
        """Test unión de abiertos."""
        locale = LocaleBuilder.discrete_locale({1, 2, 3})
        
        u1 = frozenset({1})
        u2 = frozenset({2})
        
        union = locale.union({u1, u2})
        assert union == frozenset({1, 2})
    
    def test_intersection_of_opens(self):
        """Test intersección de abiertos."""
        locale = LocaleBuilder.discrete_locale({1, 2, 3})
        
        u1 = frozenset({1, 2})
        u2 = frozenset({2, 3})
        
        intersection = locale.intersection({u1, u2})
        assert intersection == frozenset({2})
    
    def test_interior_operator(self):
        """Test operador interior."""
        locale = LocaleBuilder.discrete_locale({1, 2, 3})
        
        # En espacio discreto, todo es abierto → int(a) = a
        a = frozenset({1, 2})
        assert locale.interior(a) == a
    
    def test_closure_operator(self):
        """Test operador clausura."""
        locale = LocaleBuilder.discrete_locale({1, 2, 3})
        
        # En espacio discreto, todo es cerrado → cl(a) = a
        a = frozenset({1, 2})
        assert locale.closure(a) == a
    
    def test_boundary_operator(self):
        """Test operador frontera."""
        locale = LocaleBuilder.discrete_locale({1, 2, 3})
        
        # En espacio discreto, ∂a = ∅ (todo es abierto y cerrado)
        a = frozenset({1, 2})
        boundary = locale.boundary(a)
        assert boundary == frozenset()
    
    def test_interior_idempotent(self):
        """Test idempotencia del interior."""
        locale = LocaleBuilder.discrete_locale({1, 2, 3})
        
        a = frozenset({1, 2})
        int_a = locale.interior(a)
        int_int_a = locale.interior(int_a)
        
        # int(int(a)) = int(a)
        assert int_int_a == int_a
    
    def test_closure_idempotent(self):
        """Test idempotencia de la clausura."""
        locale = LocaleBuilder.discrete_locale({1, 2, 3})
        
        a = frozenset({1, 2})
        cl_a = locale.closure(a)
        cl_cl_a = locale.closure(cl_a)
        
        # cl(cl(a)) = cl(a)
        assert cl_cl_a == cl_a
    
    def test_interior_preserves_meets(self):
        """Test interior preserva ínfimos."""
        locale = LocaleBuilder.discrete_locale({1, 2, 3})
        
        a = frozenset({1, 2})
        b = frozenset({2, 3})
        
        # int(a ∧ b) = int(a) ∧ int(b)
        meet_ab = locale.frame.meet_binary(a, b)
        lhs = locale.interior(meet_ab)
        
        int_a = locale.interior(a)
        int_b = locale.interior(b)
        rhs = locale.frame.meet_binary(int_a, int_b)
        
        assert lhs == rhs
    
    def test_is_dense(self):
        """Test verificación de densidad."""
        locale = LocaleBuilder.discrete_locale({1, 2, 3})
        
        # Solo ⊤ es denso en espacio discreto
        assert locale.is_dense(locale.frame.top)
        assert not locale.is_dense(frozenset({1}))
    
    def test_is_nowhere_dense(self):
        """Test verificación de nowhere dense."""
        locale = LocaleBuilder.discrete_locale({1, 2, 3})
        
        # En espacio discreto, solo ∅ es nowhere dense
        assert locale.is_nowhere_dense(frozenset())
        assert not locale.is_nowhere_dense(frozenset({1}))


# ============================================================================
# Tests de Integración
# ============================================================================

class TestIntegration:
    """Tests de integración entre componentes."""
    
    def test_divisors_lattice_to_frame(self):
        """Test conversión de retículo de divisores a Frame."""
        lattice = LatticeBuilder.from_divisors(12)
        
        # Los divisores NO forman un Frame en general
        # (no satisfacen distributividad infinita)
        # Pero para números pequeños puede funcionar
        
        try:
            frame = FrameBuilder.from_complete_lattice(lattice)
            # Si pasa, verificar que es válido
            assert frame.top == 12
            assert frame.bottom == 1
        except ValueError:
            # Esperado: divisores no forman Frame
            pass
    
    def test_powerset_chain(self):
        """Test cadena completa: powerset → lattice → frame → locale."""
        # Powerset
        base = {1, 2}
        
        # Lattice
        lattice = LatticeBuilder.from_powerset(base)
        assert len(lattice.poset.elements) == 4  # 2^2
        
        # Frame
        frame = FrameBuilder.from_complete_lattice(lattice)
        assert frame.top == frozenset({1, 2})
        
        # Locale
        locale = LocaleBuilder.from_frame(frame, name="Test")
        assert locale.name == "Test"
        assert len(locale.opens()) == 4
    
    def test_operations_consistency(self):
        """Test consistencia entre operaciones de Frame y Locale."""
        frame = FrameBuilder.from_powerset({1, 2})
        locale = LocaleBuilder.from_frame(frame)
        
        a = frozenset({1})
        b = frozenset({2})
        
        # Unión en Locale = join en Frame
        union_locale = locale.union({a, b})
        join_frame = frame.join({a, b})
        assert union_locale == join_frame
        
        # Intersección en Locale = meet en Frame
        intersection_locale = locale.intersection({a, b})
        meet_frame = frame.meet({a, b})
        assert intersection_locale == meet_frame


# ============================================================================
# Tests de FrozenDict
# ============================================================================

class TestFrozenDict:
    """Tests para FrozenDict."""
    
    def test_construction(self):
        """Test construcción de FrozenDict."""
        fd = FrozenDict({'a': 1, 'b': 2})
        
        assert fd['a'] == 1
        assert fd['b'] == 2
    
    def test_immutability(self):
        """Test inmutabilidad de FrozenDict."""
        fd = FrozenDict({'a': 1})
        
        with pytest.raises(TypeError):
            fd['b'] = 2
        
        with pytest.raises(TypeError):
            del fd['a']
        
        with pytest.raises(TypeError):
            fd.clear()
    
    def test_hashable(self):
        """Test que FrozenDict es hashable."""
        fd1 = FrozenDict({'a': 1, 'b': 2})
        fd2 = FrozenDict({'a': 1, 'b': 2})
        
        # Puede usarse en sets
        s = {fd1, fd2}
        assert len(s) == 1  # Son iguales
    
    def test_equality(self):
        """Test igualdad de FrozenDicts."""
        fd1 = FrozenDict({'a': 1, 'b': 2})
        fd2 = FrozenDict({'a': 1, 'b': 2})
        fd3 = FrozenDict({'a': 1, 'c': 3})
        
        assert fd1 == fd2
        assert fd1 != fd3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

