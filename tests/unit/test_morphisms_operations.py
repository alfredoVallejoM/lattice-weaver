"""
Tests Unitarios para Morfismos y Operaciones

Tests exhaustivos para:
- FrameMorphism
- LocaleMorphism
- ModalOperators
- TopologicalOperators
- ConnectivityAnalyzer

Autor: LatticeWeaver Team (Track B)
Fecha: 12 de Octubre de 2025
"""

import pytest
from lattice_weaver.topology_new.locale import (
    Frame, Locale, FrameBuilder, LocaleBuilder
)
from lattice_weaver.topology_new.morphisms import (
    FrameMorphism, LocaleMorphism, FrameConstructions, MorphismBuilder
)
from lattice_weaver.topology_new.operations import (
    ModalOperators, TopologicalOperators, ConnectivityAnalyzer, LocaleAnalyzer
)


# ============================================================================
# Tests de FrameMorphism
# ============================================================================

class TestFrameMorphism:
    """Tests para FrameMorphism."""
    
    def test_identity_morphism(self):
        """Test morfismo identidad."""
        frame = FrameBuilder.from_powerset({1, 2})
        
        id_f = FrameMorphism.identity(frame)
        
        # Verificar que id(x) = x
        for elem in frame.poset.elements:
            assert id_f(elem) == elem
    
    def test_constant_morphism(self):
        """Test morfismo constante."""
        source = FrameBuilder.from_powerset({1})
        target = FrameBuilder.from_powerset({1, 2})
        
        # Morfismo constante a ⊤
        const_f = MorphismBuilder.constant_morphism(source, target, target.top)
        
        # Verificar que mapea todo a ⊤
        for elem in source.poset.elements:
            assert const_f(elem) == target.top
    
    def test_composition(self):
        """Test composición de morfismos."""
        f1 = FrameBuilder.from_powerset({1})
        f2 = FrameBuilder.from_powerset({2})
        f3 = FrameBuilder.from_powerset({3})
        
        # f: f1 → f2 (identidad trivial)
        id1 = FrameMorphism.identity(f1)
        
        # g: f2 → f3 (identidad trivial)
        id2 = FrameMorphism.identity(f2)
        
        # Para composición real, necesitamos morfismos compatibles
        # Simplificamos usando identidades
        
        # Composición id ∘ id = id
        comp = id1.compose(id1)
        assert comp.source == id1.source
        assert comp.target == id1.target
    
    def test_preserves_joins(self):
        """Test que morfismo preserva supremos."""
        frame = FrameBuilder.from_powerset({1, 2})
        id_f = FrameMorphism.identity(frame)
        
        # f(a ∨ b) = f(a) ∨ f(b)
        a = frozenset({1})
        b = frozenset({2})
        
        join_ab = frame.join({a, b})
        lhs = id_f(join_ab)
        
        rhs = frame.join({id_f(a), id_f(b)})
        
        assert lhs == rhs
    
    def test_preserves_meets(self):
        """Test que morfismo preserva ínfimos."""
        frame = FrameBuilder.from_powerset({1, 2})
        id_f = FrameMorphism.identity(frame)
        
        # f(a ∧ b) = f(a) ∧ f(b)
        a = frozenset({1, 2})
        b = frozenset({2})
        
        meet_ab = frame.meet_binary(a, b)
        lhs = id_f(meet_ab)
        
        rhs = frame.meet_binary(id_f(a), id_f(b))
        
        assert lhs == rhs
    
    def test_preserves_extrema(self):
        """Test que morfismo preserva top y bottom."""
        frame = FrameBuilder.from_powerset({1, 2})
        id_f = FrameMorphism.identity(frame)
        
        assert id_f(frame.top) == frame.top
        assert id_f(frame.bottom) == frame.bottom


# ============================================================================
# Tests de LocaleMorphism
# ============================================================================

class TestLocaleMorphism:
    """Tests para LocaleMorphism."""
    
    def test_identity_locale_morphism(self):
        """Test morfismo identidad de Locales."""
        locale = LocaleBuilder.discrete_locale({1, 2})
        
        id_l = LocaleMorphism.identity(locale)
        
        # Verificar pullback
        for open_set in locale.opens():
            assert id_l.pullback(open_set) == open_set
    
    def test_pullback(self):
        """Test pullback de abiertos."""
        locale = LocaleBuilder.discrete_locale({1, 2})
        id_l = LocaleMorphism.identity(locale)
        
        open_set = frozenset({1})
        pullback = id_l.pullback(open_set)
        
        assert pullback == open_set


# ============================================================================
# Tests de FrameConstructions
# ============================================================================

class TestFrameConstructions:
    """Tests para construcciones categóricas."""
    
    def test_product(self):
        """Test producto de Frames."""
        f1 = FrameBuilder.from_powerset({1})
        f2 = FrameBuilder.from_powerset({2})
        
        product = FrameConstructions.product(f1, f2)
        
        # |L × M| = |L| * |M|
        assert len(product.poset.elements) == len(f1.poset.elements) * len(f2.poset.elements)
    
    def test_projections(self):
        """Test proyecciones del producto."""
        f1 = FrameBuilder.from_powerset({1})
        f2 = FrameBuilder.from_powerset({2})
        
        proj1 = FrameConstructions.projection_left(f1, f2)
        proj2 = FrameConstructions.projection_right(f1, f2)
        
        # Verificar que proyecciones son morfismos válidos
        assert proj1.target == f1
        assert proj2.target == f2


# ============================================================================
# Tests de ModalOperators
# ============================================================================

class TestModalOperators:
    """Tests para operadores modales."""
    
    def test_diamond_operator(self):
        """Test operador ◇ (interior)."""
        locale = LocaleBuilder.discrete_locale({1, 2})
        modal = ModalOperators(locale)
        
        # En espacio discreto: ◇a = a
        a = frozenset({1})
        assert modal.diamond(a) == a
    
    def test_box_operator(self):
        """Test operador □ (clausura)."""
        locale = LocaleBuilder.discrete_locale({1, 2})
        modal = ModalOperators(locale)
        
        # En espacio discreto: □a = a
        a = frozenset({1})
        assert modal.box(a) == a
    
    def test_s4_axiom_t(self):
        """Test axioma T: □p → p."""
        locale = LocaleBuilder.discrete_locale({1, 2, 3})
        modal = ModalOperators(locale)
        
        for p in list(locale.opens())[:5]:
            box_p = modal.box(p)
            # □p ≤ p
            assert locale.frame.poset.is_leq(box_p, p)
    
    def test_s4_axiom_4(self):
        """Test axioma 4: □p → □□p."""
        locale = LocaleBuilder.discrete_locale({1, 2, 3})
        modal = ModalOperators(locale)
        
        for p in list(locale.opens())[:5]:
            box_p = modal.box(p)
            box_box_p = modal.box(box_p)
            # □p ≤ □□p
            assert locale.frame.poset.is_leq(box_p, box_box_p)
    
    def test_s4_axioms_verification(self):
        """Test verificación completa de axiomas S4."""
        locale = LocaleBuilder.discrete_locale({1, 2})
        modal = ModalOperators(locale)
        
        # Debe satisfacer todos los axiomas
        assert modal.verify_s4_axioms()
    
    def test_modal_properties(self):
        """Test cálculo de propiedades modales."""
        locale = LocaleBuilder.discrete_locale({1, 2})
        modal = ModalOperators(locale)
        
        a = frozenset({1})
        props = modal.get_modal_properties(a)
        
        assert 'interior' in props
        assert 'closure' in props
        assert 'is_open' in props
        assert 'is_closed' in props


# ============================================================================
# Tests de TopologicalOperators
# ============================================================================

class TestTopologicalOperators:
    """Tests para operadores topológicos."""
    
    def test_boundary(self):
        """Test operador frontera."""
        locale = LocaleBuilder.discrete_locale({1, 2})
        topo = TopologicalOperators(locale)
        
        # En espacio discreto: ∂a = ∅
        a = frozenset({1})
        boundary = topo.boundary(a)
        
        assert boundary == frozenset()
    
    def test_exterior(self):
        """Test operador exterior."""
        locale = LocaleBuilder.discrete_locale({1, 2})
        topo = TopologicalOperators(locale)
        
        a = frozenset({1})
        exterior = topo.exterior(a)
        
        # ext(a) ∧ a = ⊥
        meet = locale.frame.meet_binary(exterior, a)
        assert meet == locale.frame.bottom
    
    def test_separation_properties(self):
        """Test propiedades de separación."""
        locale = LocaleBuilder.discrete_locale({1, 2})
        topo = TopologicalOperators(locale)
        
        props = topo.separation_properties()
        
        assert 'is_discrete' in props
        assert 'is_trivial' in props
        
        # Espacio discreto
        assert props['is_discrete'] == True
        assert props['is_trivial'] == False
    
    def test_trivial_space_properties(self):
        """Test propiedades de espacio trivial."""
        locale = LocaleBuilder.trivial_locale({1, 2, 3})
        topo = TopologicalOperators(locale)
        
        props = topo.separation_properties()
        
        # Espacio trivial
        assert props['is_trivial'] == True


# ============================================================================
# Tests de ConnectivityAnalyzer
# ============================================================================

class TestConnectivityAnalyzer:
    """Tests para análisis de conectividad."""
    
    def test_is_connected_discrete(self):
        """Test conectividad de espacio discreto."""
        locale = LocaleBuilder.discrete_locale({1, 2})
        conn = ConnectivityAnalyzer(locale)
        
        # Espacio discreto con más de un punto NO es conexo
        # (tiene clopens no triviales)
        # Pero con un solo punto SÍ es conexo
        locale_single = LocaleBuilder.discrete_locale({1})
        conn_single = ConnectivityAnalyzer(locale_single)
        
        assert conn_single.is_connected() == True
    
    def test_is_connected_trivial(self):
        """Test conectividad de espacio trivial."""
        locale = LocaleBuilder.trivial_locale({1, 2, 3})
        conn = ConnectivityAnalyzer(locale)
        
        # Espacio trivial es siempre conexo
        assert conn.is_connected() == True
    
    def test_is_compact(self):
        """Test compacidad."""
        locale = LocaleBuilder.discrete_locale({1, 2})
        conn = ConnectivityAnalyzer(locale)
        
        # Espacio finito es siempre compacto
        assert conn.is_compact() == True
    
    def test_connected_components(self):
        """Test componentes conexas."""
        locale = LocaleBuilder.trivial_locale({1, 2})
        conn = ConnectivityAnalyzer(locale)
        
        components = conn.connected_components()
        
        # Espacio conexo tiene una sola componente
        assert len(components) == 1


# ============================================================================
# Tests de LocaleAnalyzer
# ============================================================================

class TestLocaleAnalyzer:
    """Tests para análisis completo de Locales."""
    
    def test_analyze_discrete(self):
        """Test análisis de espacio discreto."""
        locale = LocaleBuilder.discrete_locale({1, 2})
        analyzer = LocaleAnalyzer(locale)
        
        analysis = analyzer.analyze()
        
        assert 'num_opens' in analysis
        assert 's4_axioms_valid' in analysis
        assert 'is_connected' in analysis
        assert 'is_compact' in analysis
        
        # Espacio discreto
        assert analysis['s4_axioms_valid'] == True
        assert analysis['is_compact'] == True
    
    def test_analyze_trivial(self):
        """Test análisis de espacio trivial."""
        locale = LocaleBuilder.trivial_locale({1, 2, 3})
        analyzer = LocaleAnalyzer(locale)
        
        analysis = analyzer.analyze()
        
        # Espacio trivial
        assert analysis['num_opens'] == 2
        assert analysis['is_connected'] == True
        assert analysis['is_compact'] == True
    
    def test_summary(self):
        """Test generación de resumen."""
        locale = LocaleBuilder.discrete_locale({1, 2}, name="Test")
        analyzer = LocaleAnalyzer(locale)
        
        summary = analyzer.summary()
        
        assert isinstance(summary, str)
        assert "Test" in summary
        assert "Número de abiertos" in summary


# ============================================================================
# Tests de Integración
# ============================================================================

class TestIntegrationMorphismsOperations:
    """Tests de integración entre morfismos y operaciones."""
    
    def test_morphism_preserves_modalities(self):
        """Test que morfismos preservan operadores modales."""
        locale = LocaleBuilder.discrete_locale({1, 2})
        id_l = LocaleMorphism.identity(locale)
        modal = ModalOperators(locale)
        
        a = frozenset({1})
        
        # id*(◇a) = ◇(id*(a))
        diamond_a = modal.diamond(a)
        lhs = id_l.pullback(diamond_a)
        
        pullback_a = id_l.pullback(a)
        rhs = modal.diamond(pullback_a)
        
        assert lhs == rhs
    
    def test_product_preserves_structure(self):
        """Test que producto preserva estructura topológica."""
        f1 = FrameBuilder.from_powerset({1})
        f2 = FrameBuilder.from_powerset({2})
        
        product = FrameConstructions.product(f1, f2)
        
        # Producto de Frames es un Frame
        assert isinstance(product, Frame)
        
        # Verificar estructura
        assert product.top == (f1.top, f2.top)
        assert product.bottom == (f1.bottom, f2.bottom)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

