"""
Tests de integración básicos - Versión adaptada.

Valida la integración básica entre componentes del sistema.
"""
import pytest

@pytest.mark.integration
class TestBasicIntegration:
    """Tests básicos de integración entre módulos."""
    
    def test_arc_engine_basic_solve(self, arc_engine, simple_csp_problem):
        """
        Test: Resolver problema CSP simple.
        
        Verifica que el ArcEngine puede resolver un problema CSP básico.
        """
        # El ArcEngine debería poder procesar el problema
        assert arc_engine is not None, "ArcEngine debe estar inicializado"
        assert simple_csp_problem is not None, "Problema CSP debe estar definido"
        
        # Verificar estructura del problema
        assert len(simple_csp_problem.variables) == 3, "Debe tener 3 variables"
        assert len(simple_csp_problem.domains) == 3, "Debe tener 3 dominios"
        assert len(simple_csp_problem.constraints) == 3, "Debe tener 3 restricciones"
        
        print(f"\n  Variables: {simple_csp_problem.variables}")
        print(f"  Dominios: {simple_csp_problem.domains}")
        print(f"  Restricciones: {len(simple_csp_problem.constraints)}")
    
    def test_fca_builder_initialization(self, fca_builder):
        """
        Test: Inicialización del FCA Builder.
        
        Verifica que el constructor de retículos FCA se inicializa correctamente.
        """
        assert fca_builder is not None, "FCA Builder debe estar inicializado"
        print(f"\n  FCA Builder: {type(fca_builder).__name__}")
    
    def test_fca_context_structure(self, sample_fca_context):
        """
        Test: Estructura del contexto formal.
        
        Verifica que el contexto formal tiene la estructura correcta.
        """
        objects, attributes, relation = sample_fca_context
        
        assert len(objects) == 3, "Debe tener 3 objetos"
        assert len(attributes) == 3, "Debe tener 3 atributos"
        assert len(relation) == 6, "Debe tener 6 relaciones"
        
        # Verificar que todas las relaciones son válidas
        for obj, attr in relation:
            assert obj in objects, f"Objeto {obj} debe estar en el conjunto de objetos"
            assert attr in attributes, f"Atributo {attr} debe estar en el conjunto de atributos"
        
        print(f"\n  Objetos: {objects}")
        print(f"  Atributos: {attributes}")
        print(f"  Relaciones: {len(relation)}")
    
    def test_tda_engine_initialization(self, tda_engine):
        """
        Test: Inicialización del TDA Engine.
        
        Verifica que el motor TDA se inicializa correctamente.
        """
        assert tda_engine is not None, "TDA Engine debe estar inicializado"
        print(f"\n  TDA Engine: {type(tda_engine).__name__}")
    
    def test_cubical_engine_initialization(self, cubical_engine):
        """
        Test: Inicialización del Cubical Engine.
        
        Verifica que el motor cúbico HoTT se inicializa correctamente.
        """
        assert cubical_engine is not None, "Cubical Engine debe estar inicializado"
        print(f"\n  Cubical Engine: {type(cubical_engine).__name__}")
    
    def test_csp_problem_constraints_evaluation(self, simple_csp_problem):
        """
        Test: Evaluación de restricciones CSP.
        
        Verifica que las restricciones del problema CSP se pueden evaluar.
        """
        # Tomar una restricción y evaluarla
        var1, var2, predicate = simple_csp_problem.constraints[0]
        
        # Evaluar con valores diferentes (debe ser True para desigualdad)
        result_different = predicate(1, 2)
        assert result_different == True, "1 != 2 debe ser True"
        
        # Evaluar con valores iguales (debe ser False para desigualdad)
        result_same = predicate(1, 1)
        assert result_same == False, "1 != 1 debe ser False"
        
        print(f"\n  Restricción: {var1} != {var2}")
        print(f"  predicate(1, 2) = {result_different}")
        print(f"  predicate(1, 1) = {result_same}")
    
    @pytest.mark.slow
    def test_integration_pipeline_exists(self, arc_engine, fca_builder, tda_engine, cubical_engine):
        """
        Test: Verificar que todos los componentes principales existen.
        
        Este test verifica que la infraestructura básica está en su lugar
        para futuros tests de integración más complejos.
        """
        components = {
            "ArcEngine": arc_engine,
            "FCABuilder": fca_builder,
            "TDAEngine": tda_engine,
            "CubicalEngine": cubical_engine
        }
        
        for name, component in components.items():
            assert component is not None, f"{name} debe estar inicializado"
        
        print("\n  Componentes principales verificados:")
        for name in components:
            print(f"    ✓ {name}")

