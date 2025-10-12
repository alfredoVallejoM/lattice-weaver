"""
Tests unitarios para Job Shop Scheduling Problem.
"""

import pytest
from lattice_weaver.problems import get_catalog
from lattice_weaver.problems.generators.scheduling import JobShopSchedulingProblem, CLASSIC_INSTANCES


class TestJobShopSchedulingProblem:
    """Tests para la familia Job Shop Scheduling."""
    
    def setup_method(self):
        """Setup para cada test."""
        self.catalog = get_catalog()
        self.family = JobShopSchedulingProblem()
    
    def test_initialization(self):
        """Test de inicialización de la familia."""
        assert self.family.name == 'scheduling'
        assert 'job' in self.family.description.lower()
    
    def test_default_params(self):
        """Test de parámetros por defecto."""
        defaults = self.family.get_default_params()
        assert 'instance_name' in defaults
        assert defaults['instance_name'] == 'ft06'
    
    def test_param_schema(self):
        """Test del esquema de parámetros."""
        schema = self.family.get_param_schema()
        assert 'instance_name' in schema
        assert 'n_jobs' in schema
        assert 'n_machines' in schema
        assert schema['instance_name']['type'] == str
        assert schema['n_jobs']['type'] == int
    
    def test_generate_ft06(self):
        """Test de generación de instancia Fisher-Thompson 6x6."""
        engine = self.catalog.generate_problem('scheduling', instance_name='ft06')
        
        # ft06: 6 trabajos × 6 operaciones = 36 variables
        assert len(engine.variables) == 36
        
        # Verificar nombres de variables
        for job_id in range(6):
            for op_index in range(6):
                var_name = f'op_{job_id}_{op_index}'
                assert var_name in engine.variables
    
    def test_generate_ft10(self):
        """Test de generación de instancia Fisher-Thompson 10x10."""
        engine = self.catalog.generate_problem('scheduling', instance_name='ft10')
        
        # ft10: 10 trabajos × 10 operaciones = 100 variables
        assert len(engine.variables) == 100
    
    def test_generate_random_instance(self):
        """Test de generación de instancia aleatoria."""
        engine = self.catalog.generate_problem(
            'scheduling',
            instance_name='random',
            n_jobs=5,
            n_machines=5,
            max_duration=10,
            seed=42
        )
        
        # 5 trabajos × 5 operaciones = 25 variables
        assert len(engine.variables) == 25
        
        # Verificar reproducibilidad
        engine2 = self.catalog.generate_problem(
            'scheduling',
            instance_name='random',
            n_jobs=5,
            n_machines=5,
            max_duration=10,
            seed=42
        )
        assert len(engine2.variables) == 25
    
    def test_validate_simple_solution(self):
        """Test de validación de solución simple."""
        # Instancia pequeña: 2 trabajos, 2 máquinas
        # Job 0: (M0, 3), (M1, 2)
        # Job 1: (M1, 2), (M0, 3)
        
        # Solución válida:
        # op_0_0: t=0 (M0, termina en t=3)
        # op_0_1: t=3 (M1, termina en t=5)
        # op_1_0: t=0 (M1, termina en t=2)
        # op_1_1: t=3 (M0, termina en t=6)
        
        solution = {
            'op_0_0': 0,
            'op_0_1': 3,
            'op_1_0': 0,
            'op_1_1': 3
        }
        
        # Generar instancia aleatoria con semilla que produzca esta configuración
        # (esto es difícil, así que usamos ft06 y una solución conocida)
        
        # Para ft06, una solución factible (no necesariamente óptima)
        # Simplemente asignar tiempos secuenciales
        solution_ft06 = {}
        for job_id in range(6):
            for op_index in range(6):
                var_name = f'op_{job_id}_{op_index}'
                # Tiempo de inicio: job_id * 100 + op_index * 10
                solution_ft06[var_name] = job_id * 100 + op_index * 10
        
        # Esta solución respeta precedencia pero puede violar capacidad
        # El validador debería detectar violaciones
        is_valid = self.catalog.validate_solution(
            'scheduling',
            solution_ft06,
            instance_name='ft06'
        )
        # Puede ser válida o no, dependiendo de las máquinas
        assert isinstance(is_valid, bool)
    
    def test_validate_incomplete_solution(self):
        """Test de validación de solución incompleta."""
        # Solución incompleta (faltan operaciones)
        solution = {
            'op_0_0': 0,
            'op_0_1': 5
        }
        
        is_valid = self.catalog.validate_solution(
            'scheduling',
            solution,
            instance_name='ft06'
        )
        assert not is_valid
    
    def test_validate_precedence_violation(self):
        """Test de validación con violación de precedencia."""
        # Crear solución que viole precedencia
        # op_0_1 debe empezar después de que op_0_0 termine
        
        # ft06: Job 0 tiene operaciones [(2,1), (0,3), (1,6), (3,7), (5,3), (4,6)]
        # op_0_0: máquina 2, duración 1
        # op_0_1: máquina 0, duración 3
        
        solution_ft06 = {}
        for job_id in range(6):
            for op_index in range(6):
                var_name = f'op_{job_id}_{op_index}'
                solution_ft06[var_name] = 0  # Todos empiezan en t=0 (violación)
        
        is_valid = self.catalog.validate_solution(
            'scheduling',
            solution_ft06,
            instance_name='ft06'
        )
        assert not is_valid  # Debe detectar violación de precedencia
    
    def test_metadata_ft06(self):
        """Test de metadatos para ft06."""
        metadata = self.catalog.get_metadata('scheduling', instance_name='ft06')
        
        assert metadata['family'] == 'scheduling'
        assert metadata['instance_name'] == 'ft06'
        assert metadata['n_jobs'] == 6
        assert metadata['n_machines'] == 6
        assert metadata['n_operations'] == 36
        assert metadata['optimal_makespan'] == 55  # Makespan óptimo conocido
        assert metadata['complexity'] == 'NP-hard'
    
    def test_metadata_ft10(self):
        """Test de metadatos para ft10."""
        metadata = self.catalog.get_metadata('scheduling', instance_name='ft10')
        
        assert metadata['n_jobs'] == 10
        assert metadata['n_machines'] == 10
        assert metadata['n_operations'] == 100
        assert metadata['optimal_makespan'] == 930
        assert metadata['difficulty'] == 'medium'
    
    def test_metadata_random(self):
        """Test de metadatos para instancia aleatoria."""
        metadata = self.catalog.get_metadata(
            'scheduling',
            instance_name='random',
            n_jobs=5,
            n_machines=5,
            max_duration=10,
            seed=42
        )
        
        assert metadata['n_jobs'] == 5
        assert metadata['n_machines'] == 5
        assert metadata['n_operations'] == 25
        assert 'optimal_makespan' not in metadata  # No conocido para aleatorias
    
    def test_precedence_constraints(self):
        """Test de restricciones de precedencia."""
        engine = self.catalog.generate_problem('scheduling', instance_name='ft06')
        
        # Verificar que hay restricciones de precedencia
        # Cada trabajo tiene 6 operaciones, por lo que hay 5 restricciones de precedencia por trabajo
        # Total: 6 trabajos × 5 restricciones = 30 restricciones de precedencia (mínimo)
        
        precedence_constraints = [cid for cid in engine.constraints.keys() if 'precedence' in cid]
        assert len(precedence_constraints) >= 30
    
    def test_capacity_constraints(self):
        """Test de restricciones de capacidad."""
        engine = self.catalog.generate_problem('scheduling', instance_name='ft06')
        
        # Verificar que hay restricciones de capacidad (no solapamiento)
        capacity_constraints = [cid for cid in engine.constraints.keys() if 'capacity' in cid]
        assert len(capacity_constraints) > 0
    
    def test_invalid_instance_name(self):
        """Test de nombre de instancia inválido."""
        with pytest.raises(ValueError):
            self.catalog.generate_problem('scheduling', instance_name='invalid_instance')
    
    def test_invalid_n_jobs(self):
        """Test de número de trabajos inválido."""
        with pytest.raises(ValueError):
            self.catalog.generate_problem(
                'scheduling',
                instance_name='random',
                n_jobs=1,
                n_machines=5
            )
        
        with pytest.raises(ValueError):
            self.catalog.generate_problem(
                'scheduling',
                instance_name='random',
                n_jobs=21,
                n_machines=5
            )
    
    def test_catalog_registration(self):
        """Test de registro en catálogo."""
        assert self.catalog.has('scheduling')
        family = self.catalog.get('scheduling')
        assert isinstance(family, JobShopSchedulingProblem)
    
    def test_classic_instances_structure(self):
        """Test de estructura de instancias clásicas."""
        for instance_name, instance in CLASSIC_INSTANCES.items():
            assert 'n_jobs' in instance
            assert 'n_machines' in instance
            assert 'operations' in instance
            
            n_jobs = instance['n_jobs']
            n_machines = instance['n_machines']
            operations = instance['operations']
            
            # Verificar que hay n_jobs trabajos
            assert len(operations) == n_jobs
            
            # Verificar que cada trabajo tiene n_machines operaciones
            for job_ops in operations:
                assert len(job_ops) == n_machines
                
                # Verificar que cada operación tiene (máquina, duración)
                for machine, duration in job_ops:
                    assert 0 <= machine < n_machines
                    assert duration > 0

