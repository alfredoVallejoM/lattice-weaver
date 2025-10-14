#!/bin/bash

# Establecer PYTHONPATH para incluir el directorio raíz del proyecto
export PYTHONPATH="$(pwd)"

# Ejecutar las pruebas unitarias para AC-3.1
python3.11 -m unittest lattice_weaver/arc_engine/test_ac31.py

