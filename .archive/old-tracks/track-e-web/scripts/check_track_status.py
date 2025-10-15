#!/usr/bin/env python3
"""
Script para Verificar Estado del Track

Lee el estado actual del track desde GitHub y lo compara con el estado local.

Autor: Equipo LatticeWeaver
Versión: 1.0
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class TrackStatusChecker:
    """Verificador de estado de tracks"""
    
    def __init__(self, agent_id: str, track: str, repo_path: Path):
        self.agent_id = agent_id
        self.track = track
        self.repo_path = repo_path
        self.status_file = repo_path / ".agent-status" / f"{agent_id}.json"
    
    def load_status(self) -> Optional[Dict]:
        """Carga el estado del agente"""
        if not self.status_file.exists():
            print(f"⚠️  Archivo de estado no encontrado: {self.status_file}")
            return None
        
        with open(self.status_file) as f:
            return json.load(f)
    
    def display_status(self, status: Dict):
        """Muestra el estado del track de forma legible"""
        print("\n" + "=" * 70)
        print(f"📊 ESTADO DEL TRACK: {self.track.upper()}")
        print("=" * 70)
        
        # Información básica
        print(f"\n🤖 Agente: {status.get('agent_id', 'N/A')}")
        print(f"📅 Última actualización: {status.get('last_update', 'N/A')}")
        
        # Progreso
        progress = status.get('progress', {})
        print(f"\n📈 Progreso:")
        print(f"   Semana actual: {progress.get('current_week', 0)}/{progress.get('total_weeks', 0)}")
        print(f"   Porcentaje: {progress.get('percentage', 0):.1f}%")
        print(f"   Semanas completadas: {progress.get('weeks_completed', 0)}")
        
        # Tarea actual
        current_task = status.get('current_task', {})
        if current_task:
            print(f"\n🎯 Tarea actual:")
            print(f"   ID: {current_task.get('id', 'N/A')}")
            print(f"   Título: {current_task.get('title', 'N/A')}")
            print(f"   Estado: {current_task.get('status', 'N/A')}")
            if 'started_at' in current_task:
                print(f"   Iniciada: {current_task['started_at']}")
        else:
            print(f"\n🎯 Tarea actual: Ninguna")
        
        # Tareas completadas
        completed = status.get('completed_tasks', [])
        print(f"\n✅ Tareas completadas: {len(completed)}")
        if completed:
            print("   Últimas 3:")
            for task in completed[-3:]:
                print(f"   - {task.get('id')}: {task.get('title')}")
        
        # Flags
        flags = status.get('flags', {})
        print(f"\n🚩 Flags:")
        for flag, value in flags.items():
            icon = "✅" if value else "❌"
            print(f"   {icon} {flag}: {value}")
        
        # Métricas
        metrics = status.get('metrics', {})
        if metrics:
            print(f"\n📊 Métricas:")
            print(f"   Tests: {metrics.get('tests_passed', 0)}/{metrics.get('tests_total', 0)}")
            print(f"   Cobertura: {metrics.get('coverage', 0):.1f}%")
            print(f"   Líneas de código: {metrics.get('lines_of_code', 0):,}")
        
        print("\n" + "=" * 70)
    
    def get_next_task(self, status: Dict) -> Optional[Dict]:
        """Identifica la siguiente tarea a realizar"""
        current_week = status.get('progress', {}).get('current_week', 1)
        
        # Aquí se debería cargar el plan del track y encontrar la siguiente tarea
        # Por ahora, retornamos un placeholder
        
        return {
            "id": f"{self.track.upper()[6]}-W{current_week}-T1",
            "title": "Siguiente tarea según plan",
            "week": current_week,
            "estimated_hours": 8
        }
    
    def check(self):
        """Ejecuta la verificación de estado"""
        status = self.load_status()
        
        if status is None:
            print("\n❌ No se pudo cargar el estado del track")
            print(f"   Archivo esperado: {self.status_file}")
            print("\n💡 Sugerencia: Ejecuta primero sync_agent.py para inicializar")
            return False
        
        self.display_status(status)
        
        # Mostrar siguiente tarea
        next_task = self.get_next_task(status)
        if next_task:
            print(f"\n🎯 Siguiente tarea sugerida:")
            print(f"   ID: {next_task['id']}")
            print(f"   Título: {next_task['title']}")
            print(f"   Semana: {next_task['week']}")
            print(f"   Estimación: {next_task['estimated_hours']}h")
        
        return True


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verifica el estado de un track"
    )
    parser.add_argument(
        "--agent-id",
        required=True,
        help="ID del agente"
    )
    parser.add_argument(
        "--track",
        required=True,
        help="Nombre del track"
    )
    parser.add_argument(
        "--repo-path",
        type=Path,
        default=Path.cwd(),
        help="Ruta al repositorio"
    )
    
    args = parser.parse_args()
    
    checker = TrackStatusChecker(
        agent_id=args.agent_id,
        track=args.track,
        repo_path=args.repo_path
    )
    
    success = checker.check()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

