#!/usr/bin/env python3
"""
Script Universal de Sincronización de Agentes con GitHub

Este script maneja la sincronización completa de un agente con su track en GitHub:
1. Fetch de cambios remotos
2. Comparación de estado local vs remoto
3. Descarga y extracción de tar.gz si existe
4. Merge de cambios
5. Actualización de estado del agente

Autor: Equipo LatticeWeaver
Versión: 1.0
Fecha: Octubre 2025
"""

import subprocess
import json
import sys
import tarfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict


class AgentSynchronizer:
    """Gestor de sincronización de agentes con GitHub"""
    
    def __init__(self, agent_id: str, track: str, repo_path: Path):
        """
        Inicializa el sincronizador
        
        Parameters
        ----------
        agent_id : str
            ID del agente (ej: agent-track-a)
        track : str
            Nombre del track (ej: track-a-core)
        repo_path : Path
            Ruta al repositorio local
        """
        self.agent_id = agent_id
        self.track = track
        self.repo_path = repo_path
        self.status_file = repo_path / ".agent-status" / f"{agent_id}.json"
        self.tarball_path = repo_path / "releases" / f"{track}.tar.gz"
    
    def log(self, message: str, level: str = "INFO"):
        """Registra un mensaje con timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌"
        }.get(level, "ℹ️")
        
        print(f"[{timestamp}] {prefix} {message}")
    
    def run_git_command(self, args: list) -> Tuple[bool, str, str]:
        """
        Ejecuta un comando git
        
        Parameters
        ----------
        args : list
            Argumentos del comando git
        
        Returns
        -------
        Tuple[bool, str, str]
            (success, stdout, stderr)
        """
        result = subprocess.run(
            ["git"] + args,
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    
    def fetch_remote(self) -> bool:
        """Fetch cambios del remoto"""
        self.log(f"Fetching cambios de origin/{self.track}...")
        
        success, stdout, stderr = self.run_git_command(["fetch", "origin", self.track])
        
        if not success:
            self.log(f"Error en git fetch: {stderr}", "ERROR")
            return False
        
        self.log("Fetch completado", "SUCCESS")
        return True
    
    def get_commits(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Obtiene commits local y remoto
        
        Returns
        -------
        Tuple[Optional[str], Optional[str]]
            (local_commit, remote_commit)
        """
        # Commit local
        success, local_commit, _ = self.run_git_command(["rev-parse", self.track])
        if not success:
            local_commit = None
        
        # Commit remoto
        success, remote_commit, _ = self.run_git_command(["rev-parse", f"origin/{self.track}"])
        if not success:
            remote_commit = None
        
        return local_commit, remote_commit
    
    def check_for_updates(self) -> bool:
        """
        Verifica si hay actualizaciones remotas
        
        Returns
        -------
        bool
            True si hay actualizaciones
        """
        local_commit, remote_commit = self.get_commits()
        
        if local_commit is None:
            self.log("Rama local no existe, se creará", "WARNING")
            return True
        
        if remote_commit is None:
            self.log("Rama remota no existe", "ERROR")
            return False
        
        if local_commit == remote_commit:
            self.log("Código local está actualizado", "SUCCESS")
            self.log(f"   Commit: {local_commit[:8]}")
            return False
        
        self.log("Hay actualizaciones remotas disponibles", "WARNING")
        self.log(f"   Local:  {local_commit[:8]}")
        self.log(f"   Remoto: {remote_commit[:8]}")
        return True
    
    def check_for_tarball(self) -> bool:
        """
        Verifica si existe un tarball en el repositorio
        
        Returns
        -------
        bool
            True si existe tarball
        """
        if not self.tarball_path.exists():
            self.log("No se encontró tarball en releases/", "INFO")
            return False
        
        self.log(f"Tarball encontrado: {self.tarball_path.name}", "SUCCESS")
        return True
    
    def extract_tarball(self) -> bool:
        """
        Extrae el tarball del track
        
        Returns
        -------
        bool
            True si la extracción fue exitosa
        """
        if not self.check_for_tarball():
            return False
        
        self.log(f"Extrayendo {self.tarball_path.name}...")
        
        try:
            # Crear directorio temporal
            temp_dir = self.repo_path / "temp_extract"
            temp_dir.mkdir(exist_ok=True)
            
            # Extraer tarball
            with tarfile.open(self.tarball_path, "r:gz") as tar:
                tar.extractall(temp_dir)
            
            # Copiar contenido al repositorio
            extracted_dir = temp_dir / self.track
            if extracted_dir.exists():
                # Copiar archivos
                for item in extracted_dir.iterdir():
                    dest = self.repo_path / item.name
                    if item.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
                
                self.log(f"Contenido extraído correctamente", "SUCCESS")
            else:
                self.log(f"Directorio {self.track} no encontrado en tarball", "ERROR")
                return False
            
            # Limpiar
            shutil.rmtree(temp_dir)
            
            return True
        
        except Exception as e:
            self.log(f"Error extrayendo tarball: {e}", "ERROR")
            return False
    
    def merge_remote(self) -> bool:
        """
        Merge cambios remotos
        
        Returns
        -------
        bool
            True si el merge fue exitoso
        """
        self.log(f"Merging origin/{self.track}...")
        
        # Verificar si estamos en la rama correcta
        success, current_branch, _ = self.run_git_command(["branch", "--show-current"])
        
        if not success or current_branch != self.track:
            self.log(f"Cambiando a rama {self.track}...")
            success, _, stderr = self.run_git_command(["checkout", self.track])
            
            if not success:
                # Crear rama si no existe
                self.log(f"Creando rama {self.track}...")
                success, _, stderr = self.run_git_command(["checkout", "-b", self.track, f"origin/{self.track}"])
                
                if not success:
                    self.log(f"Error creando rama: {stderr}", "ERROR")
                    return False
        
        # Merge
        success, stdout, stderr = self.run_git_command(["merge", f"origin/{self.track}"])
        
        if not success:
            self.log(f"Error en merge: {stderr}", "ERROR")
            
            # Verificar si es un conflicto
            if "CONFLICT" in stderr:
                self.log("Conflictos detectados. Resolución manual requerida.", "ERROR")
                self.update_agent_flag("HAS_MERGE_CONFLICTS", True)
            
            return False
        
        self.log("Merge completado exitosamente", "SUCCESS")
        return True
    
    def load_agent_status(self) -> Dict:
        """Carga el estado del agente"""
        if not self.status_file.exists():
            return {}
        
        with open(self.status_file) as f:
            return json.load(f)
    
    def save_agent_status(self, status: Dict):
        """Guarda el estado del agente"""
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    def update_agent_flag(self, flag: str, value: bool):
        """Actualiza un flag del agente"""
        status = self.load_agent_status()
        
        if "flags" not in status:
            status["flags"] = {}
        
        status["flags"][flag] = value
        status["last_sync"] = datetime.now().isoformat()
        
        self.save_agent_status(status)
        self.log(f"Flag {flag} actualizado a {value}")
    
    def sync(self) -> bool:
        """
        Ejecuta sincronización completa
        
        Returns
        -------
        bool
            True si la sincronización fue exitosa
        """
        self.log(f"🔄 Iniciando sincronización de {self.agent_id} con {self.track}")
        self.log("=" * 60)
        
        # 1. Fetch
        if not self.fetch_remote():
            return False
        
        # 2. Verificar actualizaciones
        has_updates = self.check_for_updates()
        
        # 3. Si no hay actualizaciones, verificar tarball
        if not has_updates:
            if self.check_for_tarball():
                self.log("Tarball disponible pero código ya actualizado")
            
            self.update_agent_flag("SYNCED", True)
            self.log("=" * 60)
            self.log("✅ Sincronización completada (sin cambios)", "SUCCESS")
            return True
        
        # 4. Extraer tarball si existe
        if self.check_for_tarball():
            if not self.extract_tarball():
                self.log("Continuando con merge de Git...", "WARNING")
        
        # 5. Merge
        if not self.merge_remote():
            self.update_agent_flag("SYNCED", False)
            return False
        
        # 6. Actualizar estado
        self.update_agent_flag("SYNCED", True)
        self.update_agent_flag("HAS_MERGE_CONFLICTS", False)
        
        self.log("=" * 60)
        self.log("✅ Sincronización completada exitosamente", "SUCCESS")
        
        return True


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sincroniza un agente con su track en GitHub"
    )
    parser.add_argument(
        "--agent-id",
        required=True,
        help="ID del agente (ej: agent-track-a)"
    )
    parser.add_argument(
        "--track",
        required=True,
        help="Nombre del track (ej: track-a-core)"
    )
    parser.add_argument(
        "--repo-path",
        type=Path,
        default=Path.cwd(),
        help="Ruta al repositorio (default: directorio actual)"
    )
    
    args = parser.parse_args()
    
    # Crear sincronizador
    syncer = AgentSynchronizer(
        agent_id=args.agent_id,
        track=args.track,
        repo_path=args.repo_path
    )
    
    # Ejecutar sincronización
    success = syncer.sync()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

