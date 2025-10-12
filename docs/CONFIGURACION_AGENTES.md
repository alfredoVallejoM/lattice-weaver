# Configuración de Agentes Autónomos - LatticeWeaver

**Versión:** 1.0  
**Fecha:** 12 de Octubre, 2025

---

## Configuración de Credenciales GitHub

Cada agente necesita configurar sus credenciales de GitHub para poder sincronizarse con el repositorio.

### Paso 1: Crear Archivo de Credenciales

```bash
# Copiar template
cp .github_credentials.template .github_credentials

# Editar con tus credenciales
nano .github_credentials
```

### Paso 2: Rellenar Credenciales

Edita `.github_credentials` con:

```bash
GITHUB_USER=alfredoVallejoM
GITHUB_TOKEN=<TU_TOKEN_AQUI>
GITHUB_REPO=lattice-weaver
```

**IMPORTANTE:** El archivo `.github_credentials` está en `.gitignore` y NUNCA se subirá a GitHub por seguridad.

### Paso 3: Configurar Git

```bash
# Cargar credenciales
source .github_credentials

# Configurar Git
git config --global user.name "$GITHUB_USER"
git config --global user.email "$GITHUB_USER@users.noreply.github.com"

# Configurar credential helper
git config --global credential.helper store
echo "https://$GITHUB_USER:$GITHUB_TOKEN@github.com" > ~/.git-credentials
```

### Paso 4: Verificar Configuración

```bash
# Verificar que puedes hacer pull
git pull origin main

# Verificar que puedes hacer push
git push origin main
```

---

## Uso en Scripts de Agentes

Los scripts de sincronización (`scripts/sync_agent.py`) cargarán automáticamente las credenciales desde `.github_credentials`.

```python
# scripts/sync_agent.py

import os

def load_github_credentials():
    """Carga credenciales desde .github_credentials"""
    creds = {}
    
    creds_file = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        '.github_credentials'
    )
    
    if os.path.exists(creds_file):
        with open(creds_file) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    creds[key] = value
    
    return creds

# Uso
creds = load_github_credentials()
github_user = creds.get('GITHUB_USER')
github_token = creds.get('GITHUB_TOKEN')
```

---

## Seguridad

### ✅ Buenas Prácticas

1. **NUNCA** commitear `.github_credentials`
2. **SIEMPRE** usar `.github_credentials.template` como referencia
3. **ROTAR** tokens periódicamente
4. **REVOCAR** tokens si se comprometen

### ❌ Evitar

1. NO subir credenciales a GitHub
2. NO compartir tokens en texto plano
3. NO usar el mismo token para múltiples propósitos

---

## Tokens de Acceso Personal (PAT)

### Crear un Token

1. Ir a https://github.com/settings/tokens
2. Click en "Generate new token" → "Generate new token (classic)"
3. Nombre: `LatticeWeaver Agent Token`
4. Scopes necesarios:
   - ✅ `repo` (acceso completo a repositorios)
   - ✅ `workflow` (actualizar GitHub Actions)
5. Click en "Generate token"
6. **COPIAR** el token inmediatamente (solo se muestra una vez)

### Revocar un Token

Si un token se compromete:

1. Ir a https://github.com/settings/tokens
2. Encontrar el token comprometido
3. Click en "Delete"
4. Generar un nuevo token
5. Actualizar `.github_credentials` en todos los agentes

---

## Configuración por Track

Cada paquete de track incluye su propio `.github_credentials.template`. Los agentes deben:

1. Extraer el paquete del track
2. Copiar `.github_credentials.template` a `.github_credentials`
3. Rellenar con credenciales reales
4. Ejecutar `source .github_credentials`
5. Comenzar desarrollo

---

## Troubleshooting

### Error: "Authentication failed"

```bash
# Verificar credenciales
cat ~/.git-credentials

# Reconfigurar
source .github_credentials
echo "https://$GITHUB_USER:$GITHUB_TOKEN@github.com" > ~/.git-credentials
```

### Error: "Permission denied"

```bash
# Verificar que el token tiene permisos correctos
# Regenerar token con scopes: repo, workflow
```

### Error: "Repository not found"

```bash
# Verificar nombre del repositorio
echo $GITHUB_REPO

# Debe ser: lattice-weaver
```

---

## Ejemplo Completo

```bash
# 1. Extraer paquete de track
tar -xzf track-a-core.tar.gz
cd track-a-core/

# 2. Configurar credenciales
cp .github_credentials.template .github_credentials
nano .github_credentials  # Rellenar con credenciales reales

# 3. Configurar Git
source .github_credentials
git config --global user.name "$GITHUB_USER"
git config --global user.email "$GITHUB_USER@users.noreply.github.com"
git config --global credential.helper store
echo "https://$GITHUB_USER:$GITHUB_TOKEN@github.com" > ~/.git-credentials

# 4. Clonar repositorio
git clone https://github.com/$GITHUB_USER/$GITHUB_REPO.git
cd $GITHUB_REPO/

# 5. Verificar
git pull origin main
git status

# 6. Comenzar desarrollo
python scripts/sync_agent.py --agent-id agent-track-a --track track-a-core
python scripts/start_development.py --agent-id agent-track-a
```

---

## Automatización

Para automatizar la configuración en todos los tracks:

```bash
# Script de configuración automática
# setup_all_tracks.sh

#!/bin/bash

# NOTA: Rellenar con tus credenciales reales
GITHUB_USER="tu_usuario"
GITHUB_TOKEN="tu_token"
GITHUB_REPO="lattice-weaver"

for track in track-*; do
    echo "Configurando $track..."
    cd $track
    
    # Crear .github_credentials
    cat > .github_credentials <<EOF
GITHUB_USER=$GITHUB_USER
GITHUB_TOKEN=$GITHUB_TOKEN
GITHUB_REPO=$GITHUB_REPO
EOF
    
    # Configurar Git
    source .github_credentials
    git config --global user.name "$GITHUB_USER"
    git config --global credential.helper store
    echo "https://$GITHUB_USER:$GITHUB_TOKEN@github.com" > ~/.git-credentials
    
    cd ..
done

echo "✅ Todos los tracks configurados"
```

---

## Conclusión

La configuración de credenciales es **crítica** para el funcionamiento de los agentes autónomos. Siguiendo esta guía, cada agente podrá:

- ✅ Sincronizarse con GitHub
- ✅ Hacer pull de cambios
- ✅ Hacer push de su trabajo
- ✅ Colaborar con otros agentes

**Seguridad primero:** Nunca exponer credenciales en repositorios públicos.

