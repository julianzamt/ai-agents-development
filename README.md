# Proyecto de agentes de IA

Proyecto integrador que evoluciona con las clases del curso. El código compartido vive en `src/agentes_ia`; los notebooks originales están en `clases/`.

## Instalación

Desde la raíz:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

La instalación editable registra el comando `client` y permite que los cambios en `src/` se apliquen sin reinstalar. Las dependencias se declaran en `requirements.txt` y se incorporan al paquete mediante `pyproject.toml`.

Copiá `.env.example` a `.env` y configurá el proveedor y sus credenciales. El cliente busca ese archivo en la raíz del proyecto.

## Uso

```bash
client --system "Respondé brevemente" --temp 0.3 --max_tokens 100
client "Me llamo Julián"
client "¿Cómo me llamo?"
client --batch "¿Capital de Francia?" "¿Capital de Argentina?"
client --help
```

También se puede ejecutar `python -m agentes_ia`. Los parámetros sin prompt se guardan; junto con un prompt o batch reemplazan los guardados solo para esa ejecución. El batch usa la configuración actual, pero no usa ni modifica el historial.

El estado local se guarda en `data/client_state.json`, excluido de Git. `--history archivo.json` selecciona otro archivo; las rutas relativas se resuelven desde el directorio actual.

- `--reset`: borra el historial.
- `--reset-config`: restaura la configuración.
- `--reset-all`: restaura ambos.

## Organización

- `src/agentes_ia/cli.py`: argumentos y flujo del comando.
- `src/agentes_ia/llm/`: cliente, proveedores, respuestas y persistencia.
- `tests/`: pruebas sin llamadas reales a las APIs.
- `clases/clase2/` y `clases/clase3/`: material del curso.
- `data/`: estado local.
- `scripts/install-client.sh`: reinstala el comando en `.venv`.

Los notebooks conservan el material de clase; la implementación de clase 3 todavía no se incorporó al paquete. Para importar el cliente desde Python:

```python
from agentes_ia.llm import LLMClient
```

## Pruebas

```bash
python -m unittest discover -s tests
```
