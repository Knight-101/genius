"""Refresh the portable source zip and notebook from this release's source tree."""
import base64
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / "Genius_v5_request_based.ipynb"
if not nb_path.exists():
    nb_path = ROOT.parent / nb_path.name
files = [p for p in ROOT.rglob("*") if p.is_file()
         and p.suffix != ".ipynb" and p.name != "Genius_v5_source.zip"
         and not any(part in p.relative_to(ROOT).parts for part in
                     ("artifacts", "__pycache__", ".pytest_cache", ".git", ".gradio", ".DS_Store", ".venv", ".env"))]
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for path in files:
        z.write(path, path.relative_to(ROOT))
(nb_path.parent / "Genius_v5_source.zip").write_bytes(buffer.getvalue())
nb = json.loads(nb_path.read_text())
code = "# Embedded Genius v5 source, samples and tests\nGENIUS_CODE = " + repr(base64.b64encode(buffer.getvalue()).decode()) + "\n"
nb["cells"][2]["source"] = code.splitlines(keepends=True)
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"Packaged {len(files)} source/sample files; notebook and source zip refreshed.")
