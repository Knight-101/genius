"""Run Genius v5: python run.py [--share] [--port 7861]."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "artifacts" / "runtime"
RUNTIME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME / "cache"))
os.environ.setdefault("HF_HOME", str(RUNTIME / "huggingface"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(RUNTIME / "uploads"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from genius import app  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    app.get_db()
    demo = app.launch(share=args.share, server_name="127.0.0.1", server_port=args.port,
                      prevent_thread_lock=True, inbrowser=False,
                      blocked_paths=[str(ROOT / "artifacts" / "genius_db.pkl"), str(ROOT / "artifacts" / "demo_sessions.pkl"),
                                     str(ROOT / "genius"), str(ROOT / "tests")])
    links = dict(local_url=demo.local_url, public_url=demo.share_url,
                 note="Temporary share link; requires this process and machine to remain online.")
    (ROOT / "artifacts" / "live_link.json").write_text(json.dumps(links, indent=2))
    print("GENIUS_LINK " + json.dumps(links), flush=True)
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        demo.close()
