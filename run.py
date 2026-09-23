#!/usr/bin/env python3
"""One command: parquet -> validated CSVs + standalone offline dashboard."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import time
import webbrowser

from moneygraph.data import DataError
from moneygraph.pipeline import run


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"), help="Папка с тремя parquet")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--config", type=Path, help="JSON с изменёнными порогами")
    ap.add_argument("--serve", action="store_true", help="После расчёта открыть локальный HTTP-сервер")
    ap.add_argument("--open", action="store_true", help="Открыть браузер")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--explain", help="Показать карточку gid из уже рассчитанного graph.json")
    args = ap.parse_args()
    try:
        if args.explain:
            payload = json.loads((args.out / "graph.json").read_text(encoding="utf-8"))
            node = next((n for n in payload["nodes"] if n["gid"] == args.explain), None)
            if node is None:
                raise DataError(f"gid {args.explain} не найден")
            print(json.dumps({"node": node, "candidates": payload["role_candidates"][args.explain]}, ensure_ascii=False, indent=2))
            return 0
        started = time.perf_counter()
        audit = run(args.data, args.out, args.config)
        elapsed = time.perf_counter() - started
        print(f"Готово за {elapsed:.2f} с: {audit['nodes']} узлов, {audit['edges']} рёбер, {audit['clusters']} кластеров.")
        print(f"Роли: {audit['roles']}")
        print(f"Выход: {args.out.resolve()}")
        if args.serve:
            url = f"http://127.0.0.1:{args.port}/dashboard.html"
            handler = partial(SimpleHTTPRequestHandler, directory=str(args.out.resolve()))
            with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
                print(f"Интерфейс: {url}\nДля остановки: Ctrl+C", flush=True)
                if args.open:
                    webbrowser.open(url)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
        elif args.open:
            webbrowser.open((args.out / "dashboard.html").resolve().as_uri())
        return 0
    except (DataError, OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
