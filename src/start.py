import argparse

import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(description="Start Qwen ASR server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    from .server import app

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
