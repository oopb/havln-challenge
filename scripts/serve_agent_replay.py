"""Open the completed HA-VLN run in a read-only browser replay UI."""
import argparse
from pathlib import Path

from havln_agent.replay_server import serve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--output-dir', type=Path,
                       help='Completed run directory (contains traces/)')
    group.add_argument('--archive', type=Path,
                       help='Diagnostic ZIP (use --frames all for full playback)')
    p.add_argument('--host', default='127.0.0.1',
                   help='Bind address; loopback only by default')
    p.add_argument('--port', type=int, default=8765)
    args = p.parse_args()
    if not 1 <= args.port <= 65535:
        p.error('--port must be 1-65535')
    serve(args.output_dir or args.archive, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
