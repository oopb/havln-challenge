"""Run the OBSERVE/MOVE API Agent using the released HA-VLN image."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from havln_agent.export import main


if __name__ == '__main__':
    main()
