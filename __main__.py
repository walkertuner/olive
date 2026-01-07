import sys
from olive import train, ingest

def main():
    if len(sys.argv) < 2:
        print("usage: python -m olive {train|ingest} [args]")
        sys.exit(1)

    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # strip subcommand

    if cmd == "train":
        train.main()
    elif cmd == "ingest":
        ingest.main()
    else:
        raise SystemExit(f"unknown command: {cmd}")

if __name__ == "__main__":
    main()
