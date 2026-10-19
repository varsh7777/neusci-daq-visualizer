import argparse
import sys

from . import config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="neusci-daq", description=__doc__)
    ap.add_argument("-c", "--config", default="config.toml")
    ap.add_argument("--backend", choices=["nidaqmx", "multichannelio", "simulated"],
                    help="override the backend in the config")
    ap.add_argument("--list-devices", action="store_true", help="list NI devices and exit")
    args = ap.parse_args(argv)

    if args.list_devices:
        import nidaqmx.system
        for d in nidaqmx.system.System.local().devices:
            print(f"{d.name}: {d.product_type}  AI: {[c.name for c in d.ai_physical_chans]}")
        return 0

    cfg = config.load(args.config)
    if args.backend:
        cfg.backend = args.backend
        config.validate(cfg)
    from .gui import run
    return run(cfg, config_path=args.config)


if __name__ == "__main__":
    sys.exit(main())
