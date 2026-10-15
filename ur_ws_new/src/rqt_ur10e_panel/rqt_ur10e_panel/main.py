"""Standalone entry point for the UR10e rqt panel."""
import sys
from rqt_gui.main import Main


def main():
    plugin = "rqt_ur10e_panel.ur10e_panel.UR10ePanel"
    main_obj = Main(filename=plugin)
    sys.exit(main_obj.main(standalone=plugin))


if __name__ == "__main__":
    main()
