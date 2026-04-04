#!/usr/bin/env python3

"""
GUI wrapper for full_env_start_export_ip_inside.sh.
Reuses the existing installer UI and behavior, but targets export-IP-only setup.
"""

import sys

from gui_install import InstallGUI


class ExportIPInstallGUI(InstallGUI):
    def __init__(self):
        super().__init__()
        self.main_script = self.script_dir / 'full_env_start_export_ip_inside.sh'
        self.log_file = self.script_dir / 'build_codesign_export_ip.log'
        self.current_step = "Initializing export-IP setup..."


def main():
    extra_args = sys.argv[1:]

    gui = ExportIPInstallGUI()
    exit_code = gui.run(extra_args)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
