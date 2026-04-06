## Export-IP / FPGA Quick Start

Use this flow when you want FPGA export-IP generation from a clean install.

1. From the `codesign` root, run `./gui_install_export_ip.py` to install the export-IP-only environment.
2. If prompted, enter your sudo password. For a full setup, also provide your AMPL UUID when asked.
    a. If you do not already have an AMPL UUID, you can acquire one at https://dev.ampl.com/ampl/free.html 
3. After installation completes, open a new terminal and run `source full_env_start_FPGA.sh`.
4. Run the flow with FPGA target mode enabled:

	```bash
	run_codesign --config <your_config> --arch_opt_pipeline scalehls --target_FPGA
	```

`<your_config>` is the top-level YAML key in the config files. The flow loads built-in configs from [src/yaml/codesign_cfg.yaml](src/yaml/codesign_cfg.yaml) and also merges any YAML files in [test/additional_configs](test/additional_configs). To test a new config, add a new unique top-level key in a YAML file under [test/additional_configs](test/additional_configs), set `base_cfg` to the config you want to extend, override any needed `args`, and then run `--config <that_key>`.

The export-IP outputs are written under the current run's temporary directory in [src/tmp](src/tmp). For FPGA target runs, look in the newest `src/tmp/tmp_*` directory, especially `parse_results/fpga_artifacts`, for generated bitstream or export packages.

Notes:
- This setup skips OpenROAD, CACTI build, Verilator build, and end-of-build auto-tests.
- `full_env_start_FPGA.sh` is the environment entrypoint for future terminals.
- If you are using the GUI installer, it will source `full_env_start_FPGA_inside.sh` automatically.
