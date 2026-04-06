## Export-IP / FPGA Quick Start

Use this flow when you want FPGA export-IP generation from a clean install.

### Stanford / CMU Users

1. From the `codesign` root, run `./gui_install_FPGA.py` to install the export-IP-only environment.
2. If prompted, enter your sudo password. For a full setup, also provide your AMPL UUID when asked. If you do not already have an AMPL UUID, you can acquire one at https://dev.ampl.com/ampl/free.html 
3. After installation completes, open a new terminal and run `source full_env_start_FPGA.sh`.
4. Run the flow with FPGA target mode enabled:

```bash
run_codesign --config <your_config> --arch_opt_pipeline scalehls --target_FPGA
```

### Third-Party Institution Users

If you are not at Stanford or CMU, follow these steps to configure your Vitis HLS environment before running the installer.

1. Edit the setup scripts in `setup_scripts/third_party_environment/` to point to your institution's Vitis HLS installation. There are two scripts to configure:
   - `third_party_vitis_ScaleHLS_setup.sh` — used when running the **ScaleHLS** pipeline (requires Vitis HLS 2022.1 or compatible).
   - `third_party_vitis_StreamHLS_setup.sh` — used when running the **StreamHLS** pipeline (requires Vitis HLS 2023.2+ or compatible).

   Each script contains two commented-out options. Uncomment and adapt **one** of them:
   - **Option A (direct path):** Source the `settings64.sh` from your Vitis HLS installation directory.
   - **Option B (environment modules):** Load Vitis HLS via your institution's module system.

   After editing, remove the `exit 1` error guard at the top of each script.

2. From the `codesign` root, run the installer with the `--third_party_install` flag:

```bash
./gui_install_FPGA.py --third_party_install
```

3. If prompted, enter your sudo password and AMPL UUID as with the standard install.
4. After installation completes, open a new terminal and run `source full_env_start_FPGA.sh`.
5. Run the flow with FPGA target mode enabled:

```bash
run_codesign --config <your_config> --arch_opt_pipeline scalehls --target_FPGA
```

### Configuration

`<your_config>` is the top-level YAML key in the config files. The flow loads built-in configs from [src/yaml/codesign_cfg.yaml](src/yaml/codesign_cfg.yaml) and also merges any YAML files in [test/additional_configs](test/additional_configs). To test a new config, add a new unique top-level key in a YAML file under [test/additional_configs](test/additional_configs), set `base_cfg` to the config you want to extend, override any needed `args`, and then run `--config <that_key>`.

The export-IP outputs are written under the current run's temporary directory in [src/tmp](src/tmp). For FPGA target runs, look in the newest `src/tmp/tmp_*` directory, especially `parse_results/fpga_artifacts`, for generated bitstream or export packages.

### Notes
- This setup skips OpenROAD, CACTI build, Verilator build, and end-of-build auto-tests.
- `full_env_start_FPGA.sh` is the environment entrypoint for future terminals.
- If you are using the GUI installer, it will source `full_env_start_FPGA_inside.sh` automatically.
- The `--third_party_install` flag skips hostname-based university detection and uses the scripts in `setup_scripts/third_party_environment/` directly.
