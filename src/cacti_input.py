config_values = {
    # Cache size
    "cache_size": -1,

    # Power gating
    "Array_Power_Gating": "false",
    "WL_Power_Gating": "false",
    "CL_Power_Gating": "false",
    "Bitline_floating": "false",
    "Interconnect_Power_Gating": "false",
    "Power_Gating_Performance_Loss": 0.01,

    # Line size
    "block_size": -1,

    # To model Fully Associative cache, set associativity to zero
    "associativity": -1,
    "read_write_port": 1,
    "exclusive_read_port": 0,
    "exclusive_write_port": 0,
    "single_ended_read_ports": 0,

    # Multiple banks connected using a bus
    "UCA_bank_count": 1,
    "technology": 0.090,

    # Following three parameters are meaningful only for main memories
    "page_size": 8192,
    "burst_length": 8,
    "internal_prefetch_width": 8,

    # Following parameter can have one of five values
    "Data_array_cell_type": "itrs-hp",

    # Following parameter can have one of three values
    "Data_array_peripheral_type": "itrs-hp",

    # Following parameter can have one of five values
    "Tag_array_cell_type": "itrs-hp",

    # Following parameter can have one of three values
    "Tag_array_peripheral_type": "itrs-hp",

    # Bus width include data bits and address bits required by the decoder
    "output/input_bus_width": 512,

    # 300-400 in steps of 10
    "operating_temperature": 360,

    # Type of memory
    "cache_type": "none",

    # Tag size
    "tag_size": "default",

    # Access mode
    "access_mode": "normal",

    # Design objective for UCA
    "design_objective_weight_delay_dynamic_power_leakage_power_cycle_time_area": "0:0:0:100:0",

    # Percentage deviation from the minimum value
    "deviate_delay_dynamic_power_leakage_power_cycle_time_area": "20:100000:100000:100000:100000",

    # Objective for NUCA
    "NUCAdesign_objective_weight_delay_dynamic_power_leakage_power_cycle_time_area": "100:100:0:0:100",
    "NUCAdeviate_delay_dynamic_power_leakage_power_cycle_time_area": "10:10000:10000:10000:10000",

    # Set optimize tag
    "Optimize_ED_or_ED^2": "ED^2",
    "Cache_model_NUCA_UCA": "UCA",

    # In order for CACTI to find the optimal NUCA bank value
    "NUCA_bank_count": 0,

    # Wire signaling
    "Wire_signaling": "Global_30",
    "Wire_inside_mat": "semi-global",
    "Wire_outside_mat": "semi-global",
    "Interconnect_projection": "conservative",

    # Contention in network
    "Core_count": 8,
    "Cache_level": "L3",
    "Add_ECC": "true",
    "Print_level": "DETAILED",
    "Print_input_parameters": "true",
    # Force cache config
    "Force_cache_config": "false",
    "Ndwl": 1,
    "Ndbl": 1,
    "Nspd": 0,
    "Ndcm": 1,
    "Ndsam1": 0,
    "Ndsam2": 0,

    # Default CONFIGURATION values for baseline external IO parameters to DRAM
    # Memory Type
    "dram_type": "DDR3",

    # Memory State
    "io_state": "WRITE",

    # Address bus timing
    "addr_timing": 1.0,

    # Memory Density
    "mem_density": "4 Gb",

    # IO frequency
    "bus_freq": "800 MHz",

    # Duty Cycle
    "duty_cycle": 1.0,

    # Activity factor for Data
    "activity_dq": 1.0,

    # Activity factor for Control/Address
    "activity_ca": 0.5,

    # Number of DQ pins
    "num_dq": 72,

    # Number of DQS pins
    "num_dqs": 18,

    # Number of CA pins
    "num_ca": 25,

    # Number of CLK pins
    "num_clk": 2,

    # Number of Physical Ranks
    "num_mem_dq": 2,

    # Width of the Memory Data Bus
    "mem_data_width": 8,

    # RTT Termination Resistance
    "rtt_value": 10000,

    # RON Termination Resistance
    "ron_value": 34,

    # Time of flight for DQ
    "tflight_value": None,  # Fill this value as needed

    # Parameter related to MemCAD
    # Number of BoBs
    "num_bobs": 1,

    # Memory System Capacity in GB
    "capacity": 80,

    # Number of Channel per BoB
    "num_channels_per_bob": 1,

    # First Metric for ordering different design points
    "first_metric": "Cost",

    # Second Metric for ordering different design points
    "second_metric": "Bandwidth",

    # Third Metric for ordering different design points
    "third_metric": "Energy",

    # Possible DIMM option to consider
    "DIMM_model": "ALL",

    # If channels of each bob have the same configurations
    "mirror_in_bob": "F"

    # Uncomment if you want to see all channels/bobs/memory configurations explored
    # verbose = "F"
  }
