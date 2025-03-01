solution library add nangate-45nm_beh -- -rtlsyntool DesignCompiler -vendor Nangate -technology 045nm
# solution library add ccs_sample_mem
solution library add custom_ram_sync_1R1W
solution library add custom_ram_sync_dualport
solution library add custom_ram_sync_singleport

# set accum_buffer_module ccs_sample_mem.ccs_ram_sync_1R1W
# set double_buffer_module ccs_sample_mem.ccs_ram_sync_1R1W 
