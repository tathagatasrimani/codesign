flow package require MemGen
flow run /MemGen/MemoryGenerator_BuildLib {
VENDOR           Nangate
RTLTOOL          DesignCompiler
TECHNOLOGY       045nm
LIBRARY          custom_ram_sync_singleport
MODULE           ccs_ram_sync_singleport
OUTPUT_DIR       /nfs/rsghome/pmcewen/codesign/src/tmp/benchmark/ram_sync
FILES {
  { FILENAME /cad/mentor/2019.11/Catapult_Synthesis_10.4b-841621/Mgc_home/pkgs/siflibs/ccs_ram_sync_singleport.v FILETYPE Verilog MODELTYPE generic PARSE 1 PATHTYPE copy STATICFILE 1 VHDL_LIB_MAPS work }
}
VHDLARRAYPATH    {}
WRITEDELAY       2.5
INITDELAY        1
READDELAY        2.5
VERILOGARRAYPATH {}
TIMEUNIT         1ns
INPUTDELAY       0.01
WIDTH            data_width
AREA             0
WRITELATENCY     1
RDWRRESOLUTION   UNKNOWN
READLATENCY      1
DEPTH            depth
PARAMETERS {
  { PARAMETER data_width TYPE hdl IGNORE 0 MIN {} MAX {} DEFAULT 0 }
  { PARAMETER addr_width TYPE hdl IGNORE 0 MIN {} MAX {} DEFAULT 0 }
  { PARAMETER depth      TYPE hdl IGNORE 0 MIN {} MAX {} DEFAULT 0 }
}
PORTS {
  { NAME port_0 MODE ReadWrite  }
}
PINMAPS {
  { PHYPIN clk   LOGPIN CLOCK        DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {} PORTS port_0 }
  { PHYPIN we    LOGPIN WRITE_ENABLE DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {} PORTS port_0 }
  { PHYPIN adr   LOGPIN ADDRESS      DIRECTION in  WIDTH addr_width PHASE {} DEFAULT {} PORTS port_0 }
  { PHYPIN d     LOGPIN DATA_IN      DIRECTION in  WIDTH data_width PHASE {} DEFAULT {} PORTS port_0 }
  { PHYPIN en    LOGPIN READ_ENABLE  DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {} PORTS port_0 }
  { PHYPIN q     LOGPIN DATA_OUT     DIRECTION out WIDTH data_width PHASE {} DEFAULT {} PORTS port_0 }
}

}