flow package require MemGen
flow run /MemGen/MemoryGenerator_BuildLib {
VENDOR           Nangate
RTLTOOL          DesignCompiler
TECHNOLOGY       045nm
LIBRARY          ram_sync_1r1w
MODULE           ram_sync_1r1w
OUTPUT_DIR       /nfs/rsghome/pmcewen/dnn-accelerator-hls-master/ram_sync
FILES {
  { FILENAME /nfs/rsghome/pmcewen/dnn-accelerator-hls-master/ram_sync_1r1w.v FILETYPE Verilog MODELTYPE generic PARSE 1 PATHTYPE copy STATICFILE 1 VHDL_LIB_MAPS work }
}
VHDLARRAYPATH    {}
WRITEDELAY       2.5
INITDELAY        1
READDELAY        2.5
VERILOGARRAYPATH {}
TIMEUNIT         1ns
INPUTDELAY       0.01
WIDTH            DATA_WIDTH
AREA             0
WRITELATENCY     1
RDWRRESOLUTION   UNKNOWN
READLATENCY      1
DEPTH            DEPTH
PARAMETERS {
  { PARAMETER DATA_WIDTH TYPE hdl IGNORE 0 MIN {} MAX {} DEFAULT 0 }
  { PARAMETER ADDR_WIDTH TYPE hdl IGNORE 0 MIN {} MAX {} DEFAULT 0 }
  { PARAMETER DEPTH      TYPE hdl IGNORE 0 MIN {} MAX {} DEFAULT 0 }
}
PORTS {
  { NAME port_0 MODE Read  }
  { NAME port_1 MODE Write }
}
PINMAPS {
  { PHYPIN clk   LOGPIN CLOCK        DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {} PORTS {port_0 port_1} }
  { PHYPIN wen   LOGPIN WRITE_ENABLE DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {} PORTS port_1          }
  { PHYPIN wadr  LOGPIN ADDRESS      DIRECTION in  WIDTH ADDR_WIDTH PHASE {} DEFAULT {} PORTS port_1          }
  { PHYPIN wdata LOGPIN DATA_IN      DIRECTION in  WIDTH DATA_WIDTH PHASE {} DEFAULT {} PORTS port_1          }
  { PHYPIN ren   LOGPIN READ_ENABLE  DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {} PORTS port_0          }
  { PHYPIN radr  LOGPIN ADDRESS      DIRECTION in  WIDTH ADDR_WIDTH PHASE {} DEFAULT {} PORTS port_0          }
  { PHYPIN rdata LOGPIN DATA_OUT     DIRECTION out WIDTH DATA_WIDTH PHASE {} DEFAULT {} PORTS port_0          }
}

}
