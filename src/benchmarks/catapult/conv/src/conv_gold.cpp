template <typename IDTYPE, 
          typename ODTYPE,
          int OFMAP_HEIGHT, 
          int OFMAP_WIDTH, 
          int OFMAP_CHANNELS, 
          int IFMAP_CHANNELS, 
          int FILTER_SIZE, 
          int STRIDE>
void conv_gold( IDTYPE ifmap[(OFMAP_HEIGHT-1)*STRIDE+FILTER_SIZE][(OFMAP_WIDTH-1)*STRIDE+FILTER_SIZE][IFMAP_CHANNELS],
               IDTYPE weights[FILTER_SIZE][FILTER_SIZE][IFMAP_CHANNELS][OFMAP_CHANNELS],
               ODTYPE ofmap[OFMAP_HEIGHT][OFMAP_WIDTH][OFMAP_CHANNELS]){



  OY: for (int oy = 0; oy < OFMAP_HEIGHT; oy++) {
    OX: for (int ox = 0; ox < OFMAP_WIDTH; ox++) {
      OC: for (int oc = 0; oc < OFMAP_CHANNELS; oc++) {
        int32_t tmp=0;
        IC: for (int ic = 0; ic < IFMAP_CHANNELS; ic++) { 
          FX: for (int fx = 0; fx < FILTER_SIZE; fx++) {
            FY: for (int fy = 0; fy < FILTER_SIZE; fy++) {
              tmp += (int32_t) ifmap[STRIDE*oy+fy][STRIDE*ox+fx][ic] * 
                     (int32_t) weights[fy][fx][ic][oc];
            }
          }
        }
        ofmap[oy][ox][oc]= tmp; 
      }
    }
  }
}
