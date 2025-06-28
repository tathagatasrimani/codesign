
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

using namespace std;

void forward_node1(
  float v0[5],
  float v1[10],
  int v2
) {	// L51
  #pragma HLS inline
  #pragma HLS resource variable=v0 core=ram_t2p_bram

  for (int v3 = 0; v3 < 5; v3 += 1) {	// L52
    #pragma HLS pipeline II=1
    float v4 = v0[v3];	// L53
    v1[(v3 + (v2 * 5))] = v4;	// L54
  }
}

void forward_node2(
  float v5[512],
  float v6[8][5],
  float v7[10],
  float v8[10],
  float v9[5],
  int v10,
  int v11
) {	// L58
  #pragma HLS inline
  #pragma HLS resource variable=v5 core=ram_t2p_bram

  #pragma HLS resource variable=v6 core=ram_t2p_bram

  #pragma HLS resource variable=v7 core=ram_t2p_bram

  #pragma HLS resource variable=v8 core=ram_t2p_bram

  #pragma HLS resource variable=v9 core=ram_t2p_bram

  for (int v12 = 0; v12 < 8; v12 += 1) {	// L59
    for (int v13 = 0; v13 < 5; v13 += 1) {	// L60
      #pragma HLS pipeline II=1
      float v14 = v5[(v12 + (v10 * 8))];	// L61
      float v15 = v6[v12][v13];	// L62
      float v16 = v8[(v13 + (v11 * 5))];	// L63
      float v17 = v14 * v15;	// L64
      float v18 = v16 + v17;	// L65
      v8[(v13 + (v11 * 5))] = v18;	// L66
      float v19 = v7[(v13 + (v11 * 5))];	// L67
      float v20 = v18 + v19;	// L68
      if ((((-v12) + (v10 * -8)) + 511) == 0) {	// L69
        v9[v13] = v20;	// L70
      }
    }
  }
}

void forward_node3(
  float v21[512][10],
  float v22[8][5],
  int v23,
  int v24
) {	// L76
  #pragma HLS inline
  #pragma HLS resource variable=v22 core=ram_t2p_bram

  for (int v25 = 0; v25 < 8; v25 += 1) {	// L77
    for (int v26 = 0; v26 < 5; v26 += 1) {	// L78
      #pragma HLS pipeline II=1
      float v27 = v21[(v25 + (v23 * 8))][(v26 + (v24 * 5))];	// L79
      v22[v25][v26] = v27;	// L80
    }
  }
}

void forward_node0(
  float v28[10],
  float v29[512],
  float v30[512][10],
  float v31[10]
) {	// L85
  #pragma HLS resource variable=v28 core=ram_t2p_bram

  #pragma HLS resource variable=v29 core=ram_t2p_bram

  float v32[10];	// L86
  #pragma HLS resource variable=v32 core=ram_t2p_bram

  for (int v33 = 0; v33 < 128; v33 += 1) {	// L87
    #pragma HLS dataflow
    int v34 = (v33 % 2);	// L88
    int v35 = (v33 / 2);	// L89
    float v36[5];	// L90
    #pragma HLS resource variable=v36 core=ram_t2p_bram

    float v37[8][5];	// L91
    #pragma HLS resource variable=v37 core=ram_t2p_bram

    forward_node3(v30, v37, v35, v34);	// L92
    forward_node2(v29, v37, v28, v32, v36, v35, v34);	// L93
    forward_node1(v36, v31, v34);	// L94
  }
}

void forward_node5(
  float v38[8][2][2],
  float v39[512],
  int v40,
  int v41,
  int v42
) {	// L98
  #pragma HLS inline
  #pragma HLS resource variable=v38 core=ram_t2p_bram

  #pragma HLS resource variable=v39 core=ram_t2p_bram

  for (int v43 = 0; v43 < 2; v43 += 1) {	// L100
    for (int v44 = 0; v44 < 2; v44 += 1) {	// L101
      for (int v45 = 0; v45 < 8; v45 += 1) {	// L102
        #pragma HLS pipeline II=1
        float v46 = v38[v45][v43][v44];	// L103
        float v47 = v39[(v45 + (v40 * 8))];	// L104
        float v48 = v47 + v46;	// L105
        float v49 = v48 / (float)16.000000;	// L106
        float v50 = ((((-v43) + (v41 * -2)) + 3) == 0 && (((-v44) + (v42 * -2)) + 3) == 0) ? v49 : v48;	// L107
        v39[(v45 + (v40 * 8))] = v50;	// L108
      }
    }
  }
}

void forward_node6(
  float v51[512][4][4],
  float v52[8][2][2],
  int v53,
  int v54,
  int v55
) {	// L114
  #pragma HLS inline
  #pragma HLS resource variable=v52 core=ram_t2p_bram

  for (int v56 = 0; v56 < 8; v56 += 1) {	// L115
    for (int v57 = 0; v57 < 2; v57 += 1) {	// L116
      for (int v58 = 0; v58 < 2; v58 += 1) {	// L117
        #pragma HLS pipeline II=1
        float v59 = v51[(v56 + (v53 * 8))][(v57 + (v54 * 2))][(v58 + (v55 * 2))];	// L118
        v52[v56][v57][v58] = v59;	// L119
      }
    }
  }
}

void forward_node4(
  float v60[512][4][4],
  float v61[512]
) {	// L125
  #pragma HLS resource variable=v61 core=ram_t2p_bram

  for (int v62 = 0; v62 < 256; v62 += 1) {	// L126
    #pragma HLS dataflow
    int v63 = (v62 % 64);	// L127
    int v64 = ((v62 / 64) % 2);	// L128
    int v65 = ((v62 / 64) / 2);	// L129
    float v66[8][2][2];	// L130
    #pragma HLS resource variable=v66 core=ram_t2p_bram

    forward_node6(v60, v66, v63, v65, v64);	// L131
    forward_node5(v66, v61, v63, v65, v64);	// L132
  }
}

void forward_node8(
  float v67[8][2][2],
  float v68[512][4][4],
  int v69,
  int v70,
  int v71
) {	// L136
  #pragma HLS inline
  #pragma HLS array_partition variable=v67 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v67 cyclic factor=2 dim=3
  #pragma HLS resource variable=v67 core=ram_t2p_bram

  #pragma HLS array_partition variable=v68 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v68 cyclic factor=2 dim=3

  for (int v72 = 0; v72 < 8; v72 += 1) {	// L137
    #pragma HLS pipeline II=1
    float v73 = v67[v72][0][0];	// L138
    v68[(v72 + (v69 * 8))][(v70 * 2)][(v71 * 2)] = v73;	// L139
    float v74 = v67[v72][0][1];	// L140
    v68[(v72 + (v69 * 8))][(v70 * 2)][((v71 * 2) + 1)] = v74;	// L141
    float v75 = v67[v72][1][0];	// L142
    v68[(v72 + (v69 * 8))][((v70 * 2) + 1)][(v71 * 2)] = v75;	// L143
    float v76 = v67[v72][1][1];	// L144
    v68[(v72 + (v69 * 8))][((v70 * 2) + 1)][((v71 * 2) + 1)] = v76;	// L145
  }
}

void forward_node9(
  float v77[8][2][2],
  float v78[512][4][4],
  int v79,
  int v80,
  int v81
) {	// L149
  #pragma HLS inline
  #pragma HLS array_partition variable=v77 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v77 cyclic factor=2 dim=3
  #pragma HLS resource variable=v77 core=ram_t2p_bram

  #pragma HLS array_partition variable=v78 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v78 cyclic factor=2 dim=3

  for (int v82 = 0; v82 < 8; v82 += 1) {	// L150
    #pragma HLS pipeline II=1
    float v83 = v77[v82][0][0];	// L151
    v78[(v82 + (v79 * 8))][(v80 * 2)][(v81 * 2)] = v83;	// L152
    float v84 = v77[v82][0][1];	// L153
    v78[(v82 + (v79 * 8))][(v80 * 2)][((v81 * 2) + 1)] = v84;	// L154
    float v85 = v77[v82][1][0];	// L155
    v78[(v82 + (v79 * 8))][((v80 * 2) + 1)][(v81 * 2)] = v85;	// L156
    float v86 = v77[v82][1][1];	// L157
    v78[(v82 + (v79 * 8))][((v80 * 2) + 1)][((v81 * 2) + 1)] = v86;	// L158
  }
}

void forward_node10(
  float v87[8][2][2],
  float v88[8][8],
  float v89[8][2][2],
  float v90[8][2][2],
  float v91[8][2][2],
  float v92[8][2][2],
  int v93,
  int v94,
  int v95
) {	// L162
  #pragma HLS inline
  #pragma HLS array_partition variable=v87 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v87 cyclic factor=2 dim=3
  #pragma HLS resource variable=v87 core=ram_t2p_bram

  #pragma HLS resource variable=v88 core=ram_t2p_bram

  #pragma HLS array_partition variable=v89 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v89 cyclic factor=2 dim=3
  #pragma HLS resource variable=v89 core=ram_t2p_bram

  #pragma HLS array_partition variable=v90 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v90 cyclic factor=2 dim=3
  #pragma HLS resource variable=v90 core=ram_t2p_bram

  #pragma HLS array_partition variable=v91 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v91 cyclic factor=2 dim=3
  #pragma HLS resource variable=v91 core=ram_t2p_bram

  #pragma HLS array_partition variable=v92 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v92 cyclic factor=2 dim=3
  #pragma HLS resource variable=v92 core=ram_t2p_bram

  for (int v96 = 0; v96 < 8; v96 += 1) {	// L164
    for (int v97 = 0; v97 < 8; v97 += 1) {	// L165
      #pragma HLS pipeline II=1
      float v98 = v87[v96][0][0];	// L166
      float v99 = v88[v97][v96];	// L167
      float v100 = v90[v97][0][0];	// L168
      float v101 = v92[v97][0][0];	// L169
      float v102 = (v96 == 0) ? v100 : v101;	// L170
      float v103 = v98 * v99;	// L171
      float v104 = v102 + v103;	// L172
      v92[v97][0][0] = v104;	// L173
      float v105 = v89[v97][0][0];	// L174
      float v106 = v104 + v105;	// L175
      bool v107 = v106 > (float)0.000000;	// L176
      float v108 = v107 ? v106 : (float)0.000000;	// L177
      if ((((-v96) + (v94 * -8)) + 511) == 0 && ((-v93) + 2) == 0 && ((-v95) + 2) == 0) {	// L178
        v91[v97][0][0] = v108;	// L179
      }
      float v109 = v87[v96][0][1];	// L181
      float v110 = v90[v97][0][1];	// L182
      float v111 = v92[v97][0][1];	// L183
      float v112 = (v96 == 0) ? v110 : v111;	// L184
      float v113 = v109 * v99;	// L185
      float v114 = v112 + v113;	// L186
      v92[v97][0][1] = v114;	// L187
      float v115 = v89[v97][0][1];	// L188
      float v116 = v114 + v115;	// L189
      bool v117 = v116 > (float)0.000000;	// L190
      float v118 = v117 ? v116 : (float)0.000000;	// L191
      if ((((-v96) + (v94 * -8)) + 511) == 0 && ((-v93) + 2) == 0 && ((-v95) + 2) == 0) {	// L192
        v91[v97][0][1] = v118;	// L193
      }
      float v119 = v87[v96][1][0];	// L195
      float v120 = v90[v97][1][0];	// L196
      float v121 = v92[v97][1][0];	// L197
      float v122 = (v96 == 0) ? v120 : v121;	// L198
      float v123 = v119 * v99;	// L199
      float v124 = v122 + v123;	// L200
      v92[v97][1][0] = v124;	// L201
      float v125 = v89[v97][1][0];	// L202
      float v126 = v124 + v125;	// L203
      bool v127 = v126 > (float)0.000000;	// L204
      float v128 = v127 ? v126 : (float)0.000000;	// L205
      if ((((-v96) + (v94 * -8)) + 511) == 0 && ((-v93) + 2) == 0 && ((-v95) + 2) == 0) {	// L206
        v91[v97][1][0] = v128;	// L207
      }
      float v129 = v87[v96][1][1];	// L209
      float v130 = v90[v97][1][1];	// L210
      float v131 = v92[v97][1][1];	// L211
      float v132 = (v96 == 0) ? v130 : v131;	// L212
      float v133 = v129 * v99;	// L213
      float v134 = v132 + v133;	// L214
      v92[v97][1][1] = v134;	// L215
      float v135 = v89[v97][1][1];	// L216
      float v136 = v134 + v135;	// L217
      bool v137 = v136 > (float)0.000000;	// L218
      float v138 = v137 ? v136 : (float)0.000000;	// L219
      if ((((-v96) + (v94 * -8)) + 511) == 0 && ((-v93) + 2) == 0 && ((-v95) + 2) == 0) {	// L220
        v91[v97][1][1] = v138;	// L221
      }
    }
  }
}

void forward_node11(
  float v139[512][4][4],
  float v140[8][2][2],
  int v141,
  int v142,
  int v143
) {	// L227
  #pragma HLS inline
  #pragma HLS array_partition variable=v139 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v139 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v140 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v140 cyclic factor=2 dim=3
  #pragma HLS resource variable=v140 core=ram_t2p_bram

  for (int v144 = 0; v144 < 8; v144 += 1) {	// L228
    #pragma HLS pipeline II=1
    float v145 = v139[(v144 + (v141 * 8))][(v142 * 2)][(v143 * 2)];	// L229
    v140[v144][0][0] = v145;	// L230
    float v146 = v139[(v144 + (v141 * 8))][(v142 * 2)][((v143 * 2) + 1)];	// L231
    v140[v144][0][1] = v146;	// L232
    float v147 = v139[(v144 + (v141 * 8))][((v142 * 2) + 1)][(v143 * 2)];	// L233
    v140[v144][1][0] = v147;	// L234
    float v148 = v139[(v144 + (v141 * 8))][((v142 * 2) + 1)][((v143 * 2) + 1)];	// L235
    v140[v144][1][1] = v148;	// L236
  }
}

void forward_node12(
  float v149[512][4][4],
  float v150[8][2][2],
  int v151,
  int v152,
  int v153
) {	// L240
  #pragma HLS inline
  #pragma HLS array_partition variable=v149 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v149 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v150 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v150 cyclic factor=2 dim=3
  #pragma HLS resource variable=v150 core=ram_t2p_bram

  for (int v154 = 0; v154 < 8; v154 += 1) {	// L241
    #pragma HLS pipeline II=1
    float v155 = v149[(v154 + (v151 * 8))][(v152 * 2)][(v153 * 2)];	// L242
    v150[v154][0][0] = v155;	// L243
    float v156 = v149[(v154 + (v151 * 8))][(v152 * 2)][((v153 * 2) + 1)];	// L244
    v150[v154][0][1] = v156;	// L245
    float v157 = v149[(v154 + (v151 * 8))][((v152 * 2) + 1)][(v153 * 2)];	// L246
    v150[v154][1][0] = v157;	// L247
    float v158 = v149[(v154 + (v151 * 8))][((v152 * 2) + 1)][((v153 * 2) + 1)];	// L248
    v150[v154][1][1] = v158;	// L249
  }
}

void forward_node13(
  float v159[512][512][3][3],
  float v160[8][8],
  int v161,
  int v162,
  int v163,
  int v164
) {	// L253
  #pragma HLS inline
  #pragma HLS resource variable=v160 core=ram_t2p_bram

  for (int v165 = 0; v165 < 8; v165 += 1) {	// L254
    for (int v166 = 0; v166 < 8; v166 += 1) {	// L255
      #pragma HLS pipeline II=1
      float v167 = v159[(v165 + (v163 * 8))][(v166 + (v164 * 8))][v161][v162];	// L256
      v160[v165][v166] = v167;	// L257
    }
  }
}

void forward_node14(
  float v168[512][4][4],
  float v169[8][2][2],
  int v170,
  int v171,
  int v172,
  int v173,
  int v174
) {	// L262
  #pragma HLS inline
  #pragma HLS array_partition variable=v168 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v168 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v169 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v169 cyclic factor=2 dim=3
  #pragma HLS resource variable=v169 core=ram_t2p_bram

  for (int v175 = 0; v175 < 8; v175 += 1) {	// L263
    #pragma HLS pipeline II=1
    float v176 = v168[(v175 + (v170 * 8))][((v171 + (v172 * 2)) - 1)][((v173 + (v174 * 2)) - 1)];	// L264
    v169[v175][0][0] = v176;	// L265
    float v177 = v168[(v175 + (v170 * 8))][((v171 + (v172 * 2)) - 1)][(v173 + (v174 * 2))];	// L266
    v169[v175][0][1] = v177;	// L267
    float v178 = v168[(v175 + (v170 * 8))][(v171 + (v172 * 2))][((v173 + (v174 * 2)) - 1)];	// L268
    v169[v175][1][0] = v178;	// L269
    float v179 = v168[(v175 + (v170 * 8))][(v171 + (v172 * 2))][(v173 + (v174 * 2))];	// L270
    v169[v175][1][1] = v179;	// L271
  }
}

void forward_node7(
  float v180[512][512][3][3],
  float v181[512][4][4],
  float v182[512][4][4],
  float v183[512][4][4],
  float v184[512][4][4],
  float v185[512][4][4]
) {	// L275
  #pragma HLS array_partition variable=v181 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v181 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v182 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v182 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v183 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v183 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v184 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v184 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v185 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v185 cyclic factor=2 dim=3

  for (int v186 = 0; v186 < 147456; v186 += 1) {	// L276
    #pragma HLS dataflow
    int v187 = (v186 % 2);	// L277
    int v188 = ((v186 / 2) % 2);	// L278
    int v189 = (((v186 / 2) / 2) % 64);	// L279
    int v190 = ((((v186 / 2) / 2) / 64) % 3);	// L280
    int v191 = (((((v186 / 2) / 2) / 64) / 3) % 3);	// L281
    int v192 = (((((v186 / 2) / 2) / 64) / 3) / 3);	// L282
    float v193[8][2][2];	// L283
    #pragma HLS array_partition variable=v193 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v193 cyclic factor=2 dim=3
    #pragma HLS resource variable=v193 core=ram_t2p_bram

    float v194[8][2][2];	// L284
    #pragma HLS array_partition variable=v194 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v194 cyclic factor=2 dim=3
    #pragma HLS resource variable=v194 core=ram_t2p_bram

    float v195[8][2][2];	// L285
    #pragma HLS array_partition variable=v195 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v195 cyclic factor=2 dim=3
    #pragma HLS resource variable=v195 core=ram_t2p_bram

    float v196[8][8];	// L286
    #pragma HLS resource variable=v196 core=ram_t2p_bram

    float v197[8][2][2];	// L287
    #pragma HLS array_partition variable=v197 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v197 cyclic factor=2 dim=3
    #pragma HLS resource variable=v197 core=ram_t2p_bram

    forward_node14(v182, v197, v192, v191, v188, v190, v187);	// L288
    forward_node13(v180, v196, v191, v190, v189, v192);	// L289
    forward_node12(v183, v195, v189, v188, v187);	// L290
    forward_node11(v181, v194, v189, v188, v187);	// L291
    float v198[8][2][2];	// L292
    #pragma HLS array_partition variable=v198 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v198 cyclic factor=2 dim=3
    #pragma HLS resource variable=v198 core=ram_t2p_bram

    forward_node10(v197, v196, v194, v195, v193, v198, v191, v192, v190);	// L293
    forward_node9(v198, v185, v189, v188, v187);	// L294
    forward_node8(v193, v184, v189, v188, v187);	// L295
  }
}

void forward_node16(
  float v199[8][2][2],
  float v200[512][4][4],
  int v201,
  int v202,
  int v203
) {	// L299
  #pragma HLS inline
  #pragma HLS array_partition variable=v199 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v199 cyclic factor=2 dim=3
  #pragma HLS resource variable=v199 core=ram_t2p_bram

  #pragma HLS array_partition variable=v200 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v200 cyclic factor=2 dim=3

  for (int v204 = 0; v204 < 8; v204 += 1) {	// L300
    #pragma HLS pipeline II=1
    float v205 = v199[v204][0][0];	// L301
    v200[(v204 + (v201 * 8))][(v202 * 2)][(v203 * 2)] = v205;	// L302
    float v206 = v199[v204][0][1];	// L303
    v200[(v204 + (v201 * 8))][(v202 * 2)][((v203 * 2) + 1)] = v206;	// L304
    float v207 = v199[v204][1][0];	// L305
    v200[(v204 + (v201 * 8))][((v202 * 2) + 1)][(v203 * 2)] = v207;	// L306
    float v208 = v199[v204][1][1];	// L307
    v200[(v204 + (v201 * 8))][((v202 * 2) + 1)][((v203 * 2) + 1)] = v208;	// L308
  }
}

void forward_node17(
  float v209[8][2][2],
  float v210[8][8],
  float v211[8][2][2],
  float v212[8][2][2],
  float v213[8][2][2],
  int v214,
  int v215,
  int v216
) {	// L312
  #pragma HLS inline
  #pragma HLS array_partition variable=v209 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v209 cyclic factor=2 dim=3
  #pragma HLS resource variable=v209 core=ram_t2p_bram

  #pragma HLS resource variable=v210 core=ram_t2p_bram

  #pragma HLS array_partition variable=v211 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v211 cyclic factor=2 dim=3
  #pragma HLS resource variable=v211 core=ram_t2p_bram

  #pragma HLS array_partition variable=v212 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v212 cyclic factor=2 dim=3
  #pragma HLS resource variable=v212 core=ram_t2p_bram

  #pragma HLS array_partition variable=v213 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v213 cyclic factor=2 dim=3
  #pragma HLS resource variable=v213 core=ram_t2p_bram

  for (int v217 = 0; v217 < 8; v217 += 1) {	// L314
    for (int v218 = 0; v218 < 8; v218 += 1) {	// L315
      #pragma HLS pipeline II=1
      float v219 = v211[v218][0][0];	// L316
      float v220 = v212[v218][0][0];	// L317
      float v221 = v213[v218][0][0];	// L318
      float v222 = (v217 == 0) ? v220 : v221;	// L319
      float v223 = ((v217 + (v214 * 8)) == 0 && v215 == 0 && v216 == 0) ? v219 : v222;	// L320
      float v224 = v209[v217][0][0];	// L321
      float v225 = v210[v218][v217];	// L322
      float v226 = v224 * v225;	// L323
      float v227 = v223 + v226;	// L324
      bool v228 = v227 > (float)0.000000;	// L325
      float v229 = v228 ? v227 : (float)0.000000;	// L326
      float v230 = ((((-v217) + (v214 * -8)) + 511) == 0 && ((-v215) + 2) == 0 && ((-v216) + 2) == 0) ? v229 : v227;	// L327
      v213[v218][0][0] = v230;	// L328
      float v231 = v211[v218][0][1];	// L329
      float v232 = v212[v218][0][1];	// L330
      float v233 = v213[v218][0][1];	// L331
      float v234 = (v217 == 0) ? v232 : v233;	// L332
      float v235 = ((v217 + (v214 * 8)) == 0 && v215 == 0 && v216 == 0) ? v231 : v234;	// L333
      float v236 = v209[v217][0][1];	// L334
      float v237 = v236 * v225;	// L335
      float v238 = v235 + v237;	// L336
      bool v239 = v238 > (float)0.000000;	// L337
      float v240 = v239 ? v238 : (float)0.000000;	// L338
      float v241 = ((((-v217) + (v214 * -8)) + 511) == 0 && ((-v215) + 2) == 0 && ((-v216) + 2) == 0) ? v240 : v238;	// L339
      v213[v218][0][1] = v241;	// L340
      float v242 = v211[v218][1][0];	// L341
      float v243 = v212[v218][1][0];	// L342
      float v244 = v213[v218][1][0];	// L343
      float v245 = (v217 == 0) ? v243 : v244;	// L344
      float v246 = ((v217 + (v214 * 8)) == 0 && v215 == 0 && v216 == 0) ? v242 : v245;	// L345
      float v247 = v209[v217][1][0];	// L346
      float v248 = v247 * v225;	// L347
      float v249 = v246 + v248;	// L348
      bool v250 = v249 > (float)0.000000;	// L349
      float v251 = v250 ? v249 : (float)0.000000;	// L350
      float v252 = ((((-v217) + (v214 * -8)) + 511) == 0 && ((-v215) + 2) == 0 && ((-v216) + 2) == 0) ? v251 : v249;	// L351
      v213[v218][1][0] = v252;	// L352
      float v253 = v211[v218][1][1];	// L353
      float v254 = v212[v218][1][1];	// L354
      float v255 = v213[v218][1][1];	// L355
      float v256 = (v217 == 0) ? v254 : v255;	// L356
      float v257 = ((v217 + (v214 * 8)) == 0 && v215 == 0 && v216 == 0) ? v253 : v256;	// L357
      float v258 = v209[v217][1][1];	// L358
      float v259 = v258 * v225;	// L359
      float v260 = v257 + v259;	// L360
      bool v261 = v260 > (float)0.000000;	// L361
      float v262 = v261 ? v260 : (float)0.000000;	// L362
      float v263 = ((((-v217) + (v214 * -8)) + 511) == 0 && ((-v215) + 2) == 0 && ((-v216) + 2) == 0) ? v262 : v260;	// L363
      v213[v218][1][1] = v263;	// L364
    }
  }
}

void forward_node18(
  float v264[512][512][3][3],
  float v265[8][8],
  int v266,
  int v267,
  int v268,
  int v269
) {	// L369
  #pragma HLS inline
  #pragma HLS resource variable=v265 core=ram_t2p_bram

  for (int v270 = 0; v270 < 8; v270 += 1) {	// L370
    for (int v271 = 0; v271 < 8; v271 += 1) {	// L371
      #pragma HLS pipeline II=1
      float v272 = v264[(v270 + (v268 * 8))][(v271 + (v269 * 8))][v266][v267];	// L372
      v265[v270][v271] = v272;	// L373
    }
  }
}

void forward_node19(
  float v273[512][4][4],
  float v274[8][2][2],
  int v275,
  int v276,
  int v277,
  int v278,
  int v279
) {	// L378
  #pragma HLS inline
  #pragma HLS array_partition variable=v273 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v273 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v274 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v274 cyclic factor=2 dim=3
  #pragma HLS resource variable=v274 core=ram_t2p_bram

  for (int v280 = 0; v280 < 8; v280 += 1) {	// L379
    #pragma HLS pipeline II=1
    float v281 = v273[(v280 + (v275 * 8))][((v276 + (v277 * 2)) - 1)][((v278 + (v279 * 2)) - 1)];	// L380
    v274[v280][0][0] = v281;	// L381
    float v282 = v273[(v280 + (v275 * 8))][((v276 + (v277 * 2)) - 1)][(v278 + (v279 * 2))];	// L382
    v274[v280][0][1] = v282;	// L383
    float v283 = v273[(v280 + (v275 * 8))][(v276 + (v277 * 2))][((v278 + (v279 * 2)) - 1)];	// L384
    v274[v280][1][0] = v283;	// L385
    float v284 = v273[(v280 + (v275 * 8))][(v276 + (v277 * 2))][(v278 + (v279 * 2))];	// L386
    v274[v280][1][1] = v284;	// L387
  }
}

void forward_node20(
  float v285[512][4][4],
  float v286[8][2][2],
  int v287,
  int v288,
  int v289
) {	// L391
  #pragma HLS inline
  #pragma HLS array_partition variable=v285 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v285 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v286 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v286 cyclic factor=2 dim=3
  #pragma HLS resource variable=v286 core=ram_t2p_bram

  for (int v290 = 0; v290 < 8; v290 += 1) {	// L392
    #pragma HLS pipeline II=1
    float v291 = v285[(v290 + (v287 * 8))][(v288 * 2)][(v289 * 2)];	// L393
    v286[v290][0][0] = v291;	// L394
    float v292 = v285[(v290 + (v287 * 8))][(v288 * 2)][((v289 * 2) + 1)];	// L395
    v286[v290][0][1] = v292;	// L396
    float v293 = v285[(v290 + (v287 * 8))][((v288 * 2) + 1)][(v289 * 2)];	// L397
    v286[v290][1][0] = v293;	// L398
    float v294 = v285[(v290 + (v287 * 8))][((v288 * 2) + 1)][((v289 * 2) + 1)];	// L399
    v286[v290][1][1] = v294;	// L400
  }
}

void forward_node21(
  float v295[512][4][4],
  float v296[8][2][2],
  int v297,
  int v298,
  int v299
) {	// L404
  #pragma HLS inline
  #pragma HLS array_partition variable=v295 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v295 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v296 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v296 cyclic factor=2 dim=3
  #pragma HLS resource variable=v296 core=ram_t2p_bram

  for (int v300 = 0; v300 < 8; v300 += 1) {	// L405
    #pragma HLS pipeline II=1
    float v301 = v295[(v300 + (v297 * 8))][(v298 * 2)][(v299 * 2)];	// L406
    v296[v300][0][0] = v301;	// L407
    float v302 = v295[(v300 + (v297 * 8))][(v298 * 2)][((v299 * 2) + 1)];	// L408
    v296[v300][0][1] = v302;	// L409
    float v303 = v295[(v300 + (v297 * 8))][((v298 * 2) + 1)][(v299 * 2)];	// L410
    v296[v300][1][0] = v303;	// L411
    float v304 = v295[(v300 + (v297 * 8))][((v298 * 2) + 1)][((v299 * 2) + 1)];	// L412
    v296[v300][1][1] = v304;	// L413
  }
}

void forward_node15(
  float v305[512][4][4],
  float v306[512][4][4],
  float v307[512][512][3][3],
  float v308[512][4][4],
  float v309[512][4][4]
) {	// L417
  #pragma HLS array_partition variable=v305 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v305 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v306 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v306 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v308 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v308 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v309 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v309 cyclic factor=2 dim=3

  for (int v310 = 0; v310 < 147456; v310 += 1) {	// L418
    #pragma HLS dataflow
    int v311 = (v310 % 2);	// L419
    int v312 = ((v310 / 2) % 2);	// L420
    int v313 = (((v310 / 2) / 2) % 64);	// L421
    int v314 = ((((v310 / 2) / 2) / 64) % 3);	// L422
    int v315 = (((((v310 / 2) / 2) / 64) / 3) % 3);	// L423
    int v316 = (((((v310 / 2) / 2) / 64) / 3) / 3);	// L424
    float v317[8][8];	// L425
    #pragma HLS resource variable=v317 core=ram_t2p_bram

    float v318[8][2][2];	// L426
    #pragma HLS array_partition variable=v318 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v318 cyclic factor=2 dim=3
    #pragma HLS resource variable=v318 core=ram_t2p_bram

    float v319[8][2][2];	// L427
    #pragma HLS array_partition variable=v319 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v319 cyclic factor=2 dim=3
    #pragma HLS resource variable=v319 core=ram_t2p_bram

    float v320[8][2][2];	// L428
    #pragma HLS array_partition variable=v320 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v320 cyclic factor=2 dim=3
    #pragma HLS resource variable=v320 core=ram_t2p_bram

    forward_node21(v306, v320, v313, v312, v311);	// L429
    forward_node20(v308, v319, v313, v312, v311);	// L430
    forward_node19(v305, v318, v316, v315, v312, v314, v311);	// L431
    forward_node18(v307, v317, v315, v314, v313, v316);	// L432
    float v321[8][2][2];	// L433
    #pragma HLS array_partition variable=v321 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v321 cyclic factor=2 dim=3
    #pragma HLS resource variable=v321 core=ram_t2p_bram

    forward_node17(v318, v317, v320, v319, v321, v316, v315, v314);	// L434
    forward_node16(v321, v309, v313, v312, v311);	// L435
  }
}

void forward_node23(
  float v322[8][2][2],
  float v323[512][4][4],
  int v324,
  int v325,
  int v326
) {	// L439
  #pragma HLS inline
  #pragma HLS resource variable=v322 core=ram_t2p_bram

  for (int v327 = 0; v327 < 8; v327 += 1) {	// L440
    for (int v328 = 0; v328 < 2; v328 += 1) {	// L441
      for (int v329 = 0; v329 < 2; v329 += 1) {	// L442
        #pragma HLS pipeline II=1
        float v330 = v322[v327][v328][v329];	// L443
        v323[(v327 + (v324 * 8))][(v328 + (v325 * 2))][(v329 + (v326 * 2))] = v330;	// L444
      }
    }
  }
}

void forward_node24(
  float v331[8][2][2],
  float v332[512][4][4],
  int v333,
  int v334,
  int v335
) {	// L450
  #pragma HLS inline
  #pragma HLS resource variable=v331 core=ram_t2p_bram

  for (int v336 = 0; v336 < 8; v336 += 1) {	// L451
    for (int v337 = 0; v337 < 2; v337 += 1) {	// L452
      for (int v338 = 0; v338 < 2; v338 += 1) {	// L453
        #pragma HLS pipeline II=1
        float v339 = v331[v336][v337][v338];	// L454
        v332[(v336 + (v333 * 8))][(v337 + (v334 * 2))][(v338 + (v335 * 2))] = v339;	// L455
      }
    }
  }
}

void forward_node25(
  float v340[8][2][2],
  float v341[8][2][2],
  float v342[8][8],
  float v343[8][2][2],
  float v344[8][2][2],
  float v345[8][2][2],
  float v346[8][2][2],
  int v347
) {	// L461
  #pragma HLS inline
  #pragma HLS resource variable=v340 core=ram_t2p_bram

  #pragma HLS resource variable=v341 core=ram_t2p_bram

  #pragma HLS resource variable=v342 core=ram_t2p_bram

  #pragma HLS resource variable=v343 core=ram_t2p_bram

  #pragma HLS resource variable=v344 core=ram_t2p_bram

  #pragma HLS resource variable=v345 core=ram_t2p_bram

  #pragma HLS resource variable=v346 core=ram_t2p_bram

  for (int v348 = 0; v348 < 8; v348 += 1) {	// L463
    for (int v349 = 0; v349 < 8; v349 += 1) {	// L464
      for (int v350 = 0; v350 < 2; v350 += 1) {	// L465
        for (int v351 = 0; v351 < 2; v351 += 1) {	// L466
          #pragma HLS pipeline II=1
          float v352 = v340[v349][v350][v351];	// L467
          float v353 = v344[v349][v350][v351];	// L468
          float v354 = v345[v349][v350][v351];	// L469
          float v355 = (v348 == 0) ? v353 : v354;	// L470
          float v356 = ((v348 + (v347 * 8)) == 0) ? v352 : v355;	// L471
          float v357 = v341[v348][v350][v351];	// L472
          float v358 = v342[v349][v348];	// L473
          float v359 = v357 * v358;	// L474
          float v360 = v356 + v359;	// L475
          v345[v349][v350][v351] = v360;	// L476
          float v361 = v343[v349][v350][v351];	// L477
          float v362 = v361 + v360;	// L478
          bool v363 = v362 > (float)0.000000;	// L479
          float v364 = v363 ? v362 : (float)0.000000;	// L480
          if ((((-v348) + (v347 * -8)) + 255) == 0) {	// L481
            v346[v349][v350][v351] = v364;	// L482
          }
        }
      }
    }
  }
}

void forward_node26(
  float v365[512][4][4],
  float v366[8][2][2],
  int v367,
  int v368,
  int v369
) {	// L490
  #pragma HLS inline
  #pragma HLS resource variable=v366 core=ram_t2p_bram

  for (int v370 = 0; v370 < 8; v370 += 1) {	// L491
    for (int v371 = 0; v371 < 2; v371 += 1) {	// L492
      for (int v372 = 0; v372 < 2; v372 += 1) {	// L493
        #pragma HLS pipeline II=1
        float v373 = v365[(v370 + (v367 * 8))][(v371 + (v368 * 2))][(v372 + (v369 * 2))];	// L494
        v366[v370][v371][v372] = v373;	// L495
      }
    }
  }
}

void forward_node27(
  float v374[512][256],
  float v375[8][8],
  int v376,
  int v377
) {	// L501
  #pragma HLS inline
  #pragma HLS resource variable=v375 core=ram_t2p_bram

  for (int v378 = 0; v378 < 8; v378 += 1) {	// L502
    for (int v379 = 0; v379 < 8; v379 += 1) {	// L503
      #pragma HLS pipeline II=1
      float v380 = v374[(v378 + (v376 * 8))][(v379 + (v377 * 8))];	// L504
      v375[v378][v379] = v380;	// L505
    }
  }
}

void forward_node28(
  float v381[256][8][8],
  float v382[8][2][2],
  int v383,
  int v384,
  int v385
) {	// L510
  #pragma HLS inline
  #pragma HLS resource variable=v382 core=ram_t2p_bram

  for (int v386 = 0; v386 < 8; v386 += 1) {	// L511
    for (int v387 = 0; v387 < 2; v387 += 1) {	// L512
      for (int v388 = 0; v388 < 2; v388 += 1) {	// L513
        #pragma HLS pipeline II=1
        float v389 = v381[(v386 + (v383 * 8))][((v387 * 2) + (v384 * 4))][((v388 * 2) + (v385 * 4))];	// L514
        v382[v386][v387][v388] = v389;	// L515
      }
    }
  }
}

void forward_node29(
  float v390[512][4][4],
  float v391[8][2][2],
  int v392,
  int v393,
  int v394
) {	// L521
  #pragma HLS inline
  #pragma HLS resource variable=v391 core=ram_t2p_bram

  for (int v395 = 0; v395 < 8; v395 += 1) {	// L522
    for (int v396 = 0; v396 < 2; v396 += 1) {	// L523
      for (int v397 = 0; v397 < 2; v397 += 1) {	// L524
        #pragma HLS pipeline II=1
        float v398 = v390[(v395 + (v392 * 8))][(v396 + (v393 * 2))][(v397 + (v394 * 2))];	// L525
        v391[v395][v396][v397] = v398;	// L526
      }
    }
  }
}

void forward_node30(
  float v399[512][4][4],
  float v400[8][2][2],
  int v401,
  int v402,
  int v403
) {	// L532
  #pragma HLS inline
  #pragma HLS resource variable=v400 core=ram_t2p_bram

  for (int v404 = 0; v404 < 8; v404 += 1) {	// L533
    for (int v405 = 0; v405 < 2; v405 += 1) {	// L534
      for (int v406 = 0; v406 < 2; v406 += 1) {	// L535
        #pragma HLS pipeline II=1
        float v407 = v399[(v404 + (v401 * 8))][(v405 + (v402 * 2))][(v406 + (v403 * 2))];	// L536
        v400[v404][v405][v406] = v407;	// L537
      }
    }
  }
}

void forward_node22(
  float v408[256][8][8],
  float v409[512][4][4],
  float v410[512][4][4],
  float v411[512][256],
  float v412[512][4][4],
  float v413[512][4][4],
  float v414[512][4][4]
) {	// L543
  for (int v415 = 0; v415 < 8192; v415 += 1) {	// L544
    #pragma HLS dataflow
    int v416 = (v415 % 2);	// L545
    int v417 = ((v415 / 2) % 2);	// L546
    int v418 = (((v415 / 2) / 2) % 64);	// L547
    int v419 = (((v415 / 2) / 2) / 64);	// L548
    float v420[8][2][2];	// L549
    #pragma HLS resource variable=v420 core=ram_t2p_bram

    float v421[8][2][2];	// L550
    #pragma HLS resource variable=v421 core=ram_t2p_bram

    float v422[8][8];	// L551
    #pragma HLS resource variable=v422 core=ram_t2p_bram

    float v423[8][2][2];	// L552
    #pragma HLS resource variable=v423 core=ram_t2p_bram

    float v424[8][2][2];	// L553
    #pragma HLS resource variable=v424 core=ram_t2p_bram

    float v425[8][2][2];	// L554
    #pragma HLS resource variable=v425 core=ram_t2p_bram

    forward_node30(v409, v425, v418, v417, v416);	// L555
    forward_node29(v412, v424, v418, v417, v416);	// L556
    forward_node28(v408, v423, v419, v417, v416);	// L557
    forward_node27(v411, v422, v418, v419);	// L558
    forward_node26(v410, v421, v418, v417, v416);	// L559
    float v426[8][2][2];	// L560
    #pragma HLS resource variable=v426 core=ram_t2p_bram

    forward_node25(v425, v423, v422, v421, v424, v426, v420, v419);	// L561
    forward_node24(v426, v414, v418, v417, v416);	// L562
    forward_node23(v420, v413, v418, v417, v416);	// L563
  }
}

void forward_node32(
  float v427[8][2][2],
  float v428[512][4][4],
  int v429,
  int v430,
  int v431
) {	// L567
  #pragma HLS inline
  #pragma HLS array_partition variable=v427 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v427 cyclic factor=2 dim=3
  #pragma HLS resource variable=v427 core=ram_t2p_bram

  #pragma HLS array_partition variable=v428 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v428 cyclic factor=2 dim=3

  for (int v432 = 0; v432 < 8; v432 += 1) {	// L568
    #pragma HLS pipeline II=1
    float v433 = v427[v432][0][0];	// L569
    v428[(v432 + (v429 * 8))][(v430 * 2)][(v431 * 2)] = v433;	// L570
    float v434 = v427[v432][0][1];	// L571
    v428[(v432 + (v429 * 8))][(v430 * 2)][((v431 * 2) + 1)] = v434;	// L572
    float v435 = v427[v432][1][0];	// L573
    v428[(v432 + (v429 * 8))][((v430 * 2) + 1)][(v431 * 2)] = v435;	// L574
    float v436 = v427[v432][1][1];	// L575
    v428[(v432 + (v429 * 8))][((v430 * 2) + 1)][((v431 * 2) + 1)] = v436;	// L576
  }
}

void forward_node33(
  float v437[8][2][2],
  float v438[8][2][2],
  float v439[8][8],
  float v440[8][2][2],
  float v441[8][2][2],
  int v442,
  int v443,
  int v444
) {	// L580
  #pragma HLS inline
  #pragma HLS array_partition variable=v437 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v437 cyclic factor=2 dim=3
  #pragma HLS resource variable=v437 core=ram_t2p_bram

  #pragma HLS array_partition variable=v438 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v438 cyclic factor=2 dim=3
  #pragma HLS resource variable=v438 core=ram_t2p_bram

  #pragma HLS resource variable=v439 core=ram_t2p_bram

  #pragma HLS array_partition variable=v440 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v440 cyclic factor=2 dim=3
  #pragma HLS resource variable=v440 core=ram_t2p_bram

  #pragma HLS array_partition variable=v441 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v441 cyclic factor=2 dim=3
  #pragma HLS resource variable=v441 core=ram_t2p_bram

  for (int v445 = 0; v445 < 8; v445 += 1) {	// L581
    for (int v446 = 0; v446 < 8; v446 += 1) {	// L582
      #pragma HLS pipeline II=1
      float v447 = v437[v446][0][0];	// L583
      float v448 = v440[v446][0][0];	// L584
      float v449 = v441[v446][0][0];	// L585
      float v450 = (v445 == 0) ? v448 : v449;	// L586
      float v451 = ((v445 + (v443 * 8)) == 0 && v442 == 0 && v444 == 0) ? v447 : v450;	// L587
      float v452 = v438[v445][0][0];	// L588
      float v453 = v439[v446][v445];	// L589
      float v454 = v452 * v453;	// L590
      float v455 = v451 + v454;	// L591
      v441[v446][0][0] = v455;	// L592
      float v456 = v437[v446][0][1];	// L593
      float v457 = v440[v446][0][1];	// L594
      float v458 = v441[v446][0][1];	// L595
      float v459 = (v445 == 0) ? v457 : v458;	// L596
      float v460 = ((v445 + (v443 * 8)) == 0 && v442 == 0 && v444 == 0) ? v456 : v459;	// L597
      float v461 = v438[v445][0][1];	// L598
      float v462 = v461 * v453;	// L599
      float v463 = v460 + v462;	// L600
      v441[v446][0][1] = v463;	// L601
      float v464 = v437[v446][1][0];	// L602
      float v465 = v440[v446][1][0];	// L603
      float v466 = v441[v446][1][0];	// L604
      float v467 = (v445 == 0) ? v465 : v466;	// L605
      float v468 = ((v445 + (v443 * 8)) == 0 && v442 == 0 && v444 == 0) ? v464 : v467;	// L606
      float v469 = v438[v445][1][0];	// L607
      float v470 = v469 * v453;	// L608
      float v471 = v468 + v470;	// L609
      v441[v446][1][0] = v471;	// L610
      float v472 = v437[v446][1][1];	// L611
      float v473 = v440[v446][1][1];	// L612
      float v474 = v441[v446][1][1];	// L613
      float v475 = (v445 == 0) ? v473 : v474;	// L614
      float v476 = ((v445 + (v443 * 8)) == 0 && v442 == 0 && v444 == 0) ? v472 : v475;	// L615
      float v477 = v438[v445][1][1];	// L616
      float v478 = v477 * v453;	// L617
      float v479 = v476 + v478;	// L618
      v441[v446][1][1] = v479;	// L619
    }
  }
}

void forward_node34(
  float v480[512][512][3][3],
  float v481[8][8],
  int v482,
  int v483,
  int v484,
  int v485
) {	// L624
  #pragma HLS inline
  #pragma HLS resource variable=v481 core=ram_t2p_bram

  for (int v486 = 0; v486 < 8; v486 += 1) {	// L625
    for (int v487 = 0; v487 < 8; v487 += 1) {	// L626
      #pragma HLS pipeline II=1
      float v488 = v480[(v486 + (v484 * 8))][(v487 + (v485 * 8))][v482][v483];	// L627
      v481[v486][v487] = v488;	// L628
    }
  }
}

void forward_node35(
  float v489[512][4][4],
  float v490[8][2][2],
  int v491,
  int v492,
  int v493,
  int v494,
  int v495
) {	// L633
  #pragma HLS inline
  #pragma HLS array_partition variable=v489 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v489 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v490 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v490 cyclic factor=2 dim=3
  #pragma HLS resource variable=v490 core=ram_t2p_bram

  for (int v496 = 0; v496 < 8; v496 += 1) {	// L634
    #pragma HLS pipeline II=1
    float v497 = v489[(v496 + (v491 * 8))][((v492 + (v493 * 2)) - 1)][((v494 + (v495 * 2)) - 1)];	// L635
    v490[v496][0][0] = v497;	// L636
    float v498 = v489[(v496 + (v491 * 8))][((v492 + (v493 * 2)) - 1)][(v494 + (v495 * 2))];	// L637
    v490[v496][0][1] = v498;	// L638
    float v499 = v489[(v496 + (v491 * 8))][(v492 + (v493 * 2))][((v494 + (v495 * 2)) - 1)];	// L639
    v490[v496][1][0] = v499;	// L640
    float v500 = v489[(v496 + (v491 * 8))][(v492 + (v493 * 2))][(v494 + (v495 * 2))];	// L641
    v490[v496][1][1] = v500;	// L642
  }
}

void forward_node36(
  float v501[512][4][4],
  float v502[8][2][2],
  int v503,
  int v504,
  int v505
) {	// L646
  #pragma HLS inline
  #pragma HLS array_partition variable=v501 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v501 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v502 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v502 cyclic factor=2 dim=3
  #pragma HLS resource variable=v502 core=ram_t2p_bram

  for (int v506 = 0; v506 < 8; v506 += 1) {	// L647
    #pragma HLS pipeline II=1
    float v507 = v501[(v506 + (v503 * 8))][(v504 * 2)][(v505 * 2)];	// L648
    v502[v506][0][0] = v507;	// L649
    float v508 = v501[(v506 + (v503 * 8))][(v504 * 2)][((v505 * 2) + 1)];	// L650
    v502[v506][0][1] = v508;	// L651
    float v509 = v501[(v506 + (v503 * 8))][((v504 * 2) + 1)][(v505 * 2)];	// L652
    v502[v506][1][0] = v509;	// L653
    float v510 = v501[(v506 + (v503 * 8))][((v504 * 2) + 1)][((v505 * 2) + 1)];	// L654
    v502[v506][1][1] = v510;	// L655
  }
}

void forward_node37(
  float v511[512][4][4],
  float v512[8][2][2],
  int v513,
  int v514,
  int v515
) {	// L659
  #pragma HLS inline
  #pragma HLS array_partition variable=v511 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v511 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v512 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v512 cyclic factor=2 dim=3
  #pragma HLS resource variable=v512 core=ram_t2p_bram

  for (int v516 = 0; v516 < 8; v516 += 1) {	// L660
    #pragma HLS pipeline II=1
    float v517 = v511[(v516 + (v513 * 8))][(v514 * 2)][(v515 * 2)];	// L661
    v512[v516][0][0] = v517;	// L662
    float v518 = v511[(v516 + (v513 * 8))][(v514 * 2)][((v515 * 2) + 1)];	// L663
    v512[v516][0][1] = v518;	// L664
    float v519 = v511[(v516 + (v513 * 8))][((v514 * 2) + 1)][(v515 * 2)];	// L665
    v512[v516][1][0] = v519;	// L666
    float v520 = v511[(v516 + (v513 * 8))][((v514 * 2) + 1)][((v515 * 2) + 1)];	// L667
    v512[v516][1][1] = v520;	// L668
  }
}

void forward_node31(
  float v521[512][4][4],
  float v522[512][4][4],
  float v523[512][512][3][3],
  float v524[512][4][4],
  float v525[512][4][4]
) {	// L672
  #pragma HLS array_partition variable=v521 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v521 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v522 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v522 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v524 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v524 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v525 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v525 cyclic factor=2 dim=3

  for (int v526 = 0; v526 < 147456; v526 += 1) {	// L673
    #pragma HLS dataflow
    int v527 = (v526 % 2);	// L674
    int v528 = ((v526 / 2) % 2);	// L675
    int v529 = (((v526 / 2) / 2) % 64);	// L676
    int v530 = ((((v526 / 2) / 2) / 64) % 3);	// L677
    int v531 = (((((v526 / 2) / 2) / 64) / 3) % 3);	// L678
    int v532 = (((((v526 / 2) / 2) / 64) / 3) / 3);	// L679
    float v533[8][8];	// L680
    #pragma HLS resource variable=v533 core=ram_t2p_bram

    float v534[8][2][2];	// L681
    #pragma HLS array_partition variable=v534 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v534 cyclic factor=2 dim=3
    #pragma HLS resource variable=v534 core=ram_t2p_bram

    float v535[8][2][2];	// L682
    #pragma HLS array_partition variable=v535 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v535 cyclic factor=2 dim=3
    #pragma HLS resource variable=v535 core=ram_t2p_bram

    float v536[8][2][2];	// L683
    #pragma HLS array_partition variable=v536 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v536 cyclic factor=2 dim=3
    #pragma HLS resource variable=v536 core=ram_t2p_bram

    forward_node37(v521, v536, v529, v528, v527);	// L684
    forward_node36(v524, v535, v529, v528, v527);	// L685
    forward_node35(v522, v534, v532, v531, v528, v530, v527);	// L686
    forward_node34(v523, v533, v531, v530, v529, v532);	// L687
    float v537[8][2][2];	// L688
    #pragma HLS array_partition variable=v537 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v537 cyclic factor=2 dim=3
    #pragma HLS resource variable=v537 core=ram_t2p_bram

    forward_node33(v536, v534, v533, v535, v537, v531, v532, v530);	// L689
    forward_node32(v537, v525, v529, v528, v527);	// L690
  }
}

void forward_node39(
  float v538[8][2][2],
  float v539[512][4][4],
  int v540,
  int v541,
  int v542
) {	// L694
  #pragma HLS inline
  #pragma HLS array_partition variable=v538 cyclic factor=2 dim=3
  #pragma HLS resource variable=v538 core=ram_t2p_bram

  #pragma HLS array_partition variable=v539 cyclic factor=2 dim=3

  for (int v543 = 0; v543 < 8; v543 += 1) {	// L695
    for (int v544 = 0; v544 < 2; v544 += 1) {	// L696
      #pragma HLS pipeline II=1
      float v545 = v538[v543][v544][0];	// L697
      v539[(v543 + (v540 * 8))][(v544 + (v541 * 2))][(v542 * 2)] = v545;	// L698
      float v546 = v538[v543][v544][1];	// L699
      v539[(v543 + (v540 * 8))][(v544 + (v541 * 2))][((v542 * 2) + 1)] = v546;	// L700
    }
  }
}

void forward_node40(
  float v547[8][2][2],
  float v548[8][8],
  float v549[8][2][2],
  float v550[8][2][2],
  float v551[8][2][2],
  int v552,
  int v553,
  int v554
) {	// L705
  #pragma HLS inline
  #pragma HLS array_partition variable=v547 cyclic factor=2 dim=3
  #pragma HLS resource variable=v547 core=ram_t2p_bram

  #pragma HLS resource variable=v548 core=ram_t2p_bram

  #pragma HLS array_partition variable=v549 cyclic factor=2 dim=3
  #pragma HLS resource variable=v549 core=ram_t2p_bram

  #pragma HLS array_partition variable=v550 cyclic factor=2 dim=3
  #pragma HLS resource variable=v550 core=ram_t2p_bram

  #pragma HLS array_partition variable=v551 cyclic factor=2 dim=3
  #pragma HLS resource variable=v551 core=ram_t2p_bram

  for (int v555 = 0; v555 < 8; v555 += 1) {	// L707
    for (int v556 = 0; v556 < 8; v556 += 1) {	// L708
      for (int v557 = 0; v557 < 2; v557 += 1) {	// L709
        #pragma HLS pipeline II=1
        float v558 = v547[v556][v557][0];	// L710
        float v559 = v550[v556][v557][0];	// L711
        float v560 = v551[v556][v557][0];	// L712
        float v561 = (v555 == 0) ? v559 : v560;	// L713
        float v562 = ((v555 + (v553 * 8)) == 0 && v554 == 0 && v552 == 0) ? v558 : v561;	// L714
        float v563 = v549[v555][v557][0];	// L715
        float v564 = v548[v556][v555];	// L716
        float v565 = v563 * v564;	// L717
        float v566 = v562 + v565;	// L718
        bool v567 = v566 > (float)0.000000;	// L719
        float v568 = v567 ? v566 : (float)0.000000;	// L720
        float v569 = ((((-v555) + (v553 * -8)) + 255) == 0 && ((-v554) + 2) == 0 && ((-v552) + 2) == 0) ? v568 : v566;	// L721
        v551[v556][v557][0] = v569;	// L722
        float v570 = v547[v556][v557][1];	// L723
        float v571 = v550[v556][v557][1];	// L724
        float v572 = v551[v556][v557][1];	// L725
        float v573 = (v555 == 0) ? v571 : v572;	// L726
        float v574 = ((v555 + (v553 * 8)) == 0 && v554 == 0 && v552 == 0) ? v570 : v573;	// L727
        float v575 = v549[v555][v557][1];	// L728
        float v576 = v575 * v564;	// L729
        float v577 = v574 + v576;	// L730
        bool v578 = v577 > (float)0.000000;	// L731
        float v579 = v578 ? v577 : (float)0.000000;	// L732
        float v580 = ((((-v555) + (v553 * -8)) + 255) == 0 && ((-v554) + 2) == 0 && ((-v552) + 2) == 0) ? v579 : v577;	// L733
        v551[v556][v557][1] = v580;	// L734
      }
    }
  }
}

void forward_node41(
  float v581[512][256][3][3],
  float v582[8][8],
  int v583,
  int v584,
  int v585,
  int v586
) {	// L740
  #pragma HLS inline
  #pragma HLS resource variable=v582 core=ram_t2p_bram

  for (int v587 = 0; v587 < 8; v587 += 1) {	// L741
    for (int v588 = 0; v588 < 8; v588 += 1) {	// L742
      #pragma HLS pipeline II=1
      float v589 = v581[(v587 + (v585 * 8))][(v588 + (v586 * 8))][v583][v584];	// L743
      v582[v587][v588] = v589;	// L744
    }
  }
}

void forward_node42(
  float v590[256][8][8],
  float v591[8][2][2],
  int v592,
  int v593,
  int v594,
  int v595,
  int v596
) {	// L749
  #pragma HLS inline
  #pragma HLS array_partition variable=v590 cyclic factor=4 dim=3

  #pragma HLS array_partition variable=v591 cyclic factor=2 dim=3
  #pragma HLS resource variable=v591 core=ram_t2p_bram

  for (int v597 = 0; v597 < 8; v597 += 1) {	// L750
    for (int v598 = 0; v598 < 2; v598 += 1) {	// L751
      #pragma HLS pipeline II=1
      float v599 = v590[(v597 + (v592 * 8))][((((v598 * 2) + v593) + (v594 * 4)) - 1)][((v595 + (v596 * 4)) - 1)];	// L752
      v591[v597][v598][0] = v599;	// L753
      float v600 = v590[(v597 + (v592 * 8))][((((v598 * 2) + v593) + (v594 * 4)) - 1)][((v595 + (v596 * 4)) + 1)];	// L754
      v591[v597][v598][1] = v600;	// L755
    }
  }
}

void forward_node43(
  float v601[512][4][4],
  float v602[8][2][2],
  int v603,
  int v604,
  int v605
) {	// L760
  #pragma HLS inline
  #pragma HLS array_partition variable=v601 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v602 cyclic factor=2 dim=3
  #pragma HLS resource variable=v602 core=ram_t2p_bram

  for (int v606 = 0; v606 < 8; v606 += 1) {	// L761
    for (int v607 = 0; v607 < 2; v607 += 1) {	// L762
      #pragma HLS pipeline II=1
      float v608 = v601[(v606 + (v603 * 8))][(v607 + (v604 * 2))][(v605 * 2)];	// L763
      v602[v606][v607][0] = v608;	// L764
      float v609 = v601[(v606 + (v603 * 8))][(v607 + (v604 * 2))][((v605 * 2) + 1)];	// L765
      v602[v606][v607][1] = v609;	// L766
    }
  }
}

void forward_node44(
  float v610[512][4][4],
  float v611[8][2][2],
  int v612,
  int v613,
  int v614
) {	// L771
  #pragma HLS inline
  #pragma HLS array_partition variable=v610 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v611 cyclic factor=2 dim=3
  #pragma HLS resource variable=v611 core=ram_t2p_bram

  for (int v615 = 0; v615 < 8; v615 += 1) {	// L772
    for (int v616 = 0; v616 < 2; v616 += 1) {	// L773
      #pragma HLS pipeline II=1
      float v617 = v610[(v615 + (v612 * 8))][(v616 + (v613 * 2))][(v614 * 2)];	// L774
      v611[v615][v616][0] = v617;	// L775
      float v618 = v610[(v615 + (v612 * 8))][(v616 + (v613 * 2))][((v614 * 2) + 1)];	// L776
      v611[v615][v616][1] = v618;	// L777
    }
  }
}

void forward_node38(
  float v619[256][8][8],
  float v620[512][256][3][3],
  float v621[512][4][4],
  float v622[512][4][4],
  float v623[512][4][4]
) {	// L782
  #pragma HLS array_partition variable=v619 cyclic factor=4 dim=3

  #pragma HLS array_partition variable=v621 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v622 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v623 cyclic factor=2 dim=3

  for (int v624 = 0; v624 < 73728; v624 += 1) {	// L783
    #pragma HLS dataflow
    int v625 = (v624 % 2);	// L784
    int v626 = ((v624 / 2) % 2);	// L785
    int v627 = (((v624 / 2) / 2) % 64);	// L786
    int v628 = ((((v624 / 2) / 2) / 64) % 3);	// L787
    int v629 = (((((v624 / 2) / 2) / 64) / 3) % 3);	// L788
    int v630 = (((((v624 / 2) / 2) / 64) / 3) / 3);	// L789
    float v631[8][8];	// L790
    #pragma HLS resource variable=v631 core=ram_t2p_bram

    float v632[8][2][2];	// L791
    #pragma HLS array_partition variable=v632 cyclic factor=2 dim=3
    #pragma HLS resource variable=v632 core=ram_t2p_bram

    float v633[8][2][2];	// L792
    #pragma HLS array_partition variable=v633 cyclic factor=2 dim=3
    #pragma HLS resource variable=v633 core=ram_t2p_bram

    float v634[8][2][2];	// L793
    #pragma HLS array_partition variable=v634 cyclic factor=2 dim=3
    #pragma HLS resource variable=v634 core=ram_t2p_bram

    forward_node44(v621, v634, v627, v626, v625);	// L794
    forward_node43(v622, v633, v627, v626, v625);	// L795
    forward_node42(v619, v632, v630, v629, v626, v628, v625);	// L796
    forward_node41(v620, v631, v629, v628, v627, v630);	// L797
    float v635[8][2][2];	// L798
    #pragma HLS array_partition variable=v635 cyclic factor=2 dim=3
    #pragma HLS resource variable=v635 core=ram_t2p_bram

    forward_node40(v634, v631, v632, v633, v635, v628, v630, v629);	// L799
    forward_node39(v635, v623, v627, v626, v625);	// L800
  }
}

void forward_node46(
  float v636[8][4][4],
  float v637[256][8][8],
  int v638,
  int v639,
  int v640
) {	// L804
  #pragma HLS inline
  #pragma HLS array_partition variable=v636 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v636 cyclic factor=2 dim=3
  #pragma HLS resource variable=v636 core=ram_t2p_bram

  #pragma HLS array_partition variable=v637 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v637 cyclic factor=2 dim=3

  for (int v641 = 0; v641 < 8; v641 += 1) {	// L805
    for (int v642 = 0; v642 < 4; v642 += 2) {	// L806
      for (int v643 = 0; v643 < 4; v643 += 2) {	// L807
        #pragma HLS pipeline II=1
        float v644 = v636[v641][v642][v643];	// L808
        v637[(v641 + (v638 * 8))][(v642 + (v639 * 4))][(v643 + (v640 * 4))] = v644;	// L809
        float v645 = v636[v641][v642][(v643 + 1)];	// L810
        v637[(v641 + (v638 * 8))][(v642 + (v639 * 4))][((v643 + (v640 * 4)) + 1)] = v645;	// L811
        float v646 = v636[v641][(v642 + 1)][v643];	// L812
        v637[(v641 + (v638 * 8))][((v642 + (v639 * 4)) + 1)][(v643 + (v640 * 4))] = v646;	// L813
        float v647 = v636[v641][(v642 + 1)][(v643 + 1)];	// L814
        v637[(v641 + (v638 * 8))][((v642 + (v639 * 4)) + 1)][((v643 + (v640 * 4)) + 1)] = v647;	// L815
      }
    }
  }
}

void forward_node47(
  float v648[8][4][4],
  float v649[256][8][8],
  int v650,
  int v651,
  int v652
) {	// L821
  #pragma HLS inline
  #pragma HLS array_partition variable=v648 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v648 cyclic factor=2 dim=3
  #pragma HLS resource variable=v648 core=ram_t2p_bram

  #pragma HLS array_partition variable=v649 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v649 cyclic factor=2 dim=3

  for (int v653 = 0; v653 < 8; v653 += 1) {	// L822
    for (int v654 = 0; v654 < 4; v654 += 2) {	// L823
      for (int v655 = 0; v655 < 4; v655 += 2) {	// L824
        #pragma HLS pipeline II=1
        float v656 = v648[v653][v654][v655];	// L825
        v649[(v653 + (v650 * 8))][(v654 + (v651 * 4))][(v655 + (v652 * 4))] = v656;	// L826
        float v657 = v648[v653][v654][(v655 + 1)];	// L827
        v649[(v653 + (v650 * 8))][(v654 + (v651 * 4))][((v655 + (v652 * 4)) + 1)] = v657;	// L828
        float v658 = v648[v653][(v654 + 1)][v655];	// L829
        v649[(v653 + (v650 * 8))][((v654 + (v651 * 4)) + 1)][(v655 + (v652 * 4))] = v658;	// L830
        float v659 = v648[v653][(v654 + 1)][(v655 + 1)];	// L831
        v649[(v653 + (v650 * 8))][((v654 + (v651 * 4)) + 1)][((v655 + (v652 * 4)) + 1)] = v659;	// L832
      }
    }
  }
}

void forward_node48(
  float v660[8][4][4],
  float v661[8][8],
  float v662[8][4][4],
  float v663[8][4][4],
  float v664[8][4][4],
  float v665[8][4][4],
  int v666,
  int v667,
  int v668
) {	// L838
  #pragma HLS inline
  #pragma HLS array_partition variable=v660 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v660 cyclic factor=2 dim=3
  #pragma HLS resource variable=v660 core=ram_t2p_bram

  #pragma HLS resource variable=v661 core=ram_t2p_bram

  #pragma HLS array_partition variable=v662 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v662 cyclic factor=2 dim=3
  #pragma HLS resource variable=v662 core=ram_t2p_bram

  #pragma HLS array_partition variable=v663 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v663 cyclic factor=2 dim=3
  #pragma HLS resource variable=v663 core=ram_t2p_bram

  #pragma HLS array_partition variable=v664 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v664 cyclic factor=2 dim=3
  #pragma HLS resource variable=v664 core=ram_t2p_bram

  #pragma HLS array_partition variable=v665 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v665 cyclic factor=2 dim=3
  #pragma HLS resource variable=v665 core=ram_t2p_bram

  for (int v669 = 0; v669 < 8; v669 += 1) {	// L840
    for (int v670 = 0; v670 < 8; v670 += 1) {	// L841
      for (int v671 = 0; v671 < 4; v671 += 2) {	// L842
        for (int v672 = 0; v672 < 4; v672 += 2) {	// L843
          #pragma HLS pipeline II=1
          float v673 = v662[v669][v671][v672];	// L844
          float v674 = v661[v670][v669];	// L845
          float v675 = v663[v670][v671][v672];	// L846
          float v676 = v665[v670][v671][v672];	// L847
          float v677 = (v669 == 0) ? v675 : v676;	// L848
          float v678 = v673 * v674;	// L849
          float v679 = v677 + v678;	// L850
          v665[v670][v671][v672] = v679;	// L851
          float v680 = v660[v670][v671][v672];	// L852
          float v681 = v679 + v680;	// L853
          bool v682 = v681 > (float)0.000000;	// L854
          float v683 = v682 ? v681 : (float)0.000000;	// L855
          if ((((-v669) + (v668 * -8)) + 255) == 0 && ((-v667) + 2) == 0 && ((-v666) + 2) == 0) {	// L856
            v664[v670][v671][v672] = v683;	// L857
          }
          float v684 = v662[v669][v671][(v672 + 1)];	// L859
          float v685 = v663[v670][v671][(v672 + 1)];	// L860
          float v686 = v665[v670][v671][(v672 + 1)];	// L861
          float v687 = (v669 == 0) ? v685 : v686;	// L862
          float v688 = v684 * v674;	// L863
          float v689 = v687 + v688;	// L864
          v665[v670][v671][(v672 + 1)] = v689;	// L865
          float v690 = v660[v670][v671][(v672 + 1)];	// L866
          float v691 = v689 + v690;	// L867
          bool v692 = v691 > (float)0.000000;	// L868
          float v693 = v692 ? v691 : (float)0.000000;	// L869
          if ((((-v669) + (v668 * -8)) + 255) == 0 && ((-v667) + 2) == 0 && ((-v666) + 2) == 0) {	// L870
            v664[v670][v671][(v672 + 1)] = v693;	// L871
          }
          float v694 = v662[v669][(v671 + 1)][v672];	// L873
          float v695 = v663[v670][(v671 + 1)][v672];	// L874
          float v696 = v665[v670][(v671 + 1)][v672];	// L875
          float v697 = (v669 == 0) ? v695 : v696;	// L876
          float v698 = v694 * v674;	// L877
          float v699 = v697 + v698;	// L878
          v665[v670][(v671 + 1)][v672] = v699;	// L879
          float v700 = v660[v670][(v671 + 1)][v672];	// L880
          float v701 = v699 + v700;	// L881
          bool v702 = v701 > (float)0.000000;	// L882
          float v703 = v702 ? v701 : (float)0.000000;	// L883
          if ((((-v669) + (v668 * -8)) + 255) == 0 && ((-v667) + 2) == 0 && ((-v666) + 2) == 0) {	// L884
            v664[v670][(v671 + 1)][v672] = v703;	// L885
          }
          float v704 = v662[v669][(v671 + 1)][(v672 + 1)];	// L887
          float v705 = v663[v670][(v671 + 1)][(v672 + 1)];	// L888
          float v706 = v665[v670][(v671 + 1)][(v672 + 1)];	// L889
          float v707 = (v669 == 0) ? v705 : v706;	// L890
          float v708 = v704 * v674;	// L891
          float v709 = v707 + v708;	// L892
          v665[v670][(v671 + 1)][(v672 + 1)] = v709;	// L893
          float v710 = v660[v670][(v671 + 1)][(v672 + 1)];	// L894
          float v711 = v709 + v710;	// L895
          bool v712 = v711 > (float)0.000000;	// L896
          float v713 = v712 ? v711 : (float)0.000000;	// L897
          if ((((-v669) + (v668 * -8)) + 255) == 0 && ((-v667) + 2) == 0 && ((-v666) + 2) == 0) {	// L898
            v664[v670][(v671 + 1)][(v672 + 1)] = v713;	// L899
          }
        }
      }
    }
  }
}

void forward_node49(
  float v714[256][8][8],
  float v715[8][4][4],
  int v716,
  int v717,
  int v718
) {	// L907
  #pragma HLS inline
  #pragma HLS array_partition variable=v714 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v714 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v715 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v715 cyclic factor=2 dim=3
  #pragma HLS resource variable=v715 core=ram_t2p_bram

  for (int v719 = 0; v719 < 8; v719 += 1) {	// L908
    for (int v720 = 0; v720 < 4; v720 += 2) {	// L909
      for (int v721 = 0; v721 < 4; v721 += 2) {	// L910
        #pragma HLS pipeline II=1
        float v722 = v714[(v719 + (v716 * 8))][(v720 + (v717 * 4))][(v721 + (v718 * 4))];	// L911
        v715[v719][v720][v721] = v722;	// L912
        float v723 = v714[(v719 + (v716 * 8))][(v720 + (v717 * 4))][((v721 + (v718 * 4)) + 1)];	// L913
        v715[v719][v720][(v721 + 1)] = v723;	// L914
        float v724 = v714[(v719 + (v716 * 8))][((v720 + (v717 * 4)) + 1)][(v721 + (v718 * 4))];	// L915
        v715[v719][(v720 + 1)][v721] = v724;	// L916
        float v725 = v714[(v719 + (v716 * 8))][((v720 + (v717 * 4)) + 1)][((v721 + (v718 * 4)) + 1)];	// L917
        v715[v719][(v720 + 1)][(v721 + 1)] = v725;	// L918
      }
    }
  }
}

void forward_node50(
  float v726[256][8][8],
  float v727[8][4][4],
  int v728,
  int v729,
  int v730
) {	// L924
  #pragma HLS inline
  #pragma HLS array_partition variable=v726 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v726 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v727 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v727 cyclic factor=2 dim=3
  #pragma HLS resource variable=v727 core=ram_t2p_bram

  for (int v731 = 0; v731 < 8; v731 += 1) {	// L925
    for (int v732 = 0; v732 < 4; v732 += 2) {	// L926
      for (int v733 = 0; v733 < 4; v733 += 2) {	// L927
        #pragma HLS pipeline II=1
        float v734 = v726[(v731 + (v728 * 8))][(v732 + (v729 * 4))][(v733 + (v730 * 4))];	// L928
        v727[v731][v732][v733] = v734;	// L929
        float v735 = v726[(v731 + (v728 * 8))][(v732 + (v729 * 4))][((v733 + (v730 * 4)) + 1)];	// L930
        v727[v731][v732][(v733 + 1)] = v735;	// L931
        float v736 = v726[(v731 + (v728 * 8))][((v732 + (v729 * 4)) + 1)][(v733 + (v730 * 4))];	// L932
        v727[v731][(v732 + 1)][v733] = v736;	// L933
        float v737 = v726[(v731 + (v728 * 8))][((v732 + (v729 * 4)) + 1)][((v733 + (v730 * 4)) + 1)];	// L934
        v727[v731][(v732 + 1)][(v733 + 1)] = v737;	// L935
      }
    }
  }
}

void forward_node51(
  float v738[256][256][3][3],
  float v739[8][8],
  int v740,
  int v741,
  int v742,
  int v743
) {	// L941
  #pragma HLS inline
  #pragma HLS resource variable=v739 core=ram_t2p_bram

  for (int v744 = 0; v744 < 8; v744 += 1) {	// L942
    for (int v745 = 0; v745 < 8; v745 += 1) {	// L943
      #pragma HLS pipeline II=1
      float v746 = v738[(v744 + (v742 * 8))][(v745 + (v743 * 8))][v740][v741];	// L944
      v739[v744][v745] = v746;	// L945
    }
  }
}

void forward_node52(
  float v747[256][8][8],
  float v748[8][4][4],
  int v749,
  int v750,
  int v751,
  int v752,
  int v753
) {	// L950
  #pragma HLS inline
  #pragma HLS array_partition variable=v747 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v747 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v748 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v748 cyclic factor=2 dim=3
  #pragma HLS resource variable=v748 core=ram_t2p_bram

  for (int v754 = 0; v754 < 8; v754 += 1) {	// L951
    for (int v755 = 0; v755 < 4; v755 += 2) {	// L952
      for (int v756 = 0; v756 < 4; v756 += 2) {	// L953
        #pragma HLS pipeline II=1
        float v757 = v747[(v754 + (v749 * 8))][(((v755 + v750) + (v751 * 4)) - 1)][(((v756 + v752) + (v753 * 4)) - 1)];	// L954
        v748[v754][v755][v756] = v757;	// L955
        float v758 = v747[(v754 + (v749 * 8))][(((v755 + v750) + (v751 * 4)) - 1)][((v756 + v752) + (v753 * 4))];	// L956
        v748[v754][v755][(v756 + 1)] = v758;	// L957
        float v759 = v747[(v754 + (v749 * 8))][((v755 + v750) + (v751 * 4))][(((v756 + v752) + (v753 * 4)) - 1)];	// L958
        v748[v754][(v755 + 1)][v756] = v759;	// L959
        float v760 = v747[(v754 + (v749 * 8))][((v755 + v750) + (v751 * 4))][((v756 + v752) + (v753 * 4))];	// L960
        v748[v754][(v755 + 1)][(v756 + 1)] = v760;	// L961
      }
    }
  }
}

void forward_node45(
  float v761[256][256][3][3],
  float v762[256][8][8],
  float v763[256][8][8],
  float v764[256][8][8],
  float v765[256][8][8],
  float v766[256][8][8]
) {	// L967
  #pragma HLS array_partition variable=v762 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v762 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v763 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v763 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v764 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v764 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v765 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v765 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v766 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v766 cyclic factor=2 dim=3

  for (int v767 = 0; v767 < 36864; v767 += 1) {	// L968
    #pragma HLS dataflow
    int v768 = (v767 % 2);	// L969
    int v769 = ((v767 / 2) % 2);	// L970
    int v770 = (((v767 / 2) / 2) % 32);	// L971
    int v771 = ((((v767 / 2) / 2) / 32) % 3);	// L972
    int v772 = (((((v767 / 2) / 2) / 32) / 3) % 3);	// L973
    int v773 = (((((v767 / 2) / 2) / 32) / 3) / 3);	// L974
    float v774[8][4][4];	// L975
    #pragma HLS array_partition variable=v774 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v774 cyclic factor=2 dim=3
    #pragma HLS resource variable=v774 core=ram_t2p_bram

    float v775[8][4][4];	// L976
    #pragma HLS array_partition variable=v775 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v775 cyclic factor=2 dim=3
    #pragma HLS resource variable=v775 core=ram_t2p_bram

    float v776[8][4][4];	// L977
    #pragma HLS array_partition variable=v776 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v776 cyclic factor=2 dim=3
    #pragma HLS resource variable=v776 core=ram_t2p_bram

    float v777[8][8];	// L978
    #pragma HLS resource variable=v777 core=ram_t2p_bram

    float v778[8][4][4];	// L979
    #pragma HLS array_partition variable=v778 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v778 cyclic factor=2 dim=3
    #pragma HLS resource variable=v778 core=ram_t2p_bram

    forward_node52(v763, v778, v773, v772, v769, v771, v768);	// L980
    forward_node51(v761, v777, v772, v771, v770, v773);	// L981
    forward_node50(v764, v776, v770, v769, v768);	// L982
    forward_node49(v762, v775, v770, v769, v768);	// L983
    float v779[8][4][4];	// L984
    #pragma HLS array_partition variable=v779 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v779 cyclic factor=2 dim=3
    #pragma HLS resource variable=v779 core=ram_t2p_bram

    forward_node48(v775, v777, v778, v776, v774, v779, v771, v772, v773);	// L985
    forward_node47(v779, v766, v770, v769, v768);	// L986
    forward_node46(v774, v765, v770, v769, v768);	// L987
  }
}

void forward_node54(
  float v780[8][4][4],
  float v781[256][8][8],
  int v782,
  int v783,
  int v784
) {	// L991
  #pragma HLS inline
  #pragma HLS array_partition variable=v780 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v780 cyclic factor=2 dim=3
  #pragma HLS resource variable=v780 core=ram_t2p_bram

  #pragma HLS array_partition variable=v781 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v781 cyclic factor=2 dim=3

  for (int v785 = 0; v785 < 8; v785 += 1) {	// L992
    for (int v786 = 0; v786 < 4; v786 += 2) {	// L993
      for (int v787 = 0; v787 < 4; v787 += 2) {	// L994
        #pragma HLS pipeline II=1
        float v788 = v780[v785][v786][v787];	// L995
        v781[(v785 + (v782 * 8))][(v786 + (v783 * 4))][(v787 + (v784 * 4))] = v788;	// L996
        float v789 = v780[v785][v786][(v787 + 1)];	// L997
        v781[(v785 + (v782 * 8))][(v786 + (v783 * 4))][((v787 + (v784 * 4)) + 1)] = v789;	// L998
        float v790 = v780[v785][(v786 + 1)][v787];	// L999
        v781[(v785 + (v782 * 8))][((v786 + (v783 * 4)) + 1)][(v787 + (v784 * 4))] = v790;	// L1000
        float v791 = v780[v785][(v786 + 1)][(v787 + 1)];	// L1001
        v781[(v785 + (v782 * 8))][((v786 + (v783 * 4)) + 1)][((v787 + (v784 * 4)) + 1)] = v791;	// L1002
      }
    }
  }
}

void forward_node55(
  float v792[8][4][4],
  float v793[8][4][4],
  float v794[8][8],
  float v795[8][4][4],
  float v796[8][4][4],
  int v797,
  int v798,
  int v799
) {	// L1008
  #pragma HLS inline
  #pragma HLS array_partition variable=v792 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v792 cyclic factor=2 dim=3
  #pragma HLS resource variable=v792 core=ram_t2p_bram

  #pragma HLS array_partition variable=v793 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v793 cyclic factor=2 dim=3
  #pragma HLS resource variable=v793 core=ram_t2p_bram

  #pragma HLS resource variable=v794 core=ram_t2p_bram

  #pragma HLS array_partition variable=v795 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v795 cyclic factor=2 dim=3
  #pragma HLS resource variable=v795 core=ram_t2p_bram

  #pragma HLS array_partition variable=v796 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v796 cyclic factor=2 dim=3
  #pragma HLS resource variable=v796 core=ram_t2p_bram

  for (int v800 = 0; v800 < 8; v800 += 1) {	// L1010
    for (int v801 = 0; v801 < 8; v801 += 1) {	// L1011
      for (int v802 = 0; v802 < 4; v802 += 2) {	// L1012
        for (int v803 = 0; v803 < 4; v803 += 2) {	// L1013
          #pragma HLS pipeline II=1
          float v804 = v793[v801][v802][v803];	// L1014
          float v805 = v795[v801][v802][v803];	// L1015
          float v806 = v796[v801][v802][v803];	// L1016
          float v807 = (v800 == 0) ? v805 : v806;	// L1017
          float v808 = ((v800 + (v799 * 8)) == 0 && v797 == 0 && v798 == 0) ? v804 : v807;	// L1018
          float v809 = v792[v800][v802][v803];	// L1019
          float v810 = v794[v801][v800];	// L1020
          float v811 = v809 * v810;	// L1021
          float v812 = v808 + v811;	// L1022
          bool v813 = v812 > (float)0.000000;	// L1023
          float v814 = v813 ? v812 : (float)0.000000;	// L1024
          float v815 = ((((-v800) + (v799 * -8)) + 255) == 0 && ((-v797) + 2) == 0 && ((-v798) + 2) == 0) ? v814 : v812;	// L1025
          v796[v801][v802][v803] = v815;	// L1026
          float v816 = v793[v801][v802][(v803 + 1)];	// L1027
          float v817 = v795[v801][v802][(v803 + 1)];	// L1028
          float v818 = v796[v801][v802][(v803 + 1)];	// L1029
          float v819 = (v800 == 0) ? v817 : v818;	// L1030
          float v820 = ((v800 + (v799 * 8)) == 0 && v797 == 0 && v798 == 0) ? v816 : v819;	// L1031
          float v821 = v792[v800][v802][(v803 + 1)];	// L1032
          float v822 = v821 * v810;	// L1033
          float v823 = v820 + v822;	// L1034
          bool v824 = v823 > (float)0.000000;	// L1035
          float v825 = v824 ? v823 : (float)0.000000;	// L1036
          float v826 = ((((-v800) + (v799 * -8)) + 255) == 0 && ((-v797) + 2) == 0 && ((-v798) + 2) == 0) ? v825 : v823;	// L1037
          v796[v801][v802][(v803 + 1)] = v826;	// L1038
          float v827 = v793[v801][(v802 + 1)][v803];	// L1039
          float v828 = v795[v801][(v802 + 1)][v803];	// L1040
          float v829 = v796[v801][(v802 + 1)][v803];	// L1041
          float v830 = (v800 == 0) ? v828 : v829;	// L1042
          float v831 = ((v800 + (v799 * 8)) == 0 && v797 == 0 && v798 == 0) ? v827 : v830;	// L1043
          float v832 = v792[v800][(v802 + 1)][v803];	// L1044
          float v833 = v832 * v810;	// L1045
          float v834 = v831 + v833;	// L1046
          bool v835 = v834 > (float)0.000000;	// L1047
          float v836 = v835 ? v834 : (float)0.000000;	// L1048
          float v837 = ((((-v800) + (v799 * -8)) + 255) == 0 && ((-v797) + 2) == 0 && ((-v798) + 2) == 0) ? v836 : v834;	// L1049
          v796[v801][(v802 + 1)][v803] = v837;	// L1050
          float v838 = v793[v801][(v802 + 1)][(v803 + 1)];	// L1051
          float v839 = v795[v801][(v802 + 1)][(v803 + 1)];	// L1052
          float v840 = v796[v801][(v802 + 1)][(v803 + 1)];	// L1053
          float v841 = (v800 == 0) ? v839 : v840;	// L1054
          float v842 = ((v800 + (v799 * 8)) == 0 && v797 == 0 && v798 == 0) ? v838 : v841;	// L1055
          float v843 = v792[v800][(v802 + 1)][(v803 + 1)];	// L1056
          float v844 = v843 * v810;	// L1057
          float v845 = v842 + v844;	// L1058
          bool v846 = v845 > (float)0.000000;	// L1059
          float v847 = v846 ? v845 : (float)0.000000;	// L1060
          float v848 = ((((-v800) + (v799 * -8)) + 255) == 0 && ((-v797) + 2) == 0 && ((-v798) + 2) == 0) ? v847 : v845;	// L1061
          v796[v801][(v802 + 1)][(v803 + 1)] = v848;	// L1062
        }
      }
    }
  }
}

void forward_node56(
  float v849[256][256][3][3],
  float v850[8][8],
  int v851,
  int v852,
  int v853,
  int v854
) {	// L1069
  #pragma HLS inline
  #pragma HLS resource variable=v850 core=ram_t2p_bram

  for (int v855 = 0; v855 < 8; v855 += 1) {	// L1070
    for (int v856 = 0; v856 < 8; v856 += 1) {	// L1071
      #pragma HLS pipeline II=1
      float v857 = v849[(v855 + (v853 * 8))][(v856 + (v854 * 8))][v851][v852];	// L1072
      v850[v855][v856] = v857;	// L1073
    }
  }
}

void forward_node57(
  float v858[256][8][8],
  float v859[8][4][4],
  int v860,
  int v861,
  int v862,
  int v863,
  int v864
) {	// L1078
  #pragma HLS inline
  #pragma HLS array_partition variable=v858 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v858 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v859 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v859 cyclic factor=2 dim=3
  #pragma HLS resource variable=v859 core=ram_t2p_bram

  for (int v865 = 0; v865 < 8; v865 += 1) {	// L1079
    for (int v866 = 0; v866 < 4; v866 += 2) {	// L1080
      for (int v867 = 0; v867 < 4; v867 += 2) {	// L1081
        #pragma HLS pipeline II=1
        float v868 = v858[(v865 + (v860 * 8))][(((v866 + v861) + (v862 * 4)) - 1)][(((v867 + v863) + (v864 * 4)) - 1)];	// L1082
        v859[v865][v866][v867] = v868;	// L1083
        float v869 = v858[(v865 + (v860 * 8))][(((v866 + v861) + (v862 * 4)) - 1)][((v867 + v863) + (v864 * 4))];	// L1084
        v859[v865][v866][(v867 + 1)] = v869;	// L1085
        float v870 = v858[(v865 + (v860 * 8))][((v866 + v861) + (v862 * 4))][(((v867 + v863) + (v864 * 4)) - 1)];	// L1086
        v859[v865][(v866 + 1)][v867] = v870;	// L1087
        float v871 = v858[(v865 + (v860 * 8))][((v866 + v861) + (v862 * 4))][((v867 + v863) + (v864 * 4))];	// L1088
        v859[v865][(v866 + 1)][(v867 + 1)] = v871;	// L1089
      }
    }
  }
}

void forward_node58(
  float v872[256][8][8],
  float v873[8][4][4],
  int v874,
  int v875,
  int v876
) {	// L1095
  #pragma HLS inline
  #pragma HLS array_partition variable=v872 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v872 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v873 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v873 cyclic factor=2 dim=3
  #pragma HLS resource variable=v873 core=ram_t2p_bram

  for (int v877 = 0; v877 < 8; v877 += 1) {	// L1096
    for (int v878 = 0; v878 < 4; v878 += 2) {	// L1097
      for (int v879 = 0; v879 < 4; v879 += 2) {	// L1098
        #pragma HLS pipeline II=1
        float v880 = v872[(v877 + (v874 * 8))][(v878 + (v875 * 4))][(v879 + (v876 * 4))];	// L1099
        v873[v877][v878][v879] = v880;	// L1100
        float v881 = v872[(v877 + (v874 * 8))][(v878 + (v875 * 4))][((v879 + (v876 * 4)) + 1)];	// L1101
        v873[v877][v878][(v879 + 1)] = v881;	// L1102
        float v882 = v872[(v877 + (v874 * 8))][((v878 + (v875 * 4)) + 1)][(v879 + (v876 * 4))];	// L1103
        v873[v877][(v878 + 1)][v879] = v882;	// L1104
        float v883 = v872[(v877 + (v874 * 8))][((v878 + (v875 * 4)) + 1)][((v879 + (v876 * 4)) + 1)];	// L1105
        v873[v877][(v878 + 1)][(v879 + 1)] = v883;	// L1106
      }
    }
  }
}

void forward_node59(
  float v884[256][8][8],
  float v885[8][4][4],
  int v886,
  int v887,
  int v888
) {	// L1112
  #pragma HLS inline
  #pragma HLS array_partition variable=v884 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v884 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v885 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v885 cyclic factor=2 dim=3
  #pragma HLS resource variable=v885 core=ram_t2p_bram

  for (int v889 = 0; v889 < 8; v889 += 1) {	// L1113
    for (int v890 = 0; v890 < 4; v890 += 2) {	// L1114
      for (int v891 = 0; v891 < 4; v891 += 2) {	// L1115
        #pragma HLS pipeline II=1
        float v892 = v884[(v889 + (v886 * 8))][(v890 + (v887 * 4))][(v891 + (v888 * 4))];	// L1116
        v885[v889][v890][v891] = v892;	// L1117
        float v893 = v884[(v889 + (v886 * 8))][(v890 + (v887 * 4))][((v891 + (v888 * 4)) + 1)];	// L1118
        v885[v889][v890][(v891 + 1)] = v893;	// L1119
        float v894 = v884[(v889 + (v886 * 8))][((v890 + (v887 * 4)) + 1)][(v891 + (v888 * 4))];	// L1120
        v885[v889][(v890 + 1)][v891] = v894;	// L1121
        float v895 = v884[(v889 + (v886 * 8))][((v890 + (v887 * 4)) + 1)][((v891 + (v888 * 4)) + 1)];	// L1122
        v885[v889][(v890 + 1)][(v891 + 1)] = v895;	// L1123
      }
    }
  }
}

void forward_node53(
  float v896[256][256][3][3],
  float v897[256][8][8],
  float v898[256][8][8],
  float v899[256][8][8],
  float v900[256][8][8]
) {	// L1129
  #pragma HLS array_partition variable=v897 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v897 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v898 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v898 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v899 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v899 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v900 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v900 cyclic factor=2 dim=3

  for (int v901 = 0; v901 < 36864; v901 += 1) {	// L1130
    #pragma HLS dataflow
    int v902 = (v901 % 2);	// L1131
    int v903 = ((v901 / 2) % 2);	// L1132
    int v904 = (((v901 / 2) / 2) % 32);	// L1133
    int v905 = ((((v901 / 2) / 2) / 32) % 3);	// L1134
    int v906 = (((((v901 / 2) / 2) / 32) / 3) % 3);	// L1135
    int v907 = (((((v901 / 2) / 2) / 32) / 3) / 3);	// L1136
    float v908[8][8];	// L1137
    #pragma HLS resource variable=v908 core=ram_t2p_bram

    float v909[8][4][4];	// L1138
    #pragma HLS array_partition variable=v909 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v909 cyclic factor=2 dim=3
    #pragma HLS resource variable=v909 core=ram_t2p_bram

    float v910[8][4][4];	// L1139
    #pragma HLS array_partition variable=v910 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v910 cyclic factor=2 dim=3
    #pragma HLS resource variable=v910 core=ram_t2p_bram

    float v911[8][4][4];	// L1140
    #pragma HLS array_partition variable=v911 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v911 cyclic factor=2 dim=3
    #pragma HLS resource variable=v911 core=ram_t2p_bram

    forward_node59(v898, v911, v904, v903, v902);	// L1141
    forward_node58(v899, v910, v904, v903, v902);	// L1142
    forward_node57(v897, v909, v907, v906, v903, v905, v902);	// L1143
    forward_node56(v896, v908, v906, v905, v904, v907);	// L1144
    float v912[8][4][4];	// L1145
    #pragma HLS array_partition variable=v912 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v912 cyclic factor=2 dim=3
    #pragma HLS resource variable=v912 core=ram_t2p_bram

    forward_node55(v909, v911, v908, v910, v912, v906, v905, v907);	// L1146
    forward_node54(v912, v900, v904, v903, v902);	// L1147
  }
}

void forward_node61(
  float v913[8][4][4],
  float v914[256][8][8],
  int v915,
  int v916,
  int v917
) {	// L1151
  #pragma HLS inline
  #pragma HLS resource variable=v913 core=ram_t2p_bram

  for (int v918 = 0; v918 < 8; v918 += 1) {	// L1152
    for (int v919 = 0; v919 < 4; v919 += 1) {	// L1153
      for (int v920 = 0; v920 < 4; v920 += 1) {	// L1154
        #pragma HLS pipeline II=1
        float v921 = v913[v918][v919][v920];	// L1155
        v914[(v918 + (v915 * 8))][(v919 + (v916 * 4))][(v920 + (v917 * 4))] = v921;	// L1156
      }
    }
  }
}

void forward_node62(
  float v922[8][4][4],
  float v923[256][8][8],
  int v924,
  int v925,
  int v926
) {	// L1162
  #pragma HLS inline
  #pragma HLS resource variable=v922 core=ram_t2p_bram

  for (int v927 = 0; v927 < 8; v927 += 1) {	// L1163
    for (int v928 = 0; v928 < 4; v928 += 1) {	// L1164
      for (int v929 = 0; v929 < 4; v929 += 1) {	// L1165
        #pragma HLS pipeline II=1
        float v930 = v922[v927][v928][v929];	// L1166
        v923[(v927 + (v924 * 8))][(v928 + (v925 * 4))][(v929 + (v926 * 4))] = v930;	// L1167
      }
    }
  }
}

void forward_node63(
  float v931[8][4][4],
  float v932[8][4][4],
  float v933[8][8],
  float v934[8][4][4],
  float v935[8][4][4],
  float v936[8][4][4],
  float v937[8][4][4],
  int v938
) {	// L1173
  #pragma HLS inline
  #pragma HLS resource variable=v931 core=ram_t2p_bram

  #pragma HLS resource variable=v932 core=ram_t2p_bram

  #pragma HLS resource variable=v933 core=ram_t2p_bram

  #pragma HLS resource variable=v934 core=ram_t2p_bram

  #pragma HLS resource variable=v935 core=ram_t2p_bram

  #pragma HLS resource variable=v936 core=ram_t2p_bram

  #pragma HLS resource variable=v937 core=ram_t2p_bram

  for (int v939 = 0; v939 < 8; v939 += 1) {	// L1175
    for (int v940 = 0; v940 < 8; v940 += 1) {	// L1176
      for (int v941 = 0; v941 < 4; v941 += 1) {	// L1177
        for (int v942 = 0; v942 < 4; v942 += 1) {	// L1178
          #pragma HLS pipeline II=1
          float v943 = v931[v940][v941][v942];	// L1179
          float v944 = v935[v940][v941][v942];	// L1180
          float v945 = v936[v940][v941][v942];	// L1181
          float v946 = (v939 == 0) ? v944 : v945;	// L1182
          float v947 = ((v939 + (v938 * 8)) == 0) ? v943 : v946;	// L1183
          float v948 = v934[v939][v941][v942];	// L1184
          float v949 = v933[v940][v939];	// L1185
          float v950 = v948 * v949;	// L1186
          float v951 = v947 + v950;	// L1187
          v936[v940][v941][v942] = v951;	// L1188
          float v952 = v932[v940][v941][v942];	// L1189
          float v953 = v952 + v951;	// L1190
          bool v954 = v953 > (float)0.000000;	// L1191
          float v955 = v954 ? v953 : (float)0.000000;	// L1192
          if ((((-v939) + (v938 * -8)) + 127) == 0) {	// L1193
            v937[v940][v941][v942] = v955;	// L1194
          }
        }
      }
    }
  }
}

void forward_node64(
  float v956[256][8][8],
  float v957[8][4][4],
  int v958,
  int v959,
  int v960
) {	// L1202
  #pragma HLS inline
  #pragma HLS resource variable=v957 core=ram_t2p_bram

  for (int v961 = 0; v961 < 8; v961 += 1) {	// L1203
    for (int v962 = 0; v962 < 4; v962 += 1) {	// L1204
      for (int v963 = 0; v963 < 4; v963 += 1) {	// L1205
        #pragma HLS pipeline II=1
        float v964 = v956[(v961 + (v958 * 8))][(v962 + (v959 * 4))][(v963 + (v960 * 4))];	// L1206
        v957[v961][v962][v963] = v964;	// L1207
      }
    }
  }
}

void forward_node65(
  float v965[256][128],
  float v966[8][8],
  int v967,
  int v968
) {	// L1213
  #pragma HLS inline
  #pragma HLS resource variable=v966 core=ram_t2p_bram

  for (int v969 = 0; v969 < 8; v969 += 1) {	// L1214
    for (int v970 = 0; v970 < 8; v970 += 1) {	// L1215
      #pragma HLS pipeline II=1
      float v971 = v965[(v969 + (v967 * 8))][(v970 + (v968 * 8))];	// L1216
      v966[v969][v970] = v971;	// L1217
    }
  }
}

void forward_node66(
  float v972[128][16][16],
  float v973[8][4][4],
  int v974,
  int v975,
  int v976
) {	// L1222
  #pragma HLS inline
  #pragma HLS resource variable=v973 core=ram_t2p_bram

  for (int v977 = 0; v977 < 8; v977 += 1) {	// L1223
    for (int v978 = 0; v978 < 4; v978 += 1) {	// L1224
      for (int v979 = 0; v979 < 4; v979 += 1) {	// L1225
        #pragma HLS pipeline II=1
        float v980 = v972[(v977 + (v974 * 8))][((v978 * 2) + (v975 * 8))][((v979 * 2) + (v976 * 8))];	// L1226
        v973[v977][v978][v979] = v980;	// L1227
      }
    }
  }
}

void forward_node67(
  float v981[256][8][8],
  float v982[8][4][4],
  int v983,
  int v984,
  int v985
) {	// L1233
  #pragma HLS inline
  #pragma HLS resource variable=v982 core=ram_t2p_bram

  for (int v986 = 0; v986 < 8; v986 += 1) {	// L1234
    for (int v987 = 0; v987 < 4; v987 += 1) {	// L1235
      for (int v988 = 0; v988 < 4; v988 += 1) {	// L1236
        #pragma HLS pipeline II=1
        float v989 = v981[(v986 + (v983 * 8))][(v987 + (v984 * 4))][(v988 + (v985 * 4))];	// L1237
        v982[v986][v987][v988] = v989;	// L1238
      }
    }
  }
}

void forward_node68(
  float v990[256][8][8],
  float v991[8][4][4],
  int v992,
  int v993,
  int v994
) {	// L1244
  #pragma HLS inline
  #pragma HLS resource variable=v991 core=ram_t2p_bram

  for (int v995 = 0; v995 < 8; v995 += 1) {	// L1245
    for (int v996 = 0; v996 < 4; v996 += 1) {	// L1246
      for (int v997 = 0; v997 < 4; v997 += 1) {	// L1247
        #pragma HLS pipeline II=1
        float v998 = v990[(v995 + (v992 * 8))][(v996 + (v993 * 4))][(v997 + (v994 * 4))];	// L1248
        v991[v995][v996][v997] = v998;	// L1249
      }
    }
  }
}

void forward_node60(
  float v999[256][128],
  float v1000[256][8][8],
  float v1001[256][8][8],
  float v1002[128][16][16],
  float v1003[256][8][8],
  float v1004[256][8][8],
  float v1005[256][8][8]
) {	// L1255
  for (int v1006 = 0; v1006 < 2048; v1006 += 1) {	// L1256
    #pragma HLS dataflow
    int v1007 = (v1006 % 2);	// L1257
    int v1008 = ((v1006 / 2) % 2);	// L1258
    int v1009 = (((v1006 / 2) / 2) % 32);	// L1259
    int v1010 = (((v1006 / 2) / 2) / 32);	// L1260
    float v1011[8][4][4];	// L1261
    #pragma HLS resource variable=v1011 core=ram_t2p_bram

    float v1012[8][4][4];	// L1262
    #pragma HLS resource variable=v1012 core=ram_t2p_bram

    float v1013[8][8];	// L1263
    #pragma HLS resource variable=v1013 core=ram_t2p_bram

    float v1014[8][4][4];	// L1264
    #pragma HLS resource variable=v1014 core=ram_t2p_bram

    float v1015[8][4][4];	// L1265
    #pragma HLS resource variable=v1015 core=ram_t2p_bram

    float v1016[8][4][4];	// L1266
    #pragma HLS resource variable=v1016 core=ram_t2p_bram

    forward_node68(v1001, v1016, v1009, v1008, v1007);	// L1267
    forward_node67(v1003, v1015, v1009, v1008, v1007);	// L1268
    forward_node66(v1002, v1014, v1010, v1008, v1007);	// L1269
    forward_node65(v999, v1013, v1009, v1010);	// L1270
    forward_node64(v1000, v1012, v1009, v1008, v1007);	// L1271
    float v1017[8][4][4];	// L1272
    #pragma HLS resource variable=v1017 core=ram_t2p_bram

    forward_node63(v1016, v1012, v1013, v1014, v1015, v1017, v1011, v1010);	// L1273
    forward_node62(v1017, v1004, v1009, v1008, v1007);	// L1274
    forward_node61(v1011, v1005, v1009, v1008, v1007);	// L1275
  }
}

void forward_node70(
  float v1018[8][4][4],
  float v1019[256][8][8],
  int v1020,
  int v1021,
  int v1022
) {	// L1279
  #pragma HLS inline
  #pragma HLS array_partition variable=v1018 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1018 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1018 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1019 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1019 cyclic factor=2 dim=3

  for (int v1023 = 0; v1023 < 8; v1023 += 1) {	// L1280
    for (int v1024 = 0; v1024 < 4; v1024 += 2) {	// L1281
      for (int v1025 = 0; v1025 < 4; v1025 += 2) {	// L1282
        #pragma HLS pipeline II=1
        float v1026 = v1018[v1023][v1024][v1025];	// L1283
        v1019[(v1023 + (v1020 * 8))][(v1024 + (v1021 * 4))][(v1025 + (v1022 * 4))] = v1026;	// L1284
        float v1027 = v1018[v1023][v1024][(v1025 + 1)];	// L1285
        v1019[(v1023 + (v1020 * 8))][(v1024 + (v1021 * 4))][((v1025 + (v1022 * 4)) + 1)] = v1027;	// L1286
        float v1028 = v1018[v1023][(v1024 + 1)][v1025];	// L1287
        v1019[(v1023 + (v1020 * 8))][((v1024 + (v1021 * 4)) + 1)][(v1025 + (v1022 * 4))] = v1028;	// L1288
        float v1029 = v1018[v1023][(v1024 + 1)][(v1025 + 1)];	// L1289
        v1019[(v1023 + (v1020 * 8))][((v1024 + (v1021 * 4)) + 1)][((v1025 + (v1022 * 4)) + 1)] = v1029;	// L1290
      }
    }
  }
}

void forward_node71(
  float v1030[8][8],
  float v1031[8][4][4],
  float v1032[8][4][4],
  float v1033[8][4][4],
  float v1034[8][4][4],
  int v1035,
  int v1036,
  int v1037
) {	// L1296
  #pragma HLS inline
  #pragma HLS resource variable=v1030 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1031 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1031 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1031 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1032 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1032 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1032 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1033 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1033 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1033 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1034 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1034 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1034 core=ram_t2p_bram

  for (int v1038 = 0; v1038 < 8; v1038 += 1) {	// L1297
    for (int v1039 = 0; v1039 < 8; v1039 += 1) {	// L1298
      for (int v1040 = 0; v1040 < 4; v1040 += 2) {	// L1299
        for (int v1041 = 0; v1041 < 4; v1041 += 2) {	// L1300
          #pragma HLS pipeline II=1
          float v1042 = v1032[v1039][v1040][v1041];	// L1301
          float v1043 = v1033[v1039][v1040][v1041];	// L1302
          float v1044 = v1034[v1039][v1040][v1041];	// L1303
          float v1045 = (v1038 == 0) ? v1043 : v1044;	// L1304
          float v1046 = ((v1038 + (v1036 * 8)) == 0 && v1035 == 0 && v1037 == 0) ? v1042 : v1045;	// L1305
          float v1047 = v1031[v1038][v1040][v1041];	// L1306
          float v1048 = v1030[v1039][v1038];	// L1307
          float v1049 = v1047 * v1048;	// L1308
          float v1050 = v1046 + v1049;	// L1309
          v1034[v1039][v1040][v1041] = v1050;	// L1310
          float v1051 = v1032[v1039][v1040][(v1041 + 1)];	// L1311
          float v1052 = v1033[v1039][v1040][(v1041 + 1)];	// L1312
          float v1053 = v1034[v1039][v1040][(v1041 + 1)];	// L1313
          float v1054 = (v1038 == 0) ? v1052 : v1053;	// L1314
          float v1055 = ((v1038 + (v1036 * 8)) == 0 && v1035 == 0 && v1037 == 0) ? v1051 : v1054;	// L1315
          float v1056 = v1031[v1038][v1040][(v1041 + 1)];	// L1316
          float v1057 = v1056 * v1048;	// L1317
          float v1058 = v1055 + v1057;	// L1318
          v1034[v1039][v1040][(v1041 + 1)] = v1058;	// L1319
          float v1059 = v1032[v1039][(v1040 + 1)][v1041];	// L1320
          float v1060 = v1033[v1039][(v1040 + 1)][v1041];	// L1321
          float v1061 = v1034[v1039][(v1040 + 1)][v1041];	// L1322
          float v1062 = (v1038 == 0) ? v1060 : v1061;	// L1323
          float v1063 = ((v1038 + (v1036 * 8)) == 0 && v1035 == 0 && v1037 == 0) ? v1059 : v1062;	// L1324
          float v1064 = v1031[v1038][(v1040 + 1)][v1041];	// L1325
          float v1065 = v1064 * v1048;	// L1326
          float v1066 = v1063 + v1065;	// L1327
          v1034[v1039][(v1040 + 1)][v1041] = v1066;	// L1328
          float v1067 = v1032[v1039][(v1040 + 1)][(v1041 + 1)];	// L1329
          float v1068 = v1033[v1039][(v1040 + 1)][(v1041 + 1)];	// L1330
          float v1069 = v1034[v1039][(v1040 + 1)][(v1041 + 1)];	// L1331
          float v1070 = (v1038 == 0) ? v1068 : v1069;	// L1332
          float v1071 = ((v1038 + (v1036 * 8)) == 0 && v1035 == 0 && v1037 == 0) ? v1067 : v1070;	// L1333
          float v1072 = v1031[v1038][(v1040 + 1)][(v1041 + 1)];	// L1334
          float v1073 = v1072 * v1048;	// L1335
          float v1074 = v1071 + v1073;	// L1336
          v1034[v1039][(v1040 + 1)][(v1041 + 1)] = v1074;	// L1337
        }
      }
    }
  }
}

void forward_node72(
  float v1075[256][256][3][3],
  float v1076[8][8],
  int v1077,
  int v1078,
  int v1079,
  int v1080
) {	// L1344
  #pragma HLS inline
  #pragma HLS resource variable=v1076 core=ram_t2p_bram

  for (int v1081 = 0; v1081 < 8; v1081 += 1) {	// L1345
    for (int v1082 = 0; v1082 < 8; v1082 += 1) {	// L1346
      #pragma HLS pipeline II=1
      float v1083 = v1075[(v1081 + (v1079 * 8))][(v1082 + (v1080 * 8))][v1077][v1078];	// L1347
      v1076[v1081][v1082] = v1083;	// L1348
    }
  }
}

void forward_node73(
  float v1084[256][8][8],
  float v1085[8][4][4],
  int v1086,
  int v1087,
  int v1088,
  int v1089,
  int v1090
) {	// L1353
  #pragma HLS inline
  #pragma HLS array_partition variable=v1084 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1084 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1085 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1085 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1085 core=ram_t2p_bram

  for (int v1091 = 0; v1091 < 8; v1091 += 1) {	// L1354
    for (int v1092 = 0; v1092 < 4; v1092 += 2) {	// L1355
      for (int v1093 = 0; v1093 < 4; v1093 += 2) {	// L1356
        #pragma HLS pipeline II=1
        float v1094 = v1084[(v1091 + (v1086 * 8))][(((v1092 + v1087) + (v1088 * 4)) - 1)][(((v1093 + v1089) + (v1090 * 4)) - 1)];	// L1357
        v1085[v1091][v1092][v1093] = v1094;	// L1358
        float v1095 = v1084[(v1091 + (v1086 * 8))][(((v1092 + v1087) + (v1088 * 4)) - 1)][((v1093 + v1089) + (v1090 * 4))];	// L1359
        v1085[v1091][v1092][(v1093 + 1)] = v1095;	// L1360
        float v1096 = v1084[(v1091 + (v1086 * 8))][((v1092 + v1087) + (v1088 * 4))][(((v1093 + v1089) + (v1090 * 4)) - 1)];	// L1361
        v1085[v1091][(v1092 + 1)][v1093] = v1096;	// L1362
        float v1097 = v1084[(v1091 + (v1086 * 8))][((v1092 + v1087) + (v1088 * 4))][((v1093 + v1089) + (v1090 * 4))];	// L1363
        v1085[v1091][(v1092 + 1)][(v1093 + 1)] = v1097;	// L1364
      }
    }
  }
}

void forward_node74(
  float v1098[256][8][8],
  float v1099[8][4][4],
  int v1100,
  int v1101,
  int v1102
) {	// L1370
  #pragma HLS inline
  #pragma HLS array_partition variable=v1098 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1098 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1099 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1099 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1099 core=ram_t2p_bram

  for (int v1103 = 0; v1103 < 8; v1103 += 1) {	// L1371
    for (int v1104 = 0; v1104 < 4; v1104 += 2) {	// L1372
      for (int v1105 = 0; v1105 < 4; v1105 += 2) {	// L1373
        #pragma HLS pipeline II=1
        float v1106 = v1098[(v1103 + (v1100 * 8))][(v1104 + (v1101 * 4))][(v1105 + (v1102 * 4))];	// L1374
        v1099[v1103][v1104][v1105] = v1106;	// L1375
        float v1107 = v1098[(v1103 + (v1100 * 8))][(v1104 + (v1101 * 4))][((v1105 + (v1102 * 4)) + 1)];	// L1376
        v1099[v1103][v1104][(v1105 + 1)] = v1107;	// L1377
        float v1108 = v1098[(v1103 + (v1100 * 8))][((v1104 + (v1101 * 4)) + 1)][(v1105 + (v1102 * 4))];	// L1378
        v1099[v1103][(v1104 + 1)][v1105] = v1108;	// L1379
        float v1109 = v1098[(v1103 + (v1100 * 8))][((v1104 + (v1101 * 4)) + 1)][((v1105 + (v1102 * 4)) + 1)];	// L1380
        v1099[v1103][(v1104 + 1)][(v1105 + 1)] = v1109;	// L1381
      }
    }
  }
}

void forward_node75(
  float v1110[256][8][8],
  float v1111[8][4][4],
  int v1112,
  int v1113,
  int v1114
) {	// L1387
  #pragma HLS inline
  #pragma HLS array_partition variable=v1110 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1110 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1111 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1111 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1111 core=ram_t2p_bram

  for (int v1115 = 0; v1115 < 8; v1115 += 1) {	// L1388
    for (int v1116 = 0; v1116 < 4; v1116 += 2) {	// L1389
      for (int v1117 = 0; v1117 < 4; v1117 += 2) {	// L1390
        #pragma HLS pipeline II=1
        float v1118 = v1110[(v1115 + (v1112 * 8))][(v1116 + (v1113 * 4))][(v1117 + (v1114 * 4))];	// L1391
        v1111[v1115][v1116][v1117] = v1118;	// L1392
        float v1119 = v1110[(v1115 + (v1112 * 8))][(v1116 + (v1113 * 4))][((v1117 + (v1114 * 4)) + 1)];	// L1393
        v1111[v1115][v1116][(v1117 + 1)] = v1119;	// L1394
        float v1120 = v1110[(v1115 + (v1112 * 8))][((v1116 + (v1113 * 4)) + 1)][(v1117 + (v1114 * 4))];	// L1395
        v1111[v1115][(v1116 + 1)][v1117] = v1120;	// L1396
        float v1121 = v1110[(v1115 + (v1112 * 8))][((v1116 + (v1113 * 4)) + 1)][((v1117 + (v1114 * 4)) + 1)];	// L1397
        v1111[v1115][(v1116 + 1)][(v1117 + 1)] = v1121;	// L1398
      }
    }
  }
}

void forward_node69(
  float v1122[256][8][8],
  float v1123[256][8][8],
  float v1124[256][256][3][3],
  float v1125[256][8][8],
  float v1126[256][8][8]
) {	// L1404
  #pragma HLS array_partition variable=v1122 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1122 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1123 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1123 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1125 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1125 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1126 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1126 cyclic factor=2 dim=3

  for (int v1127 = 0; v1127 < 36864; v1127 += 1) {	// L1405
    #pragma HLS dataflow
    int v1128 = (v1127 % 2);	// L1406
    int v1129 = ((v1127 / 2) % 2);	// L1407
    int v1130 = (((v1127 / 2) / 2) % 32);	// L1408
    int v1131 = ((((v1127 / 2) / 2) / 32) % 3);	// L1409
    int v1132 = (((((v1127 / 2) / 2) / 32) / 3) % 3);	// L1410
    int v1133 = (((((v1127 / 2) / 2) / 32) / 3) / 3);	// L1411
    float v1134[8][8];	// L1412
    #pragma HLS resource variable=v1134 core=ram_t2p_bram

    float v1135[8][4][4];	// L1413
    #pragma HLS array_partition variable=v1135 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1135 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1135 core=ram_t2p_bram

    float v1136[8][4][4];	// L1414
    #pragma HLS array_partition variable=v1136 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1136 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1136 core=ram_t2p_bram

    float v1137[8][4][4];	// L1415
    #pragma HLS array_partition variable=v1137 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1137 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1137 core=ram_t2p_bram

    forward_node75(v1123, v1137, v1130, v1129, v1128);	// L1416
    forward_node74(v1125, v1136, v1130, v1129, v1128);	// L1417
    forward_node73(v1122, v1135, v1133, v1132, v1129, v1131, v1128);	// L1418
    forward_node72(v1124, v1134, v1132, v1131, v1130, v1133);	// L1419
    float v1138[8][4][4];	// L1420
    #pragma HLS array_partition variable=v1138 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1138 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1138 core=ram_t2p_bram

    forward_node71(v1134, v1135, v1137, v1136, v1138, v1132, v1133, v1131);	// L1421
    forward_node70(v1138, v1126, v1130, v1129, v1128);	// L1422
  }
}

void forward_node77(
  float v1139[8][4][4],
  float v1140[256][8][8],
  int v1141,
  int v1142,
  int v1143
) {	// L1426
  #pragma HLS inline
  #pragma HLS array_partition variable=v1139 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1139 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1140 cyclic factor=2 dim=3

  for (int v1144 = 0; v1144 < 8; v1144 += 1) {	// L1427
    for (int v1145 = 0; v1145 < 4; v1145 += 1) {	// L1428
      for (int v1146 = 0; v1146 < 4; v1146 += 2) {	// L1429
        #pragma HLS pipeline II=1
        float v1147 = v1139[v1144][v1145][v1146];	// L1430
        v1140[(v1144 + (v1141 * 8))][(v1145 + (v1142 * 4))][(v1146 + (v1143 * 4))] = v1147;	// L1431
        float v1148 = v1139[v1144][v1145][(v1146 + 1)];	// L1432
        v1140[(v1144 + (v1141 * 8))][(v1145 + (v1142 * 4))][((v1146 + (v1143 * 4)) + 1)] = v1148;	// L1433
      }
    }
  }
}

void forward_node78(
  float v1149[8][4][4],
  float v1150[8][4][4],
  float v1151[8][8],
  float v1152[8][4][4],
  float v1153[8][4][4],
  int v1154,
  int v1155,
  int v1156
) {	// L1439
  #pragma HLS inline
  #pragma HLS array_partition variable=v1149 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1149 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1150 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1150 core=ram_t2p_bram

  #pragma HLS resource variable=v1151 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1152 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1152 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1153 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1153 core=ram_t2p_bram

  for (int v1157 = 0; v1157 < 8; v1157 += 1) {	// L1441
    for (int v1158 = 0; v1158 < 8; v1158 += 1) {	// L1442
      for (int v1159 = 0; v1159 < 4; v1159 += 1) {	// L1443
        for (int v1160 = 0; v1160 < 4; v1160 += 2) {	// L1444
          #pragma HLS pipeline II=1
          float v1161 = v1150[v1158][v1159][v1160];	// L1445
          float v1162 = v1152[v1158][v1159][v1160];	// L1446
          float v1163 = v1153[v1158][v1159][v1160];	// L1447
          float v1164 = (v1157 == 0) ? v1162 : v1163;	// L1448
          float v1165 = ((v1157 + (v1154 * 8)) == 0 && v1155 == 0 && v1156 == 0) ? v1161 : v1164;	// L1449
          float v1166 = v1149[v1157][v1159][v1160];	// L1450
          float v1167 = v1151[v1158][v1157];	// L1451
          float v1168 = v1166 * v1167;	// L1452
          float v1169 = v1165 + v1168;	// L1453
          bool v1170 = v1169 > (float)0.000000;	// L1454
          float v1171 = v1170 ? v1169 : (float)0.000000;	// L1455
          float v1172 = ((((-v1157) + (v1154 * -8)) + 127) == 0 && ((-v1155) + 2) == 0 && ((-v1156) + 2) == 0) ? v1171 : v1169;	// L1456
          v1153[v1158][v1159][v1160] = v1172;	// L1457
          float v1173 = v1150[v1158][v1159][(v1160 + 1)];	// L1458
          float v1174 = v1152[v1158][v1159][(v1160 + 1)];	// L1459
          float v1175 = v1153[v1158][v1159][(v1160 + 1)];	// L1460
          float v1176 = (v1157 == 0) ? v1174 : v1175;	// L1461
          float v1177 = ((v1157 + (v1154 * 8)) == 0 && v1155 == 0 && v1156 == 0) ? v1173 : v1176;	// L1462
          float v1178 = v1149[v1157][v1159][(v1160 + 1)];	// L1463
          float v1179 = v1178 * v1167;	// L1464
          float v1180 = v1177 + v1179;	// L1465
          bool v1181 = v1180 > (float)0.000000;	// L1466
          float v1182 = v1181 ? v1180 : (float)0.000000;	// L1467
          float v1183 = ((((-v1157) + (v1154 * -8)) + 127) == 0 && ((-v1155) + 2) == 0 && ((-v1156) + 2) == 0) ? v1182 : v1180;	// L1468
          v1153[v1158][v1159][(v1160 + 1)] = v1183;	// L1469
        }
      }
    }
  }
}

void forward_node79(
  float v1184[256][128][3][3],
  float v1185[8][8],
  int v1186,
  int v1187,
  int v1188,
  int v1189
) {	// L1476
  #pragma HLS inline
  #pragma HLS resource variable=v1185 core=ram_t2p_bram

  for (int v1190 = 0; v1190 < 8; v1190 += 1) {	// L1477
    for (int v1191 = 0; v1191 < 8; v1191 += 1) {	// L1478
      #pragma HLS pipeline II=1
      float v1192 = v1184[(v1190 + (v1188 * 8))][(v1191 + (v1189 * 8))][v1186][v1187];	// L1479
      v1185[v1190][v1191] = v1192;	// L1480
    }
  }
}

void forward_node80(
  float v1193[128][16][16],
  float v1194[8][4][4],
  int v1195,
  int v1196,
  int v1197,
  int v1198,
  int v1199
) {	// L1485
  #pragma HLS inline
  #pragma HLS array_partition variable=v1193 cyclic factor=4 dim=3

  #pragma HLS array_partition variable=v1194 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1194 core=ram_t2p_bram

  for (int v1200 = 0; v1200 < 8; v1200 += 1) {	// L1486
    for (int v1201 = 0; v1201 < 4; v1201 += 1) {	// L1487
      for (int v1202 = 0; v1202 < 4; v1202 += 2) {	// L1488
        #pragma HLS pipeline II=1
        float v1203 = v1193[(v1200 + (v1195 * 8))][((((v1201 * 2) + v1196) + (v1197 * 8)) - 1)][((((v1202 * 2) + v1198) + (v1199 * 8)) - 1)];	// L1489
        v1194[v1200][v1201][v1202] = v1203;	// L1490
        float v1204 = v1193[(v1200 + (v1195 * 8))][((((v1201 * 2) + v1196) + (v1197 * 8)) - 1)][((((v1202 * 2) + v1198) + (v1199 * 8)) + 1)];	// L1491
        v1194[v1200][v1201][(v1202 + 1)] = v1204;	// L1492
      }
    }
  }
}

void forward_node81(
  float v1205[256][8][8],
  float v1206[8][4][4],
  int v1207,
  int v1208,
  int v1209
) {	// L1498
  #pragma HLS inline
  #pragma HLS array_partition variable=v1205 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1206 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1206 core=ram_t2p_bram

  for (int v1210 = 0; v1210 < 8; v1210 += 1) {	// L1499
    for (int v1211 = 0; v1211 < 4; v1211 += 1) {	// L1500
      for (int v1212 = 0; v1212 < 4; v1212 += 2) {	// L1501
        #pragma HLS pipeline II=1
        float v1213 = v1205[(v1210 + (v1207 * 8))][(v1211 + (v1208 * 4))][(v1212 + (v1209 * 4))];	// L1502
        v1206[v1210][v1211][v1212] = v1213;	// L1503
        float v1214 = v1205[(v1210 + (v1207 * 8))][(v1211 + (v1208 * 4))][((v1212 + (v1209 * 4)) + 1)];	// L1504
        v1206[v1210][v1211][(v1212 + 1)] = v1214;	// L1505
      }
    }
  }
}

void forward_node82(
  float v1215[256][8][8],
  float v1216[8][4][4],
  int v1217,
  int v1218,
  int v1219
) {	// L1511
  #pragma HLS inline
  #pragma HLS array_partition variable=v1215 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1216 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1216 core=ram_t2p_bram

  for (int v1220 = 0; v1220 < 8; v1220 += 1) {	// L1512
    for (int v1221 = 0; v1221 < 4; v1221 += 1) {	// L1513
      for (int v1222 = 0; v1222 < 4; v1222 += 2) {	// L1514
        #pragma HLS pipeline II=1
        float v1223 = v1215[(v1220 + (v1217 * 8))][(v1221 + (v1218 * 4))][(v1222 + (v1219 * 4))];	// L1515
        v1216[v1220][v1221][v1222] = v1223;	// L1516
        float v1224 = v1215[(v1220 + (v1217 * 8))][(v1221 + (v1218 * 4))][((v1222 + (v1219 * 4)) + 1)];	// L1517
        v1216[v1220][v1221][(v1222 + 1)] = v1224;	// L1518
      }
    }
  }
}

void forward_node76(
  float v1225[256][128][3][3],
  float v1226[256][8][8],
  float v1227[128][16][16],
  float v1228[256][8][8],
  float v1229[256][8][8]
) {	// L1524
  #pragma HLS array_partition variable=v1226 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1227 cyclic factor=4 dim=3

  #pragma HLS array_partition variable=v1228 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1229 cyclic factor=2 dim=3

  for (int v1230 = 0; v1230 < 18432; v1230 += 1) {	// L1525
    #pragma HLS dataflow
    int v1231 = (v1230 % 2);	// L1526
    int v1232 = ((v1230 / 2) % 2);	// L1527
    int v1233 = (((v1230 / 2) / 2) % 32);	// L1528
    int v1234 = ((((v1230 / 2) / 2) / 32) % 3);	// L1529
    int v1235 = (((((v1230 / 2) / 2) / 32) / 3) % 3);	// L1530
    int v1236 = (((((v1230 / 2) / 2) / 32) / 3) / 3);	// L1531
    float v1237[8][8];	// L1532
    #pragma HLS resource variable=v1237 core=ram_t2p_bram

    float v1238[8][4][4];	// L1533
    #pragma HLS array_partition variable=v1238 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1238 core=ram_t2p_bram

    float v1239[8][4][4];	// L1534
    #pragma HLS array_partition variable=v1239 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1239 core=ram_t2p_bram

    float v1240[8][4][4];	// L1535
    #pragma HLS array_partition variable=v1240 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1240 core=ram_t2p_bram

    forward_node82(v1226, v1240, v1233, v1232, v1231);	// L1536
    forward_node81(v1228, v1239, v1233, v1232, v1231);	// L1537
    forward_node80(v1227, v1238, v1236, v1235, v1232, v1234, v1231);	// L1538
    forward_node79(v1225, v1237, v1235, v1234, v1233, v1236);	// L1539
    float v1241[8][4][4];	// L1540
    #pragma HLS array_partition variable=v1241 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1241 core=ram_t2p_bram

    forward_node78(v1238, v1240, v1237, v1239, v1241, v1236, v1235, v1234);	// L1541
    forward_node77(v1241, v1229, v1233, v1232, v1231);	// L1542
  }
}

void forward_node84(
  float v1242[8][8][8],
  float v1243[128][16][16],
  int v1244,
  int v1245,
  int v1246
) {	// L1546
  #pragma HLS inline
  #pragma HLS array_partition variable=v1242 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1242 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1242 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1243 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1243 cyclic factor=2 dim=3

  for (int v1247 = 0; v1247 < 8; v1247 += 1) {	// L1547
    for (int v1248 = 0; v1248 < 8; v1248 += 2) {	// L1548
      for (int v1249 = 0; v1249 < 8; v1249 += 2) {	// L1549
        #pragma HLS pipeline II=1
        float v1250 = v1242[v1247][v1248][v1249];	// L1550
        v1243[(v1247 + (v1244 * 8))][(v1248 + (v1245 * 8))][(v1249 + (v1246 * 8))] = v1250;	// L1551
        float v1251 = v1242[v1247][v1248][(v1249 + 1)];	// L1552
        v1243[(v1247 + (v1244 * 8))][(v1248 + (v1245 * 8))][((v1249 + (v1246 * 8)) + 1)] = v1251;	// L1553
        float v1252 = v1242[v1247][(v1248 + 1)][v1249];	// L1554
        v1243[(v1247 + (v1244 * 8))][((v1248 + (v1245 * 8)) + 1)][(v1249 + (v1246 * 8))] = v1252;	// L1555
        float v1253 = v1242[v1247][(v1248 + 1)][(v1249 + 1)];	// L1556
        v1243[(v1247 + (v1244 * 8))][((v1248 + (v1245 * 8)) + 1)][((v1249 + (v1246 * 8)) + 1)] = v1253;	// L1557
      }
    }
  }
}

void forward_node85(
  float v1254[8][8][8],
  float v1255[128][16][16],
  int v1256,
  int v1257,
  int v1258
) {	// L1563
  #pragma HLS inline
  #pragma HLS array_partition variable=v1254 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1254 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1254 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1255 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1255 cyclic factor=2 dim=3

  for (int v1259 = 0; v1259 < 8; v1259 += 1) {	// L1564
    for (int v1260 = 0; v1260 < 8; v1260 += 2) {	// L1565
      for (int v1261 = 0; v1261 < 8; v1261 += 2) {	// L1566
        #pragma HLS pipeline II=1
        float v1262 = v1254[v1259][v1260][v1261];	// L1567
        v1255[(v1259 + (v1256 * 8))][(v1260 + (v1257 * 8))][(v1261 + (v1258 * 8))] = v1262;	// L1568
        float v1263 = v1254[v1259][v1260][(v1261 + 1)];	// L1569
        v1255[(v1259 + (v1256 * 8))][(v1260 + (v1257 * 8))][((v1261 + (v1258 * 8)) + 1)] = v1263;	// L1570
        float v1264 = v1254[v1259][(v1260 + 1)][v1261];	// L1571
        v1255[(v1259 + (v1256 * 8))][((v1260 + (v1257 * 8)) + 1)][(v1261 + (v1258 * 8))] = v1264;	// L1572
        float v1265 = v1254[v1259][(v1260 + 1)][(v1261 + 1)];	// L1573
        v1255[(v1259 + (v1256 * 8))][((v1260 + (v1257 * 8)) + 1)][((v1261 + (v1258 * 8)) + 1)] = v1265;	// L1574
      }
    }
  }
}

void forward_node86(
  float v1266[8][8][8],
  float v1267[8][8][8],
  float v1268[8][8],
  float v1269[8][8][8],
  float v1270[8][8][8],
  float v1271[8][8][8],
  int v1272,
  int v1273,
  int v1274
) {	// L1580
  #pragma HLS inline
  #pragma HLS array_partition variable=v1266 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1266 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1266 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1267 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1267 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1267 core=ram_t2p_bram

  #pragma HLS resource variable=v1268 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1269 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1269 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1269 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1270 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1270 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1270 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1271 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1271 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1271 core=ram_t2p_bram

  for (int v1275 = 0; v1275 < 8; v1275 += 1) {	// L1582
    for (int v1276 = 0; v1276 < 8; v1276 += 1) {	// L1583
      for (int v1277 = 0; v1277 < 8; v1277 += 2) {	// L1584
        for (int v1278 = 0; v1278 < 8; v1278 += 2) {	// L1585
          #pragma HLS pipeline II=1
          float v1279 = v1266[v1275][v1277][v1278];	// L1586
          float v1280 = v1268[v1276][v1275];	// L1587
          float v1281 = v1269[v1276][v1277][v1278];	// L1588
          float v1282 = v1271[v1276][v1277][v1278];	// L1589
          float v1283 = (v1275 == 0) ? v1281 : v1282;	// L1590
          float v1284 = v1279 * v1280;	// L1591
          float v1285 = v1283 + v1284;	// L1592
          v1271[v1276][v1277][v1278] = v1285;	// L1593
          float v1286 = v1267[v1276][v1277][v1278];	// L1594
          float v1287 = v1285 + v1286;	// L1595
          bool v1288 = v1287 > (float)0.000000;	// L1596
          float v1289 = v1288 ? v1287 : (float)0.000000;	// L1597
          if ((((-v1275) + (v1272 * -8)) + 127) == 0 && ((-v1274) + 2) == 0 && ((-v1273) + 2) == 0) {	// L1598
            v1270[v1276][v1277][v1278] = v1289;	// L1599
          }
          float v1290 = v1266[v1275][v1277][(v1278 + 1)];	// L1601
          float v1291 = v1269[v1276][v1277][(v1278 + 1)];	// L1602
          float v1292 = v1271[v1276][v1277][(v1278 + 1)];	// L1603
          float v1293 = (v1275 == 0) ? v1291 : v1292;	// L1604
          float v1294 = v1290 * v1280;	// L1605
          float v1295 = v1293 + v1294;	// L1606
          v1271[v1276][v1277][(v1278 + 1)] = v1295;	// L1607
          float v1296 = v1267[v1276][v1277][(v1278 + 1)];	// L1608
          float v1297 = v1295 + v1296;	// L1609
          bool v1298 = v1297 > (float)0.000000;	// L1610
          float v1299 = v1298 ? v1297 : (float)0.000000;	// L1611
          if ((((-v1275) + (v1272 * -8)) + 127) == 0 && ((-v1274) + 2) == 0 && ((-v1273) + 2) == 0) {	// L1612
            v1270[v1276][v1277][(v1278 + 1)] = v1299;	// L1613
          }
          float v1300 = v1266[v1275][(v1277 + 1)][v1278];	// L1615
          float v1301 = v1269[v1276][(v1277 + 1)][v1278];	// L1616
          float v1302 = v1271[v1276][(v1277 + 1)][v1278];	// L1617
          float v1303 = (v1275 == 0) ? v1301 : v1302;	// L1618
          float v1304 = v1300 * v1280;	// L1619
          float v1305 = v1303 + v1304;	// L1620
          v1271[v1276][(v1277 + 1)][v1278] = v1305;	// L1621
          float v1306 = v1267[v1276][(v1277 + 1)][v1278];	// L1622
          float v1307 = v1305 + v1306;	// L1623
          bool v1308 = v1307 > (float)0.000000;	// L1624
          float v1309 = v1308 ? v1307 : (float)0.000000;	// L1625
          if ((((-v1275) + (v1272 * -8)) + 127) == 0 && ((-v1274) + 2) == 0 && ((-v1273) + 2) == 0) {	// L1626
            v1270[v1276][(v1277 + 1)][v1278] = v1309;	// L1627
          }
          float v1310 = v1266[v1275][(v1277 + 1)][(v1278 + 1)];	// L1629
          float v1311 = v1269[v1276][(v1277 + 1)][(v1278 + 1)];	// L1630
          float v1312 = v1271[v1276][(v1277 + 1)][(v1278 + 1)];	// L1631
          float v1313 = (v1275 == 0) ? v1311 : v1312;	// L1632
          float v1314 = v1310 * v1280;	// L1633
          float v1315 = v1313 + v1314;	// L1634
          v1271[v1276][(v1277 + 1)][(v1278 + 1)] = v1315;	// L1635
          float v1316 = v1267[v1276][(v1277 + 1)][(v1278 + 1)];	// L1636
          float v1317 = v1315 + v1316;	// L1637
          bool v1318 = v1317 > (float)0.000000;	// L1638
          float v1319 = v1318 ? v1317 : (float)0.000000;	// L1639
          if ((((-v1275) + (v1272 * -8)) + 127) == 0 && ((-v1274) + 2) == 0 && ((-v1273) + 2) == 0) {	// L1640
            v1270[v1276][(v1277 + 1)][(v1278 + 1)] = v1319;	// L1641
          }
        }
      }
    }
  }
}

void forward_node87(
  float v1320[128][16][16],
  float v1321[8][8][8],
  int v1322,
  int v1323,
  int v1324
) {	// L1649
  #pragma HLS inline
  #pragma HLS array_partition variable=v1320 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1320 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1321 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1321 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1321 core=ram_t2p_bram

  for (int v1325 = 0; v1325 < 8; v1325 += 1) {	// L1650
    for (int v1326 = 0; v1326 < 8; v1326 += 2) {	// L1651
      for (int v1327 = 0; v1327 < 8; v1327 += 2) {	// L1652
        #pragma HLS pipeline II=1
        float v1328 = v1320[(v1325 + (v1322 * 8))][(v1326 + (v1323 * 8))][(v1327 + (v1324 * 8))];	// L1653
        v1321[v1325][v1326][v1327] = v1328;	// L1654
        float v1329 = v1320[(v1325 + (v1322 * 8))][(v1326 + (v1323 * 8))][((v1327 + (v1324 * 8)) + 1)];	// L1655
        v1321[v1325][v1326][(v1327 + 1)] = v1329;	// L1656
        float v1330 = v1320[(v1325 + (v1322 * 8))][((v1326 + (v1323 * 8)) + 1)][(v1327 + (v1324 * 8))];	// L1657
        v1321[v1325][(v1326 + 1)][v1327] = v1330;	// L1658
        float v1331 = v1320[(v1325 + (v1322 * 8))][((v1326 + (v1323 * 8)) + 1)][((v1327 + (v1324 * 8)) + 1)];	// L1659
        v1321[v1325][(v1326 + 1)][(v1327 + 1)] = v1331;	// L1660
      }
    }
  }
}

void forward_node88(
  float v1332[128][16][16],
  float v1333[8][8][8],
  int v1334,
  int v1335,
  int v1336
) {	// L1666
  #pragma HLS inline
  #pragma HLS array_partition variable=v1332 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1332 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1333 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1333 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1333 core=ram_t2p_bram

  for (int v1337 = 0; v1337 < 8; v1337 += 1) {	// L1667
    for (int v1338 = 0; v1338 < 8; v1338 += 2) {	// L1668
      for (int v1339 = 0; v1339 < 8; v1339 += 2) {	// L1669
        #pragma HLS pipeline II=1
        float v1340 = v1332[(v1337 + (v1334 * 8))][(v1338 + (v1335 * 8))][(v1339 + (v1336 * 8))];	// L1670
        v1333[v1337][v1338][v1339] = v1340;	// L1671
        float v1341 = v1332[(v1337 + (v1334 * 8))][(v1338 + (v1335 * 8))][((v1339 + (v1336 * 8)) + 1)];	// L1672
        v1333[v1337][v1338][(v1339 + 1)] = v1341;	// L1673
        float v1342 = v1332[(v1337 + (v1334 * 8))][((v1338 + (v1335 * 8)) + 1)][(v1339 + (v1336 * 8))];	// L1674
        v1333[v1337][(v1338 + 1)][v1339] = v1342;	// L1675
        float v1343 = v1332[(v1337 + (v1334 * 8))][((v1338 + (v1335 * 8)) + 1)][((v1339 + (v1336 * 8)) + 1)];	// L1676
        v1333[v1337][(v1338 + 1)][(v1339 + 1)] = v1343;	// L1677
      }
    }
  }
}

void forward_node89(
  float v1344[128][128][3][3],
  float v1345[8][8],
  int v1346,
  int v1347,
  int v1348,
  int v1349
) {	// L1683
  #pragma HLS inline
  #pragma HLS resource variable=v1345 core=ram_t2p_bram

  for (int v1350 = 0; v1350 < 8; v1350 += 1) {	// L1684
    for (int v1351 = 0; v1351 < 8; v1351 += 1) {	// L1685
      #pragma HLS pipeline II=1
      float v1352 = v1344[(v1350 + (v1348 * 8))][(v1351 + (v1349 * 8))][v1346][v1347];	// L1686
      v1345[v1350][v1351] = v1352;	// L1687
    }
  }
}

void forward_node90(
  float v1353[128][16][16],
  float v1354[8][8][8],
  int v1355,
  int v1356,
  int v1357,
  int v1358,
  int v1359
) {	// L1692
  #pragma HLS inline
  #pragma HLS array_partition variable=v1353 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1353 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1354 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1354 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1354 core=ram_t2p_bram

  for (int v1360 = 0; v1360 < 8; v1360 += 1) {	// L1693
    for (int v1361 = 0; v1361 < 8; v1361 += 2) {	// L1694
      for (int v1362 = 0; v1362 < 8; v1362 += 2) {	// L1695
        #pragma HLS pipeline II=1
        float v1363 = v1353[(v1360 + (v1355 * 8))][(((v1361 + v1356) + (v1357 * 8)) - 1)][(((v1362 + v1358) + (v1359 * 8)) - 1)];	// L1696
        v1354[v1360][v1361][v1362] = v1363;	// L1697
        float v1364 = v1353[(v1360 + (v1355 * 8))][(((v1361 + v1356) + (v1357 * 8)) - 1)][((v1362 + v1358) + (v1359 * 8))];	// L1698
        v1354[v1360][v1361][(v1362 + 1)] = v1364;	// L1699
        float v1365 = v1353[(v1360 + (v1355 * 8))][((v1361 + v1356) + (v1357 * 8))][(((v1362 + v1358) + (v1359 * 8)) - 1)];	// L1700
        v1354[v1360][(v1361 + 1)][v1362] = v1365;	// L1701
        float v1366 = v1353[(v1360 + (v1355 * 8))][((v1361 + v1356) + (v1357 * 8))][((v1362 + v1358) + (v1359 * 8))];	// L1702
        v1354[v1360][(v1361 + 1)][(v1362 + 1)] = v1366;	// L1703
      }
    }
  }
}

void forward_node83(
  float v1367[128][16][16],
  float v1368[128][16][16],
  float v1369[128][128][3][3],
  float v1370[128][16][16],
  float v1371[128][16][16],
  float v1372[128][16][16]
) {	// L1709
  #pragma HLS array_partition variable=v1367 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1367 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1368 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1368 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1370 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1370 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1371 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1371 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1372 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1372 cyclic factor=2 dim=3

  for (int v1373 = 0; v1373 < 9216; v1373 += 1) {	// L1710
    #pragma HLS dataflow
    int v1374 = (v1373 % 2);	// L1711
    int v1375 = ((v1373 / 2) % 2);	// L1712
    int v1376 = (((v1373 / 2) / 2) % 16);	// L1713
    int v1377 = ((((v1373 / 2) / 2) / 16) % 3);	// L1714
    int v1378 = (((((v1373 / 2) / 2) / 16) / 3) % 3);	// L1715
    int v1379 = (((((v1373 / 2) / 2) / 16) / 3) / 3);	// L1716
    float v1380[8][8][8];	// L1717
    #pragma HLS array_partition variable=v1380 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1380 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1380 core=ram_t2p_bram

    float v1381[8][8][8];	// L1718
    #pragma HLS array_partition variable=v1381 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1381 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1381 core=ram_t2p_bram

    float v1382[8][8][8];	// L1719
    #pragma HLS array_partition variable=v1382 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1382 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1382 core=ram_t2p_bram

    float v1383[8][8];	// L1720
    #pragma HLS resource variable=v1383 core=ram_t2p_bram

    float v1384[8][8][8];	// L1721
    #pragma HLS array_partition variable=v1384 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1384 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1384 core=ram_t2p_bram

    forward_node90(v1368, v1384, v1379, v1378, v1375, v1377, v1374);	// L1722
    forward_node89(v1369, v1383, v1378, v1377, v1376, v1379);	// L1723
    forward_node88(v1370, v1382, v1376, v1375, v1374);	// L1724
    forward_node87(v1367, v1381, v1376, v1375, v1374);	// L1725
    float v1385[8][8][8];	// L1726
    #pragma HLS array_partition variable=v1385 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1385 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1385 core=ram_t2p_bram

    forward_node86(v1384, v1381, v1383, v1382, v1380, v1385, v1379, v1377, v1378);	// L1727
    forward_node85(v1385, v1371, v1376, v1375, v1374);	// L1728
    forward_node84(v1380, v1372, v1376, v1375, v1374);	// L1729
  }
}

void forward_node92(
  float v1386[8][8][8],
  float v1387[128][16][16],
  int v1388,
  int v1389,
  int v1390
) {	// L1733
  #pragma HLS inline
  #pragma HLS array_partition variable=v1386 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1386 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1386 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1387 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1387 cyclic factor=2 dim=3

  for (int v1391 = 0; v1391 < 8; v1391 += 1) {	// L1734
    for (int v1392 = 0; v1392 < 8; v1392 += 2) {	// L1735
      for (int v1393 = 0; v1393 < 8; v1393 += 2) {	// L1736
        #pragma HLS pipeline II=1
        float v1394 = v1386[v1391][v1392][v1393];	// L1737
        v1387[(v1391 + (v1388 * 8))][(v1392 + (v1389 * 8))][(v1393 + (v1390 * 8))] = v1394;	// L1738
        float v1395 = v1386[v1391][v1392][(v1393 + 1)];	// L1739
        v1387[(v1391 + (v1388 * 8))][(v1392 + (v1389 * 8))][((v1393 + (v1390 * 8)) + 1)] = v1395;	// L1740
        float v1396 = v1386[v1391][(v1392 + 1)][v1393];	// L1741
        v1387[(v1391 + (v1388 * 8))][((v1392 + (v1389 * 8)) + 1)][(v1393 + (v1390 * 8))] = v1396;	// L1742
        float v1397 = v1386[v1391][(v1392 + 1)][(v1393 + 1)];	// L1743
        v1387[(v1391 + (v1388 * 8))][((v1392 + (v1389 * 8)) + 1)][((v1393 + (v1390 * 8)) + 1)] = v1397;	// L1744
      }
    }
  }
}

void forward_node93(
  float v1398[8][8],
  float v1399[8][8][8],
  float v1400[8][8][8],
  float v1401[8][8][8],
  float v1402[8][8][8],
  int v1403,
  int v1404,
  int v1405
) {	// L1750
  #pragma HLS inline
  #pragma HLS resource variable=v1398 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1399 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1399 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1399 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1400 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1400 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1400 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1401 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1401 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1401 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1402 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1402 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1402 core=ram_t2p_bram

  for (int v1406 = 0; v1406 < 8; v1406 += 1) {	// L1752
    for (int v1407 = 0; v1407 < 8; v1407 += 1) {	// L1753
      for (int v1408 = 0; v1408 < 8; v1408 += 2) {	// L1754
        for (int v1409 = 0; v1409 < 8; v1409 += 2) {	// L1755
          #pragma HLS pipeline II=1
          float v1410 = v1400[v1407][v1408][v1409];	// L1756
          float v1411 = v1401[v1407][v1408][v1409];	// L1757
          float v1412 = v1402[v1407][v1408][v1409];	// L1758
          float v1413 = (v1406 == 0) ? v1411 : v1412;	// L1759
          float v1414 = ((v1406 + (v1404 * 8)) == 0 && v1405 == 0 && v1403 == 0) ? v1410 : v1413;	// L1760
          float v1415 = v1399[v1406][v1408][v1409];	// L1761
          float v1416 = v1398[v1407][v1406];	// L1762
          float v1417 = v1415 * v1416;	// L1763
          float v1418 = v1414 + v1417;	// L1764
          bool v1419 = v1418 > (float)0.000000;	// L1765
          float v1420 = v1419 ? v1418 : (float)0.000000;	// L1766
          float v1421 = ((((-v1406) + (v1404 * -8)) + 127) == 0 && ((-v1405) + 2) == 0 && ((-v1403) + 2) == 0) ? v1420 : v1418;	// L1767
          v1402[v1407][v1408][v1409] = v1421;	// L1768
          float v1422 = v1400[v1407][v1408][(v1409 + 1)];	// L1769
          float v1423 = v1401[v1407][v1408][(v1409 + 1)];	// L1770
          float v1424 = v1402[v1407][v1408][(v1409 + 1)];	// L1771
          float v1425 = (v1406 == 0) ? v1423 : v1424;	// L1772
          float v1426 = ((v1406 + (v1404 * 8)) == 0 && v1405 == 0 && v1403 == 0) ? v1422 : v1425;	// L1773
          float v1427 = v1399[v1406][v1408][(v1409 + 1)];	// L1774
          float v1428 = v1427 * v1416;	// L1775
          float v1429 = v1426 + v1428;	// L1776
          bool v1430 = v1429 > (float)0.000000;	// L1777
          float v1431 = v1430 ? v1429 : (float)0.000000;	// L1778
          float v1432 = ((((-v1406) + (v1404 * -8)) + 127) == 0 && ((-v1405) + 2) == 0 && ((-v1403) + 2) == 0) ? v1431 : v1429;	// L1779
          v1402[v1407][v1408][(v1409 + 1)] = v1432;	// L1780
          float v1433 = v1400[v1407][(v1408 + 1)][v1409];	// L1781
          float v1434 = v1401[v1407][(v1408 + 1)][v1409];	// L1782
          float v1435 = v1402[v1407][(v1408 + 1)][v1409];	// L1783
          float v1436 = (v1406 == 0) ? v1434 : v1435;	// L1784
          float v1437 = ((v1406 + (v1404 * 8)) == 0 && v1405 == 0 && v1403 == 0) ? v1433 : v1436;	// L1785
          float v1438 = v1399[v1406][(v1408 + 1)][v1409];	// L1786
          float v1439 = v1438 * v1416;	// L1787
          float v1440 = v1437 + v1439;	// L1788
          bool v1441 = v1440 > (float)0.000000;	// L1789
          float v1442 = v1441 ? v1440 : (float)0.000000;	// L1790
          float v1443 = ((((-v1406) + (v1404 * -8)) + 127) == 0 && ((-v1405) + 2) == 0 && ((-v1403) + 2) == 0) ? v1442 : v1440;	// L1791
          v1402[v1407][(v1408 + 1)][v1409] = v1443;	// L1792
          float v1444 = v1400[v1407][(v1408 + 1)][(v1409 + 1)];	// L1793
          float v1445 = v1401[v1407][(v1408 + 1)][(v1409 + 1)];	// L1794
          float v1446 = v1402[v1407][(v1408 + 1)][(v1409 + 1)];	// L1795
          float v1447 = (v1406 == 0) ? v1445 : v1446;	// L1796
          float v1448 = ((v1406 + (v1404 * 8)) == 0 && v1405 == 0 && v1403 == 0) ? v1444 : v1447;	// L1797
          float v1449 = v1399[v1406][(v1408 + 1)][(v1409 + 1)];	// L1798
          float v1450 = v1449 * v1416;	// L1799
          float v1451 = v1448 + v1450;	// L1800
          bool v1452 = v1451 > (float)0.000000;	// L1801
          float v1453 = v1452 ? v1451 : (float)0.000000;	// L1802
          float v1454 = ((((-v1406) + (v1404 * -8)) + 127) == 0 && ((-v1405) + 2) == 0 && ((-v1403) + 2) == 0) ? v1453 : v1451;	// L1803
          v1402[v1407][(v1408 + 1)][(v1409 + 1)] = v1454;	// L1804
        }
      }
    }
  }
}

void forward_node94(
  float v1455[128][128][3][3],
  float v1456[8][8],
  int v1457,
  int v1458,
  int v1459,
  int v1460
) {	// L1811
  #pragma HLS inline
  #pragma HLS resource variable=v1456 core=ram_t2p_bram

  for (int v1461 = 0; v1461 < 8; v1461 += 1) {	// L1812
    for (int v1462 = 0; v1462 < 8; v1462 += 1) {	// L1813
      #pragma HLS pipeline II=1
      float v1463 = v1455[(v1461 + (v1459 * 8))][(v1462 + (v1460 * 8))][v1457][v1458];	// L1814
      v1456[v1461][v1462] = v1463;	// L1815
    }
  }
}

void forward_node95(
  float v1464[128][16][16],
  float v1465[8][8][8],
  int v1466,
  int v1467,
  int v1468,
  int v1469,
  int v1470
) {	// L1820
  #pragma HLS inline
  #pragma HLS array_partition variable=v1464 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1464 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1465 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1465 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1465 core=ram_t2p_bram

  for (int v1471 = 0; v1471 < 8; v1471 += 1) {	// L1821
    for (int v1472 = 0; v1472 < 8; v1472 += 2) {	// L1822
      for (int v1473 = 0; v1473 < 8; v1473 += 2) {	// L1823
        #pragma HLS pipeline II=1
        float v1474 = v1464[(v1471 + (v1466 * 8))][(((v1472 + v1467) + (v1468 * 8)) - 1)][(((v1473 + v1469) + (v1470 * 8)) - 1)];	// L1824
        v1465[v1471][v1472][v1473] = v1474;	// L1825
        float v1475 = v1464[(v1471 + (v1466 * 8))][(((v1472 + v1467) + (v1468 * 8)) - 1)][((v1473 + v1469) + (v1470 * 8))];	// L1826
        v1465[v1471][v1472][(v1473 + 1)] = v1475;	// L1827
        float v1476 = v1464[(v1471 + (v1466 * 8))][((v1472 + v1467) + (v1468 * 8))][(((v1473 + v1469) + (v1470 * 8)) - 1)];	// L1828
        v1465[v1471][(v1472 + 1)][v1473] = v1476;	// L1829
        float v1477 = v1464[(v1471 + (v1466 * 8))][((v1472 + v1467) + (v1468 * 8))][((v1473 + v1469) + (v1470 * 8))];	// L1830
        v1465[v1471][(v1472 + 1)][(v1473 + 1)] = v1477;	// L1831
      }
    }
  }
}

void forward_node96(
  float v1478[128][16][16],
  float v1479[8][8][8],
  int v1480,
  int v1481,
  int v1482
) {	// L1837
  #pragma HLS inline
  #pragma HLS array_partition variable=v1478 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1478 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1479 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1479 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1479 core=ram_t2p_bram

  for (int v1483 = 0; v1483 < 8; v1483 += 1) {	// L1838
    for (int v1484 = 0; v1484 < 8; v1484 += 2) {	// L1839
      for (int v1485 = 0; v1485 < 8; v1485 += 2) {	// L1840
        #pragma HLS pipeline II=1
        float v1486 = v1478[(v1483 + (v1480 * 8))][(v1484 + (v1481 * 8))][(v1485 + (v1482 * 8))];	// L1841
        v1479[v1483][v1484][v1485] = v1486;	// L1842
        float v1487 = v1478[(v1483 + (v1480 * 8))][(v1484 + (v1481 * 8))][((v1485 + (v1482 * 8)) + 1)];	// L1843
        v1479[v1483][v1484][(v1485 + 1)] = v1487;	// L1844
        float v1488 = v1478[(v1483 + (v1480 * 8))][((v1484 + (v1481 * 8)) + 1)][(v1485 + (v1482 * 8))];	// L1845
        v1479[v1483][(v1484 + 1)][v1485] = v1488;	// L1846
        float v1489 = v1478[(v1483 + (v1480 * 8))][((v1484 + (v1481 * 8)) + 1)][((v1485 + (v1482 * 8)) + 1)];	// L1847
        v1479[v1483][(v1484 + 1)][(v1485 + 1)] = v1489;	// L1848
      }
    }
  }
}

void forward_node97(
  float v1490[128][16][16],
  float v1491[8][8][8],
  int v1492,
  int v1493,
  int v1494
) {	// L1854
  #pragma HLS inline
  #pragma HLS array_partition variable=v1490 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1490 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1491 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1491 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1491 core=ram_t2p_bram

  for (int v1495 = 0; v1495 < 8; v1495 += 1) {	// L1855
    for (int v1496 = 0; v1496 < 8; v1496 += 2) {	// L1856
      for (int v1497 = 0; v1497 < 8; v1497 += 2) {	// L1857
        #pragma HLS pipeline II=1
        float v1498 = v1490[(v1495 + (v1492 * 8))][(v1496 + (v1493 * 8))][(v1497 + (v1494 * 8))];	// L1858
        v1491[v1495][v1496][v1497] = v1498;	// L1859
        float v1499 = v1490[(v1495 + (v1492 * 8))][(v1496 + (v1493 * 8))][((v1497 + (v1494 * 8)) + 1)];	// L1860
        v1491[v1495][v1496][(v1497 + 1)] = v1499;	// L1861
        float v1500 = v1490[(v1495 + (v1492 * 8))][((v1496 + (v1493 * 8)) + 1)][(v1497 + (v1494 * 8))];	// L1862
        v1491[v1495][(v1496 + 1)][v1497] = v1500;	// L1863
        float v1501 = v1490[(v1495 + (v1492 * 8))][((v1496 + (v1493 * 8)) + 1)][((v1497 + (v1494 * 8)) + 1)];	// L1864
        v1491[v1495][(v1496 + 1)][(v1497 + 1)] = v1501;	// L1865
      }
    }
  }
}

void forward_node91(
  float v1502[128][16][16],
  float v1503[128][128][3][3],
  float v1504[128][16][16],
  float v1505[128][16][16],
  float v1506[128][16][16]
) {	// L1871
  #pragma HLS array_partition variable=v1502 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1502 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1504 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1504 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1505 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1505 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1506 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1506 cyclic factor=2 dim=3

  for (int v1507 = 0; v1507 < 9216; v1507 += 1) {	// L1872
    #pragma HLS dataflow
    int v1508 = (v1507 % 2);	// L1873
    int v1509 = ((v1507 / 2) % 2);	// L1874
    int v1510 = (((v1507 / 2) / 2) % 16);	// L1875
    int v1511 = ((((v1507 / 2) / 2) / 16) % 3);	// L1876
    int v1512 = (((((v1507 / 2) / 2) / 16) / 3) % 3);	// L1877
    int v1513 = (((((v1507 / 2) / 2) / 16) / 3) / 3);	// L1878
    float v1514[8][8];	// L1879
    #pragma HLS resource variable=v1514 core=ram_t2p_bram

    float v1515[8][8][8];	// L1880
    #pragma HLS array_partition variable=v1515 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1515 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1515 core=ram_t2p_bram

    float v1516[8][8][8];	// L1881
    #pragma HLS array_partition variable=v1516 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1516 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1516 core=ram_t2p_bram

    float v1517[8][8][8];	// L1882
    #pragma HLS array_partition variable=v1517 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1517 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1517 core=ram_t2p_bram

    forward_node97(v1504, v1517, v1510, v1509, v1508);	// L1883
    forward_node96(v1505, v1516, v1510, v1509, v1508);	// L1884
    forward_node95(v1502, v1515, v1513, v1512, v1509, v1511, v1508);	// L1885
    forward_node94(v1503, v1514, v1512, v1511, v1510, v1513);	// L1886
    float v1518[8][8][8];	// L1887
    #pragma HLS array_partition variable=v1518 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1518 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1518 core=ram_t2p_bram

    forward_node93(v1514, v1515, v1517, v1516, v1518, v1511, v1513, v1512);	// L1888
    forward_node92(v1518, v1506, v1510, v1509, v1508);	// L1889
  }
}

void forward_node99(
  float v1519[8][8][8],
  float v1520[128][16][16],
  int v1521,
  int v1522,
  int v1523
) {	// L1893
  #pragma HLS inline
  #pragma HLS resource variable=v1519 core=ram_t2p_bram

  for (int v1524 = 0; v1524 < 8; v1524 += 1) {	// L1894
    for (int v1525 = 0; v1525 < 8; v1525 += 1) {	// L1895
      for (int v1526 = 0; v1526 < 8; v1526 += 1) {	// L1896
        #pragma HLS pipeline II=1
        float v1527 = v1519[v1524][v1525][v1526];	// L1897
        v1520[(v1524 + (v1521 * 8))][(v1525 + (v1522 * 8))][(v1526 + (v1523 * 8))] = v1527;	// L1898
      }
    }
  }
}

void forward_node100(
  float v1528[8][8][8],
  float v1529[128][16][16],
  int v1530,
  int v1531,
  int v1532
) {	// L1904
  #pragma HLS inline
  #pragma HLS resource variable=v1528 core=ram_t2p_bram

  for (int v1533 = 0; v1533 < 8; v1533 += 1) {	// L1905
    for (int v1534 = 0; v1534 < 8; v1534 += 1) {	// L1906
      for (int v1535 = 0; v1535 < 8; v1535 += 1) {	// L1907
        #pragma HLS pipeline II=1
        float v1536 = v1528[v1533][v1534][v1535];	// L1908
        v1529[(v1533 + (v1530 * 8))][(v1534 + (v1531 * 8))][(v1535 + (v1532 * 8))] = v1536;	// L1909
      }
    }
  }
}

void forward_node101(
  float v1537[8][8][8],
  float v1538[8][8][8],
  float v1539[8][8],
  float v1540[8][8][8],
  float v1541[8][8][8],
  float v1542[8][8][8],
  float v1543[8][8][8],
  int v1544
) {	// L1915
  #pragma HLS inline
  #pragma HLS resource variable=v1537 core=ram_t2p_bram

  #pragma HLS resource variable=v1538 core=ram_t2p_bram

  #pragma HLS resource variable=v1539 core=ram_t2p_bram

  #pragma HLS resource variable=v1540 core=ram_t2p_bram

  #pragma HLS resource variable=v1541 core=ram_t2p_bram

  #pragma HLS resource variable=v1542 core=ram_t2p_bram

  #pragma HLS resource variable=v1543 core=ram_t2p_bram

  for (int v1545 = 0; v1545 < 8; v1545 += 1) {	// L1917
    for (int v1546 = 0; v1546 < 8; v1546 += 1) {	// L1918
      for (int v1547 = 0; v1547 < 8; v1547 += 1) {	// L1919
        for (int v1548 = 0; v1548 < 8; v1548 += 1) {	// L1920
          #pragma HLS pipeline II=1
          float v1549 = v1538[v1546][v1547][v1548];	// L1921
          float v1550 = v1541[v1546][v1547][v1548];	// L1922
          float v1551 = v1542[v1546][v1547][v1548];	// L1923
          float v1552 = (v1545 == 0) ? v1550 : v1551;	// L1924
          float v1553 = ((v1545 + (v1544 * 8)) == 0) ? v1549 : v1552;	// L1925
          float v1554 = v1537[v1545][v1547][v1548];	// L1926
          float v1555 = v1539[v1546][v1545];	// L1927
          float v1556 = v1554 * v1555;	// L1928
          float v1557 = v1553 + v1556;	// L1929
          v1542[v1546][v1547][v1548] = v1557;	// L1930
          float v1558 = v1540[v1546][v1547][v1548];	// L1931
          float v1559 = v1558 + v1557;	// L1932
          bool v1560 = v1559 > (float)0.000000;	// L1933
          float v1561 = v1560 ? v1559 : (float)0.000000;	// L1934
          if ((((-v1545) + (v1544 * -8)) + 63) == 0) {	// L1935
            v1543[v1546][v1547][v1548] = v1561;	// L1936
          }
        }
      }
    }
  }
}

void forward_node102(
  float v1562[128][16][16],
  float v1563[8][8][8],
  int v1564,
  int v1565,
  int v1566
) {	// L1944
  #pragma HLS inline
  #pragma HLS resource variable=v1563 core=ram_t2p_bram

  for (int v1567 = 0; v1567 < 8; v1567 += 1) {	// L1945
    for (int v1568 = 0; v1568 < 8; v1568 += 1) {	// L1946
      for (int v1569 = 0; v1569 < 8; v1569 += 1) {	// L1947
        #pragma HLS pipeline II=1
        float v1570 = v1562[(v1567 + (v1564 * 8))][(v1568 + (v1565 * 8))][(v1569 + (v1566 * 8))];	// L1948
        v1563[v1567][v1568][v1569] = v1570;	// L1949
      }
    }
  }
}

void forward_node103(
  float v1571[128][64],
  float v1572[8][8],
  int v1573,
  int v1574
) {	// L1955
  #pragma HLS inline
  #pragma HLS resource variable=v1572 core=ram_t2p_bram

  for (int v1575 = 0; v1575 < 8; v1575 += 1) {	// L1956
    for (int v1576 = 0; v1576 < 8; v1576 += 1) {	// L1957
      #pragma HLS pipeline II=1
      float v1577 = v1571[(v1575 + (v1573 * 8))][(v1576 + (v1574 * 8))];	// L1958
      v1572[v1575][v1576] = v1577;	// L1959
    }
  }
}

void forward_node104(
  float v1578[64][32][32],
  float v1579[8][8][8],
  int v1580,
  int v1581,
  int v1582
) {	// L1964
  #pragma HLS inline
  #pragma HLS resource variable=v1579 core=ram_t2p_bram

  for (int v1583 = 0; v1583 < 8; v1583 += 1) {	// L1965
    for (int v1584 = 0; v1584 < 8; v1584 += 1) {	// L1966
      for (int v1585 = 0; v1585 < 8; v1585 += 1) {	// L1967
        #pragma HLS pipeline II=1
        float v1586 = v1578[(v1583 + (v1580 * 8))][((v1584 * 2) + (v1581 * 16))][((v1585 * 2) + (v1582 * 16))];	// L1968
        v1579[v1583][v1584][v1585] = v1586;	// L1969
      }
    }
  }
}

void forward_node105(
  float v1587[128][16][16],
  float v1588[8][8][8],
  int v1589,
  int v1590,
  int v1591
) {	// L1975
  #pragma HLS inline
  #pragma HLS resource variable=v1588 core=ram_t2p_bram

  for (int v1592 = 0; v1592 < 8; v1592 += 1) {	// L1976
    for (int v1593 = 0; v1593 < 8; v1593 += 1) {	// L1977
      for (int v1594 = 0; v1594 < 8; v1594 += 1) {	// L1978
        #pragma HLS pipeline II=1
        float v1595 = v1587[(v1592 + (v1589 * 8))][(v1593 + (v1590 * 8))][(v1594 + (v1591 * 8))];	// L1979
        v1588[v1592][v1593][v1594] = v1595;	// L1980
      }
    }
  }
}

void forward_node106(
  float v1596[128][16][16],
  float v1597[8][8][8],
  int v1598,
  int v1599,
  int v1600
) {	// L1986
  #pragma HLS inline
  #pragma HLS resource variable=v1597 core=ram_t2p_bram

  for (int v1601 = 0; v1601 < 8; v1601 += 1) {	// L1987
    for (int v1602 = 0; v1602 < 8; v1602 += 1) {	// L1988
      for (int v1603 = 0; v1603 < 8; v1603 += 1) {	// L1989
        #pragma HLS pipeline II=1
        float v1604 = v1596[(v1601 + (v1598 * 8))][(v1602 + (v1599 * 8))][(v1603 + (v1600 * 8))];	// L1990
        v1597[v1601][v1602][v1603] = v1604;	// L1991
      }
    }
  }
}

void forward_node98(
  float v1605[64][32][32],
  float v1606[128][16][16],
  float v1607[128][16][16],
  float v1608[128][64],
  float v1609[128][16][16],
  float v1610[128][16][16],
  float v1611[128][16][16]
) {	// L1997
  for (int v1612 = 0; v1612 < 512; v1612 += 1) {	// L1998
    #pragma HLS dataflow
    int v1613 = (v1612 % 2);	// L1999
    int v1614 = ((v1612 / 2) % 2);	// L2000
    int v1615 = (((v1612 / 2) / 2) % 16);	// L2001
    int v1616 = (((v1612 / 2) / 2) / 16);	// L2002
    float v1617[8][8][8];	// L2003
    #pragma HLS resource variable=v1617 core=ram_t2p_bram

    float v1618[8][8][8];	// L2004
    #pragma HLS resource variable=v1618 core=ram_t2p_bram

    float v1619[8][8];	// L2005
    #pragma HLS resource variable=v1619 core=ram_t2p_bram

    float v1620[8][8][8];	// L2006
    #pragma HLS resource variable=v1620 core=ram_t2p_bram

    float v1621[8][8][8];	// L2007
    #pragma HLS resource variable=v1621 core=ram_t2p_bram

    float v1622[8][8][8];	// L2008
    #pragma HLS resource variable=v1622 core=ram_t2p_bram

    forward_node106(v1607, v1622, v1615, v1614, v1613);	// L2009
    forward_node105(v1609, v1621, v1615, v1614, v1613);	// L2010
    forward_node104(v1605, v1620, v1616, v1614, v1613);	// L2011
    forward_node103(v1608, v1619, v1615, v1616);	// L2012
    forward_node102(v1606, v1618, v1615, v1614, v1613);	// L2013
    float v1623[8][8][8];	// L2014
    #pragma HLS resource variable=v1623 core=ram_t2p_bram

    forward_node101(v1620, v1622, v1619, v1618, v1621, v1623, v1617, v1616);	// L2015
    forward_node100(v1623, v1610, v1615, v1614, v1613);	// L2016
    forward_node99(v1617, v1611, v1615, v1614, v1613);	// L2017
  }
}

void forward_node108(
  float v1624[8][8][8],
  float v1625[128][16][16],
  int v1626,
  int v1627,
  int v1628
) {	// L2021
  #pragma HLS inline
  #pragma HLS array_partition variable=v1624 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1624 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1624 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1625 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1625 cyclic factor=2 dim=3

  for (int v1629 = 0; v1629 < 8; v1629 += 1) {	// L2022
    for (int v1630 = 0; v1630 < 8; v1630 += 2) {	// L2023
      for (int v1631 = 0; v1631 < 8; v1631 += 2) {	// L2024
        #pragma HLS pipeline II=1
        float v1632 = v1624[v1629][v1630][v1631];	// L2025
        v1625[(v1629 + (v1626 * 8))][(v1630 + (v1627 * 8))][(v1631 + (v1628 * 8))] = v1632;	// L2026
        float v1633 = v1624[v1629][v1630][(v1631 + 1)];	// L2027
        v1625[(v1629 + (v1626 * 8))][(v1630 + (v1627 * 8))][((v1631 + (v1628 * 8)) + 1)] = v1633;	// L2028
        float v1634 = v1624[v1629][(v1630 + 1)][v1631];	// L2029
        v1625[(v1629 + (v1626 * 8))][((v1630 + (v1627 * 8)) + 1)][(v1631 + (v1628 * 8))] = v1634;	// L2030
        float v1635 = v1624[v1629][(v1630 + 1)][(v1631 + 1)];	// L2031
        v1625[(v1629 + (v1626 * 8))][((v1630 + (v1627 * 8)) + 1)][((v1631 + (v1628 * 8)) + 1)] = v1635;	// L2032
      }
    }
  }
}

void forward_node109(
  float v1636[8][8][8],
  float v1637[8][8][8],
  float v1638[8][8],
  float v1639[8][8][8],
  float v1640[8][8][8],
  int v1641,
  int v1642,
  int v1643
) {	// L2038
  #pragma HLS inline
  #pragma HLS array_partition variable=v1636 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1636 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1636 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1637 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1637 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1637 core=ram_t2p_bram

  #pragma HLS resource variable=v1638 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1639 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1639 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1639 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1640 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1640 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1640 core=ram_t2p_bram

  for (int v1644 = 0; v1644 < 8; v1644 += 1) {	// L2039
    for (int v1645 = 0; v1645 < 8; v1645 += 1) {	// L2040
      for (int v1646 = 0; v1646 < 8; v1646 += 2) {	// L2041
        for (int v1647 = 0; v1647 < 8; v1647 += 2) {	// L2042
          #pragma HLS pipeline II=1
          float v1648 = v1636[v1645][v1646][v1647];	// L2043
          float v1649 = v1639[v1645][v1646][v1647];	// L2044
          float v1650 = v1640[v1645][v1646][v1647];	// L2045
          float v1651 = (v1644 == 0) ? v1649 : v1650;	// L2046
          float v1652 = ((v1644 + (v1641 * 8)) == 0 && v1642 == 0 && v1643 == 0) ? v1648 : v1651;	// L2047
          float v1653 = v1637[v1644][v1646][v1647];	// L2048
          float v1654 = v1638[v1645][v1644];	// L2049
          float v1655 = v1653 * v1654;	// L2050
          float v1656 = v1652 + v1655;	// L2051
          v1640[v1645][v1646][v1647] = v1656;	// L2052
          float v1657 = v1636[v1645][v1646][(v1647 + 1)];	// L2053
          float v1658 = v1639[v1645][v1646][(v1647 + 1)];	// L2054
          float v1659 = v1640[v1645][v1646][(v1647 + 1)];	// L2055
          float v1660 = (v1644 == 0) ? v1658 : v1659;	// L2056
          float v1661 = ((v1644 + (v1641 * 8)) == 0 && v1642 == 0 && v1643 == 0) ? v1657 : v1660;	// L2057
          float v1662 = v1637[v1644][v1646][(v1647 + 1)];	// L2058
          float v1663 = v1662 * v1654;	// L2059
          float v1664 = v1661 + v1663;	// L2060
          v1640[v1645][v1646][(v1647 + 1)] = v1664;	// L2061
          float v1665 = v1636[v1645][(v1646 + 1)][v1647];	// L2062
          float v1666 = v1639[v1645][(v1646 + 1)][v1647];	// L2063
          float v1667 = v1640[v1645][(v1646 + 1)][v1647];	// L2064
          float v1668 = (v1644 == 0) ? v1666 : v1667;	// L2065
          float v1669 = ((v1644 + (v1641 * 8)) == 0 && v1642 == 0 && v1643 == 0) ? v1665 : v1668;	// L2066
          float v1670 = v1637[v1644][(v1646 + 1)][v1647];	// L2067
          float v1671 = v1670 * v1654;	// L2068
          float v1672 = v1669 + v1671;	// L2069
          v1640[v1645][(v1646 + 1)][v1647] = v1672;	// L2070
          float v1673 = v1636[v1645][(v1646 + 1)][(v1647 + 1)];	// L2071
          float v1674 = v1639[v1645][(v1646 + 1)][(v1647 + 1)];	// L2072
          float v1675 = v1640[v1645][(v1646 + 1)][(v1647 + 1)];	// L2073
          float v1676 = (v1644 == 0) ? v1674 : v1675;	// L2074
          float v1677 = ((v1644 + (v1641 * 8)) == 0 && v1642 == 0 && v1643 == 0) ? v1673 : v1676;	// L2075
          float v1678 = v1637[v1644][(v1646 + 1)][(v1647 + 1)];	// L2076
          float v1679 = v1678 * v1654;	// L2077
          float v1680 = v1677 + v1679;	// L2078
          v1640[v1645][(v1646 + 1)][(v1647 + 1)] = v1680;	// L2079
        }
      }
    }
  }
}

void forward_node110(
  float v1681[128][128][3][3],
  float v1682[8][8],
  int v1683,
  int v1684,
  int v1685,
  int v1686
) {	// L2086
  #pragma HLS inline
  #pragma HLS resource variable=v1682 core=ram_t2p_bram

  for (int v1687 = 0; v1687 < 8; v1687 += 1) {	// L2087
    for (int v1688 = 0; v1688 < 8; v1688 += 1) {	// L2088
      #pragma HLS pipeline II=1
      float v1689 = v1681[(v1687 + (v1685 * 8))][(v1688 + (v1686 * 8))][v1683][v1684];	// L2089
      v1682[v1687][v1688] = v1689;	// L2090
    }
  }
}

void forward_node111(
  float v1690[128][16][16],
  float v1691[8][8][8],
  int v1692,
  int v1693,
  int v1694,
  int v1695,
  int v1696
) {	// L2095
  #pragma HLS inline
  #pragma HLS array_partition variable=v1690 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1690 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1691 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1691 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1691 core=ram_t2p_bram

  for (int v1697 = 0; v1697 < 8; v1697 += 1) {	// L2096
    for (int v1698 = 0; v1698 < 8; v1698 += 2) {	// L2097
      for (int v1699 = 0; v1699 < 8; v1699 += 2) {	// L2098
        #pragma HLS pipeline II=1
        float v1700 = v1690[(v1697 + (v1692 * 8))][(((v1698 + v1693) + (v1694 * 8)) - 1)][(((v1699 + v1695) + (v1696 * 8)) - 1)];	// L2099
        v1691[v1697][v1698][v1699] = v1700;	// L2100
        float v1701 = v1690[(v1697 + (v1692 * 8))][(((v1698 + v1693) + (v1694 * 8)) - 1)][((v1699 + v1695) + (v1696 * 8))];	// L2101
        v1691[v1697][v1698][(v1699 + 1)] = v1701;	// L2102
        float v1702 = v1690[(v1697 + (v1692 * 8))][((v1698 + v1693) + (v1694 * 8))][(((v1699 + v1695) + (v1696 * 8)) - 1)];	// L2103
        v1691[v1697][(v1698 + 1)][v1699] = v1702;	// L2104
        float v1703 = v1690[(v1697 + (v1692 * 8))][((v1698 + v1693) + (v1694 * 8))][((v1699 + v1695) + (v1696 * 8))];	// L2105
        v1691[v1697][(v1698 + 1)][(v1699 + 1)] = v1703;	// L2106
      }
    }
  }
}

void forward_node112(
  float v1704[128][16][16],
  float v1705[8][8][8],
  int v1706,
  int v1707,
  int v1708
) {	// L2112
  #pragma HLS inline
  #pragma HLS array_partition variable=v1704 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1704 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1705 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1705 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1705 core=ram_t2p_bram

  for (int v1709 = 0; v1709 < 8; v1709 += 1) {	// L2113
    for (int v1710 = 0; v1710 < 8; v1710 += 2) {	// L2114
      for (int v1711 = 0; v1711 < 8; v1711 += 2) {	// L2115
        #pragma HLS pipeline II=1
        float v1712 = v1704[(v1709 + (v1706 * 8))][(v1710 + (v1707 * 8))][(v1711 + (v1708 * 8))];	// L2116
        v1705[v1709][v1710][v1711] = v1712;	// L2117
        float v1713 = v1704[(v1709 + (v1706 * 8))][(v1710 + (v1707 * 8))][((v1711 + (v1708 * 8)) + 1)];	// L2118
        v1705[v1709][v1710][(v1711 + 1)] = v1713;	// L2119
        float v1714 = v1704[(v1709 + (v1706 * 8))][((v1710 + (v1707 * 8)) + 1)][(v1711 + (v1708 * 8))];	// L2120
        v1705[v1709][(v1710 + 1)][v1711] = v1714;	// L2121
        float v1715 = v1704[(v1709 + (v1706 * 8))][((v1710 + (v1707 * 8)) + 1)][((v1711 + (v1708 * 8)) + 1)];	// L2122
        v1705[v1709][(v1710 + 1)][(v1711 + 1)] = v1715;	// L2123
      }
    }
  }
}

void forward_node113(
  float v1716[128][16][16],
  float v1717[8][8][8],
  int v1718,
  int v1719,
  int v1720
) {	// L2129
  #pragma HLS inline
  #pragma HLS array_partition variable=v1716 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1716 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1717 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1717 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1717 core=ram_t2p_bram

  for (int v1721 = 0; v1721 < 8; v1721 += 1) {	// L2130
    for (int v1722 = 0; v1722 < 8; v1722 += 2) {	// L2131
      for (int v1723 = 0; v1723 < 8; v1723 += 2) {	// L2132
        #pragma HLS pipeline II=1
        float v1724 = v1716[(v1721 + (v1718 * 8))][(v1722 + (v1719 * 8))][(v1723 + (v1720 * 8))];	// L2133
        v1717[v1721][v1722][v1723] = v1724;	// L2134
        float v1725 = v1716[(v1721 + (v1718 * 8))][(v1722 + (v1719 * 8))][((v1723 + (v1720 * 8)) + 1)];	// L2135
        v1717[v1721][v1722][(v1723 + 1)] = v1725;	// L2136
        float v1726 = v1716[(v1721 + (v1718 * 8))][((v1722 + (v1719 * 8)) + 1)][(v1723 + (v1720 * 8))];	// L2137
        v1717[v1721][(v1722 + 1)][v1723] = v1726;	// L2138
        float v1727 = v1716[(v1721 + (v1718 * 8))][((v1722 + (v1719 * 8)) + 1)][((v1723 + (v1720 * 8)) + 1)];	// L2139
        v1717[v1721][(v1722 + 1)][(v1723 + 1)] = v1727;	// L2140
      }
    }
  }
}

void forward_node107(
  float v1728[128][16][16],
  float v1729[128][16][16],
  float v1730[128][128][3][3],
  float v1731[128][16][16],
  float v1732[128][16][16]
) {	// L2146
  #pragma HLS array_partition variable=v1728 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1728 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1729 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1729 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1731 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1731 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1732 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1732 cyclic factor=2 dim=3

  for (int v1733 = 0; v1733 < 9216; v1733 += 1) {	// L2147
    #pragma HLS dataflow
    int v1734 = (v1733 % 2);	// L2148
    int v1735 = ((v1733 / 2) % 2);	// L2149
    int v1736 = (((v1733 / 2) / 2) % 16);	// L2150
    int v1737 = ((((v1733 / 2) / 2) / 16) % 3);	// L2151
    int v1738 = (((((v1733 / 2) / 2) / 16) / 3) % 3);	// L2152
    int v1739 = (((((v1733 / 2) / 2) / 16) / 3) / 3);	// L2153
    float v1740[8][8];	// L2154
    #pragma HLS resource variable=v1740 core=ram_t2p_bram

    float v1741[8][8][8];	// L2155
    #pragma HLS array_partition variable=v1741 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1741 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1741 core=ram_t2p_bram

    float v1742[8][8][8];	// L2156
    #pragma HLS array_partition variable=v1742 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1742 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1742 core=ram_t2p_bram

    float v1743[8][8][8];	// L2157
    #pragma HLS array_partition variable=v1743 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1743 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1743 core=ram_t2p_bram

    forward_node113(v1729, v1743, v1736, v1735, v1734);	// L2158
    forward_node112(v1731, v1742, v1736, v1735, v1734);	// L2159
    forward_node111(v1728, v1741, v1739, v1738, v1735, v1737, v1734);	// L2160
    forward_node110(v1730, v1740, v1738, v1737, v1736, v1739);	// L2161
    float v1744[8][8][8];	// L2162
    #pragma HLS array_partition variable=v1744 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1744 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1744 core=ram_t2p_bram

    forward_node109(v1743, v1741, v1740, v1742, v1744, v1739, v1738, v1737);	// L2163
    forward_node108(v1744, v1732, v1736, v1735, v1734);	// L2164
  }
}

void forward_node115(
  float v1745[8][8][8],
  float v1746[128][16][16],
  int v1747,
  int v1748,
  int v1749
) {	// L2168
  #pragma HLS inline
  #pragma HLS array_partition variable=v1745 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1745 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1746 cyclic factor=2 dim=3

  for (int v1750 = 0; v1750 < 8; v1750 += 1) {	// L2169
    for (int v1751 = 0; v1751 < 8; v1751 += 1) {	// L2170
      for (int v1752 = 0; v1752 < 8; v1752 += 2) {	// L2171
        #pragma HLS pipeline II=1
        float v1753 = v1745[v1750][v1751][v1752];	// L2172
        v1746[(v1750 + (v1747 * 8))][(v1751 + (v1748 * 8))][(v1752 + (v1749 * 8))] = v1753;	// L2173
        float v1754 = v1745[v1750][v1751][(v1752 + 1)];	// L2174
        v1746[(v1750 + (v1747 * 8))][(v1751 + (v1748 * 8))][((v1752 + (v1749 * 8)) + 1)] = v1754;	// L2175
      }
    }
  }
}

void forward_node116(
  float v1755[8][8],
  float v1756[8][8][8],
  float v1757[8][8][8],
  float v1758[8][8][8],
  float v1759[8][8][8],
  int v1760,
  int v1761,
  int v1762
) {	// L2181
  #pragma HLS inline
  #pragma HLS resource variable=v1755 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1756 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1756 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1757 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1757 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1758 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1758 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1759 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1759 core=ram_t2p_bram

  for (int v1763 = 0; v1763 < 8; v1763 += 1) {	// L2183
    for (int v1764 = 0; v1764 < 8; v1764 += 1) {	// L2184
      for (int v1765 = 0; v1765 < 8; v1765 += 1) {	// L2185
        for (int v1766 = 0; v1766 < 8; v1766 += 2) {	// L2186
          #pragma HLS pipeline II=1
          float v1767 = v1757[v1764][v1765][v1766];	// L2187
          float v1768 = v1758[v1764][v1765][v1766];	// L2188
          float v1769 = v1759[v1764][v1765][v1766];	// L2189
          float v1770 = (v1763 == 0) ? v1768 : v1769;	// L2190
          float v1771 = ((v1763 + (v1762 * 8)) == 0 && v1761 == 0 && v1760 == 0) ? v1767 : v1770;	// L2191
          float v1772 = v1756[v1763][v1765][v1766];	// L2192
          float v1773 = v1755[v1764][v1763];	// L2193
          float v1774 = v1772 * v1773;	// L2194
          float v1775 = v1771 + v1774;	// L2195
          bool v1776 = v1775 > (float)0.000000;	// L2196
          float v1777 = v1776 ? v1775 : (float)0.000000;	// L2197
          float v1778 = ((((-v1763) + (v1762 * -8)) + 63) == 0 && ((-v1761) + 2) == 0 && ((-v1760) + 2) == 0) ? v1777 : v1775;	// L2198
          v1759[v1764][v1765][v1766] = v1778;	// L2199
          float v1779 = v1757[v1764][v1765][(v1766 + 1)];	// L2200
          float v1780 = v1758[v1764][v1765][(v1766 + 1)];	// L2201
          float v1781 = v1759[v1764][v1765][(v1766 + 1)];	// L2202
          float v1782 = (v1763 == 0) ? v1780 : v1781;	// L2203
          float v1783 = ((v1763 + (v1762 * 8)) == 0 && v1761 == 0 && v1760 == 0) ? v1779 : v1782;	// L2204
          float v1784 = v1756[v1763][v1765][(v1766 + 1)];	// L2205
          float v1785 = v1784 * v1773;	// L2206
          float v1786 = v1783 + v1785;	// L2207
          bool v1787 = v1786 > (float)0.000000;	// L2208
          float v1788 = v1787 ? v1786 : (float)0.000000;	// L2209
          float v1789 = ((((-v1763) + (v1762 * -8)) + 63) == 0 && ((-v1761) + 2) == 0 && ((-v1760) + 2) == 0) ? v1788 : v1786;	// L2210
          v1759[v1764][v1765][(v1766 + 1)] = v1789;	// L2211
        }
      }
    }
  }
}

void forward_node117(
  float v1790[128][64][3][3],
  float v1791[8][8],
  int v1792,
  int v1793,
  int v1794,
  int v1795
) {	// L2218
  #pragma HLS inline
  #pragma HLS resource variable=v1791 core=ram_t2p_bram

  for (int v1796 = 0; v1796 < 8; v1796 += 1) {	// L2219
    for (int v1797 = 0; v1797 < 8; v1797 += 1) {	// L2220
      #pragma HLS pipeline II=1
      float v1798 = v1790[(v1796 + (v1794 * 8))][(v1797 + (v1795 * 8))][v1792][v1793];	// L2221
      v1791[v1796][v1797] = v1798;	// L2222
    }
  }
}

void forward_node118(
  float v1799[64][32][32],
  float v1800[8][8][8],
  int v1801,
  int v1802,
  int v1803,
  int v1804,
  int v1805
) {	// L2227
  #pragma HLS inline
  #pragma HLS array_partition variable=v1799 cyclic factor=4 dim=3

  #pragma HLS array_partition variable=v1800 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1800 core=ram_t2p_bram

  for (int v1806 = 0; v1806 < 8; v1806 += 1) {	// L2228
    for (int v1807 = 0; v1807 < 8; v1807 += 1) {	// L2229
      for (int v1808 = 0; v1808 < 8; v1808 += 2) {	// L2230
        #pragma HLS pipeline II=1
        float v1809 = v1799[(v1806 + (v1801 * 8))][((((v1807 * 2) + v1802) + (v1803 * 16)) - 1)][((((v1808 * 2) + v1804) + (v1805 * 16)) - 1)];	// L2231
        v1800[v1806][v1807][v1808] = v1809;	// L2232
        float v1810 = v1799[(v1806 + (v1801 * 8))][((((v1807 * 2) + v1802) + (v1803 * 16)) - 1)][((((v1808 * 2) + v1804) + (v1805 * 16)) + 1)];	// L2233
        v1800[v1806][v1807][(v1808 + 1)] = v1810;	// L2234
      }
    }
  }
}

void forward_node119(
  float v1811[128][16][16],
  float v1812[8][8][8],
  int v1813,
  int v1814,
  int v1815
) {	// L2240
  #pragma HLS inline
  #pragma HLS array_partition variable=v1811 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1812 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1812 core=ram_t2p_bram

  for (int v1816 = 0; v1816 < 8; v1816 += 1) {	// L2241
    for (int v1817 = 0; v1817 < 8; v1817 += 1) {	// L2242
      for (int v1818 = 0; v1818 < 8; v1818 += 2) {	// L2243
        #pragma HLS pipeline II=1
        float v1819 = v1811[(v1816 + (v1813 * 8))][(v1817 + (v1814 * 8))][(v1818 + (v1815 * 8))];	// L2244
        v1812[v1816][v1817][v1818] = v1819;	// L2245
        float v1820 = v1811[(v1816 + (v1813 * 8))][(v1817 + (v1814 * 8))][((v1818 + (v1815 * 8)) + 1)];	// L2246
        v1812[v1816][v1817][(v1818 + 1)] = v1820;	// L2247
      }
    }
  }
}

void forward_node120(
  float v1821[128][16][16],
  float v1822[8][8][8],
  int v1823,
  int v1824,
  int v1825
) {	// L2253
  #pragma HLS inline
  #pragma HLS array_partition variable=v1821 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1822 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1822 core=ram_t2p_bram

  for (int v1826 = 0; v1826 < 8; v1826 += 1) {	// L2254
    for (int v1827 = 0; v1827 < 8; v1827 += 1) {	// L2255
      for (int v1828 = 0; v1828 < 8; v1828 += 2) {	// L2256
        #pragma HLS pipeline II=1
        float v1829 = v1821[(v1826 + (v1823 * 8))][(v1827 + (v1824 * 8))][(v1828 + (v1825 * 8))];	// L2257
        v1822[v1826][v1827][v1828] = v1829;	// L2258
        float v1830 = v1821[(v1826 + (v1823 * 8))][(v1827 + (v1824 * 8))][((v1828 + (v1825 * 8)) + 1)];	// L2259
        v1822[v1826][v1827][(v1828 + 1)] = v1830;	// L2260
      }
    }
  }
}

void forward_node114(
  float v1831[64][32][32],
  float v1832[128][64][3][3],
  float v1833[128][16][16],
  float v1834[128][16][16],
  float v1835[128][16][16]
) {	// L2266
  #pragma HLS array_partition variable=v1831 cyclic factor=4 dim=3

  #pragma HLS array_partition variable=v1833 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1834 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1835 cyclic factor=2 dim=3

  for (int v1836 = 0; v1836 < 4608; v1836 += 1) {	// L2267
    #pragma HLS dataflow
    int v1837 = (v1836 % 2);	// L2268
    int v1838 = ((v1836 / 2) % 2);	// L2269
    int v1839 = (((v1836 / 2) / 2) % 16);	// L2270
    int v1840 = ((((v1836 / 2) / 2) / 16) % 3);	// L2271
    int v1841 = (((((v1836 / 2) / 2) / 16) / 3) % 3);	// L2272
    int v1842 = (((((v1836 / 2) / 2) / 16) / 3) / 3);	// L2273
    float v1843[8][8];	// L2274
    #pragma HLS resource variable=v1843 core=ram_t2p_bram

    float v1844[8][8][8];	// L2275
    #pragma HLS array_partition variable=v1844 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1844 core=ram_t2p_bram

    float v1845[8][8][8];	// L2276
    #pragma HLS array_partition variable=v1845 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1845 core=ram_t2p_bram

    float v1846[8][8][8];	// L2277
    #pragma HLS array_partition variable=v1846 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1846 core=ram_t2p_bram

    forward_node120(v1833, v1846, v1839, v1838, v1837);	// L2278
    forward_node119(v1834, v1845, v1839, v1838, v1837);	// L2279
    forward_node118(v1831, v1844, v1842, v1841, v1838, v1840, v1837);	// L2280
    forward_node117(v1832, v1843, v1841, v1840, v1839, v1842);	// L2281
    float v1847[8][8][8];	// L2282
    #pragma HLS array_partition variable=v1847 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1847 core=ram_t2p_bram

    forward_node116(v1843, v1844, v1846, v1845, v1847, v1840, v1841, v1842);	// L2283
    forward_node115(v1847, v1835, v1839, v1838, v1837);	// L2284
  }
}

void forward_node122(
  float v1848[8][8][8],
  float v1849[64][32][32],
  int v1850,
  int v1851,
  int v1852
) {	// L2288
  #pragma HLS inline
  #pragma HLS array_partition variable=v1848 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1848 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1848 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1849 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1849 cyclic factor=2 dim=3

  for (int v1853 = 0; v1853 < 8; v1853 += 1) {	// L2289
    for (int v1854 = 0; v1854 < 8; v1854 += 2) {	// L2290
      for (int v1855 = 0; v1855 < 8; v1855 += 2) {	// L2291
        #pragma HLS pipeline II=1
        float v1856 = v1848[v1853][v1854][v1855];	// L2292
        v1849[(v1853 + (v1850 * 8))][(v1854 + (v1851 * 8))][(v1855 + (v1852 * 8))] = v1856;	// L2293
        float v1857 = v1848[v1853][v1854][(v1855 + 1)];	// L2294
        v1849[(v1853 + (v1850 * 8))][(v1854 + (v1851 * 8))][((v1855 + (v1852 * 8)) + 1)] = v1857;	// L2295
        float v1858 = v1848[v1853][(v1854 + 1)][v1855];	// L2296
        v1849[(v1853 + (v1850 * 8))][((v1854 + (v1851 * 8)) + 1)][(v1855 + (v1852 * 8))] = v1858;	// L2297
        float v1859 = v1848[v1853][(v1854 + 1)][(v1855 + 1)];	// L2298
        v1849[(v1853 + (v1850 * 8))][((v1854 + (v1851 * 8)) + 1)][((v1855 + (v1852 * 8)) + 1)] = v1859;	// L2299
      }
    }
  }
}

void forward_node123(
  float v1860[8][8][8],
  float v1861[64][32][32],
  int v1862,
  int v1863,
  int v1864
) {	// L2305
  #pragma HLS inline
  #pragma HLS array_partition variable=v1860 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1860 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1860 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1861 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1861 cyclic factor=2 dim=3

  for (int v1865 = 0; v1865 < 8; v1865 += 1) {	// L2306
    for (int v1866 = 0; v1866 < 8; v1866 += 2) {	// L2307
      for (int v1867 = 0; v1867 < 8; v1867 += 2) {	// L2308
        #pragma HLS pipeline II=1
        float v1868 = v1860[v1865][v1866][v1867];	// L2309
        v1861[(v1865 + (v1862 * 8))][(v1866 + (v1863 * 8))][(v1867 + (v1864 * 8))] = v1868;	// L2310
        float v1869 = v1860[v1865][v1866][(v1867 + 1)];	// L2311
        v1861[(v1865 + (v1862 * 8))][(v1866 + (v1863 * 8))][((v1867 + (v1864 * 8)) + 1)] = v1869;	// L2312
        float v1870 = v1860[v1865][(v1866 + 1)][v1867];	// L2313
        v1861[(v1865 + (v1862 * 8))][((v1866 + (v1863 * 8)) + 1)][(v1867 + (v1864 * 8))] = v1870;	// L2314
        float v1871 = v1860[v1865][(v1866 + 1)][(v1867 + 1)];	// L2315
        v1861[(v1865 + (v1862 * 8))][((v1866 + (v1863 * 8)) + 1)][((v1867 + (v1864 * 8)) + 1)] = v1871;	// L2316
      }
    }
  }
}

void forward_node124(
  float v1872[8][8],
  float v1873[8][8][8],
  float v1874[8][8][8],
  float v1875[8][8][8],
  float v1876[8][8][8],
  float v1877[8][8][8],
  int v1878,
  int v1879,
  int v1880
) {	// L2322
  #pragma HLS inline
  #pragma HLS resource variable=v1872 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1873 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1873 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1873 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1874 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1874 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1874 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1875 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1875 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1875 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1876 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1876 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1876 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1877 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1877 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1877 core=ram_t2p_bram

  for (int v1881 = 0; v1881 < 8; v1881 += 1) {	// L2324
    for (int v1882 = 0; v1882 < 8; v1882 += 1) {	// L2325
      for (int v1883 = 0; v1883 < 8; v1883 += 2) {	// L2326
        for (int v1884 = 0; v1884 < 8; v1884 += 2) {	// L2327
          #pragma HLS pipeline II=1
          float v1885 = v1873[v1881][v1883][v1884];	// L2328
          float v1886 = v1872[v1882][v1881];	// L2329
          float v1887 = v1875[v1882][v1883][v1884];	// L2330
          float v1888 = v1876[v1882][v1883][v1884];	// L2331
          float v1889 = (v1881 == 0) ? v1887 : v1888;	// L2332
          float v1890 = v1885 * v1886;	// L2333
          float v1891 = v1889 + v1890;	// L2334
          v1876[v1882][v1883][v1884] = v1891;	// L2335
          float v1892 = v1874[v1882][v1883][v1884];	// L2336
          float v1893 = v1891 + v1892;	// L2337
          bool v1894 = v1893 > (float)0.000000;	// L2338
          float v1895 = v1894 ? v1893 : (float)0.000000;	// L2339
          if ((((-v1881) + (v1879 * -8)) + 63) == 0 && ((-v1880) + 2) == 0 && ((-v1878) + 2) == 0) {	// L2340
            v1877[v1882][v1883][v1884] = v1895;	// L2341
          }
          float v1896 = v1873[v1881][v1883][(v1884 + 1)];	// L2343
          float v1897 = v1875[v1882][v1883][(v1884 + 1)];	// L2344
          float v1898 = v1876[v1882][v1883][(v1884 + 1)];	// L2345
          float v1899 = (v1881 == 0) ? v1897 : v1898;	// L2346
          float v1900 = v1896 * v1886;	// L2347
          float v1901 = v1899 + v1900;	// L2348
          v1876[v1882][v1883][(v1884 + 1)] = v1901;	// L2349
          float v1902 = v1874[v1882][v1883][(v1884 + 1)];	// L2350
          float v1903 = v1901 + v1902;	// L2351
          bool v1904 = v1903 > (float)0.000000;	// L2352
          float v1905 = v1904 ? v1903 : (float)0.000000;	// L2353
          if ((((-v1881) + (v1879 * -8)) + 63) == 0 && ((-v1880) + 2) == 0 && ((-v1878) + 2) == 0) {	// L2354
            v1877[v1882][v1883][(v1884 + 1)] = v1905;	// L2355
          }
          float v1906 = v1873[v1881][(v1883 + 1)][v1884];	// L2357
          float v1907 = v1875[v1882][(v1883 + 1)][v1884];	// L2358
          float v1908 = v1876[v1882][(v1883 + 1)][v1884];	// L2359
          float v1909 = (v1881 == 0) ? v1907 : v1908;	// L2360
          float v1910 = v1906 * v1886;	// L2361
          float v1911 = v1909 + v1910;	// L2362
          v1876[v1882][(v1883 + 1)][v1884] = v1911;	// L2363
          float v1912 = v1874[v1882][(v1883 + 1)][v1884];	// L2364
          float v1913 = v1911 + v1912;	// L2365
          bool v1914 = v1913 > (float)0.000000;	// L2366
          float v1915 = v1914 ? v1913 : (float)0.000000;	// L2367
          if ((((-v1881) + (v1879 * -8)) + 63) == 0 && ((-v1880) + 2) == 0 && ((-v1878) + 2) == 0) {	// L2368
            v1877[v1882][(v1883 + 1)][v1884] = v1915;	// L2369
          }
          float v1916 = v1873[v1881][(v1883 + 1)][(v1884 + 1)];	// L2371
          float v1917 = v1875[v1882][(v1883 + 1)][(v1884 + 1)];	// L2372
          float v1918 = v1876[v1882][(v1883 + 1)][(v1884 + 1)];	// L2373
          float v1919 = (v1881 == 0) ? v1917 : v1918;	// L2374
          float v1920 = v1916 * v1886;	// L2375
          float v1921 = v1919 + v1920;	// L2376
          v1876[v1882][(v1883 + 1)][(v1884 + 1)] = v1921;	// L2377
          float v1922 = v1874[v1882][(v1883 + 1)][(v1884 + 1)];	// L2378
          float v1923 = v1921 + v1922;	// L2379
          bool v1924 = v1923 > (float)0.000000;	// L2380
          float v1925 = v1924 ? v1923 : (float)0.000000;	// L2381
          if ((((-v1881) + (v1879 * -8)) + 63) == 0 && ((-v1880) + 2) == 0 && ((-v1878) + 2) == 0) {	// L2382
            v1877[v1882][(v1883 + 1)][(v1884 + 1)] = v1925;	// L2383
          }
        }
      }
    }
  }
}

void forward_node125(
  float v1926[64][32][32],
  float v1927[8][8][8],
  int v1928,
  int v1929,
  int v1930
) {	// L2391
  #pragma HLS inline
  #pragma HLS array_partition variable=v1926 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1926 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1927 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1927 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1927 core=ram_t2p_bram

  for (int v1931 = 0; v1931 < 8; v1931 += 1) {	// L2392
    for (int v1932 = 0; v1932 < 8; v1932 += 2) {	// L2393
      for (int v1933 = 0; v1933 < 8; v1933 += 2) {	// L2394
        #pragma HLS pipeline II=1
        float v1934 = v1926[(v1931 + (v1928 * 8))][(v1932 + (v1929 * 8))][(v1933 + (v1930 * 8))];	// L2395
        v1927[v1931][v1932][v1933] = v1934;	// L2396
        float v1935 = v1926[(v1931 + (v1928 * 8))][(v1932 + (v1929 * 8))][((v1933 + (v1930 * 8)) + 1)];	// L2397
        v1927[v1931][v1932][(v1933 + 1)] = v1935;	// L2398
        float v1936 = v1926[(v1931 + (v1928 * 8))][((v1932 + (v1929 * 8)) + 1)][(v1933 + (v1930 * 8))];	// L2399
        v1927[v1931][(v1932 + 1)][v1933] = v1936;	// L2400
        float v1937 = v1926[(v1931 + (v1928 * 8))][((v1932 + (v1929 * 8)) + 1)][((v1933 + (v1930 * 8)) + 1)];	// L2401
        v1927[v1931][(v1932 + 1)][(v1933 + 1)] = v1937;	// L2402
      }
    }
  }
}

void forward_node126(
  float v1938[64][32][32],
  float v1939[8][8][8],
  int v1940,
  int v1941,
  int v1942
) {	// L2408
  #pragma HLS inline
  #pragma HLS array_partition variable=v1938 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1938 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1939 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1939 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1939 core=ram_t2p_bram

  for (int v1943 = 0; v1943 < 8; v1943 += 1) {	// L2409
    for (int v1944 = 0; v1944 < 8; v1944 += 2) {	// L2410
      for (int v1945 = 0; v1945 < 8; v1945 += 2) {	// L2411
        #pragma HLS pipeline II=1
        float v1946 = v1938[(v1943 + (v1940 * 8))][(v1944 + (v1941 * 8))][(v1945 + (v1942 * 8))];	// L2412
        v1939[v1943][v1944][v1945] = v1946;	// L2413
        float v1947 = v1938[(v1943 + (v1940 * 8))][(v1944 + (v1941 * 8))][((v1945 + (v1942 * 8)) + 1)];	// L2414
        v1939[v1943][v1944][(v1945 + 1)] = v1947;	// L2415
        float v1948 = v1938[(v1943 + (v1940 * 8))][((v1944 + (v1941 * 8)) + 1)][(v1945 + (v1942 * 8))];	// L2416
        v1939[v1943][(v1944 + 1)][v1945] = v1948;	// L2417
        float v1949 = v1938[(v1943 + (v1940 * 8))][((v1944 + (v1941 * 8)) + 1)][((v1945 + (v1942 * 8)) + 1)];	// L2418
        v1939[v1943][(v1944 + 1)][(v1945 + 1)] = v1949;	// L2419
      }
    }
  }
}

void forward_node127(
  float v1950[64][64][3][3],
  float v1951[8][8],
  int v1952,
  int v1953,
  int v1954,
  int v1955
) {	// L2425
  #pragma HLS inline
  #pragma HLS resource variable=v1951 core=ram_t2p_bram

  for (int v1956 = 0; v1956 < 8; v1956 += 1) {	// L2426
    for (int v1957 = 0; v1957 < 8; v1957 += 1) {	// L2427
      #pragma HLS pipeline II=1
      float v1958 = v1950[(v1956 + (v1954 * 8))][(v1957 + (v1955 * 8))][v1952][v1953];	// L2428
      v1951[v1956][v1957] = v1958;	// L2429
    }
  }
}

void forward_node128(
  float v1959[64][32][32],
  float v1960[8][8][8],
  int v1961,
  int v1962,
  int v1963,
  int v1964,
  int v1965
) {	// L2434
  #pragma HLS inline
  #pragma HLS array_partition variable=v1959 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1959 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1960 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1960 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1960 core=ram_t2p_bram

  for (int v1966 = 0; v1966 < 8; v1966 += 1) {	// L2435
    for (int v1967 = 0; v1967 < 8; v1967 += 2) {	// L2436
      for (int v1968 = 0; v1968 < 8; v1968 += 2) {	// L2437
        #pragma HLS pipeline II=1
        float v1969 = v1959[(v1966 + (v1961 * 8))][(((v1967 + v1962) + (v1963 * 8)) - 1)][(((v1968 + v1964) + (v1965 * 8)) - 1)];	// L2438
        v1960[v1966][v1967][v1968] = v1969;	// L2439
        float v1970 = v1959[(v1966 + (v1961 * 8))][(((v1967 + v1962) + (v1963 * 8)) - 1)][((v1968 + v1964) + (v1965 * 8))];	// L2440
        v1960[v1966][v1967][(v1968 + 1)] = v1970;	// L2441
        float v1971 = v1959[(v1966 + (v1961 * 8))][((v1967 + v1962) + (v1963 * 8))][(((v1968 + v1964) + (v1965 * 8)) - 1)];	// L2442
        v1960[v1966][(v1967 + 1)][v1968] = v1971;	// L2443
        float v1972 = v1959[(v1966 + (v1961 * 8))][((v1967 + v1962) + (v1963 * 8))][((v1968 + v1964) + (v1965 * 8))];	// L2444
        v1960[v1966][(v1967 + 1)][(v1968 + 1)] = v1972;	// L2445
      }
    }
  }
}

void forward_node121(
  float v1973[64][32][32],
  float v1974[64][64][3][3],
  float v1975[64][32][32],
  float v1976[64][32][32],
  float v1977[64][32][32],
  float v1978[64][32][32]
) {	// L2451
  #pragma HLS array_partition variable=v1973 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1973 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1975 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1975 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1976 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1976 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1977 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1977 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v1978 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1978 cyclic factor=2 dim=3

  for (int v1979 = 0; v1979 < 9216; v1979 += 1) {	// L2452
    #pragma HLS dataflow
    int v1980 = (v1979 % 4);	// L2453
    int v1981 = ((v1979 / 4) % 4);	// L2454
    int v1982 = (((v1979 / 4) / 4) % 8);	// L2455
    int v1983 = ((((v1979 / 4) / 4) / 8) % 3);	// L2456
    int v1984 = (((((v1979 / 4) / 4) / 8) / 3) % 3);	// L2457
    int v1985 = (((((v1979 / 4) / 4) / 8) / 3) / 3);	// L2458
    float v1986[8][8][8];	// L2459
    #pragma HLS array_partition variable=v1986 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1986 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1986 core=ram_t2p_bram

    float v1987[8][8][8];	// L2460
    #pragma HLS array_partition variable=v1987 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1987 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1987 core=ram_t2p_bram

    float v1988[8][8][8];	// L2461
    #pragma HLS array_partition variable=v1988 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1988 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1988 core=ram_t2p_bram

    float v1989[8][8];	// L2462
    #pragma HLS resource variable=v1989 core=ram_t2p_bram

    float v1990[8][8][8];	// L2463
    #pragma HLS array_partition variable=v1990 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1990 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1990 core=ram_t2p_bram

    forward_node128(v1975, v1990, v1985, v1984, v1981, v1983, v1980);	// L2464
    forward_node127(v1974, v1989, v1984, v1983, v1982, v1985);	// L2465
    forward_node126(v1976, v1988, v1982, v1981, v1980);	// L2466
    forward_node125(v1973, v1987, v1982, v1981, v1980);	// L2467
    float v1991[8][8][8];	// L2468
    #pragma HLS array_partition variable=v1991 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v1991 cyclic factor=2 dim=3
    #pragma HLS resource variable=v1991 core=ram_t2p_bram

    forward_node124(v1989, v1990, v1987, v1988, v1991, v1986, v1983, v1985, v1984);	// L2469
    forward_node123(v1991, v1978, v1982, v1981, v1980);	// L2470
    forward_node122(v1986, v1977, v1982, v1981, v1980);	// L2471
  }
}

void forward_node130(
  float v1992[8][8][8],
  float v1993[64][32][32],
  int v1994,
  int v1995,
  int v1996
) {	// L2475
  #pragma HLS inline
  #pragma HLS array_partition variable=v1992 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1992 cyclic factor=2 dim=3
  #pragma HLS resource variable=v1992 core=ram_t2p_bram

  #pragma HLS array_partition variable=v1993 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v1993 cyclic factor=2 dim=3

  for (int v1997 = 0; v1997 < 8; v1997 += 1) {	// L2476
    for (int v1998 = 0; v1998 < 8; v1998 += 2) {	// L2477
      for (int v1999 = 0; v1999 < 8; v1999 += 2) {	// L2478
        #pragma HLS pipeline II=1
        float v2000 = v1992[v1997][v1998][v1999];	// L2479
        v1993[(v1997 + (v1994 * 8))][(v1998 + (v1995 * 8))][(v1999 + (v1996 * 8))] = v2000;	// L2480
        float v2001 = v1992[v1997][v1998][(v1999 + 1)];	// L2481
        v1993[(v1997 + (v1994 * 8))][(v1998 + (v1995 * 8))][((v1999 + (v1996 * 8)) + 1)] = v2001;	// L2482
        float v2002 = v1992[v1997][(v1998 + 1)][v1999];	// L2483
        v1993[(v1997 + (v1994 * 8))][((v1998 + (v1995 * 8)) + 1)][(v1999 + (v1996 * 8))] = v2002;	// L2484
        float v2003 = v1992[v1997][(v1998 + 1)][(v1999 + 1)];	// L2485
        v1993[(v1997 + (v1994 * 8))][((v1998 + (v1995 * 8)) + 1)][((v1999 + (v1996 * 8)) + 1)] = v2003;	// L2486
      }
    }
  }
}

void forward_node131(
  float v2004[8][8],
  float v2005[8][8][8],
  float v2006[8][8][8],
  float v2007[8][8][8],
  float v2008[8][8][8],
  int v2009,
  int v2010,
  int v2011
) {	// L2492
  #pragma HLS inline
  #pragma HLS resource variable=v2004 core=ram_t2p_bram

  #pragma HLS array_partition variable=v2005 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2005 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2005 core=ram_t2p_bram

  #pragma HLS array_partition variable=v2006 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2006 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2006 core=ram_t2p_bram

  #pragma HLS array_partition variable=v2007 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2007 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2007 core=ram_t2p_bram

  #pragma HLS array_partition variable=v2008 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2008 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2008 core=ram_t2p_bram

  for (int v2012 = 0; v2012 < 8; v2012 += 1) {	// L2494
    for (int v2013 = 0; v2013 < 8; v2013 += 1) {	// L2495
      for (int v2014 = 0; v2014 < 8; v2014 += 2) {	// L2496
        for (int v2015 = 0; v2015 < 8; v2015 += 2) {	// L2497
          #pragma HLS pipeline II=1
          float v2016 = v2005[v2013][v2014][v2015];	// L2498
          float v2017 = v2007[v2013][v2014][v2015];	// L2499
          float v2018 = v2008[v2013][v2014][v2015];	// L2500
          float v2019 = (v2012 == 0) ? v2017 : v2018;	// L2501
          float v2020 = ((v2012 + (v2009 * 8)) == 0 && v2011 == 0 && v2010 == 0) ? v2016 : v2019;	// L2502
          float v2021 = v2006[v2012][v2014][v2015];	// L2503
          float v2022 = v2004[v2013][v2012];	// L2504
          float v2023 = v2021 * v2022;	// L2505
          float v2024 = v2020 + v2023;	// L2506
          bool v2025 = v2024 > (float)0.000000;	// L2507
          float v2026 = v2025 ? v2024 : (float)0.000000;	// L2508
          float v2027 = ((((-v2012) + (v2009 * -8)) + 63) == 0 && ((-v2011) + 2) == 0 && ((-v2010) + 2) == 0) ? v2026 : v2024;	// L2509
          v2008[v2013][v2014][v2015] = v2027;	// L2510
          float v2028 = v2005[v2013][v2014][(v2015 + 1)];	// L2511
          float v2029 = v2007[v2013][v2014][(v2015 + 1)];	// L2512
          float v2030 = v2008[v2013][v2014][(v2015 + 1)];	// L2513
          float v2031 = (v2012 == 0) ? v2029 : v2030;	// L2514
          float v2032 = ((v2012 + (v2009 * 8)) == 0 && v2011 == 0 && v2010 == 0) ? v2028 : v2031;	// L2515
          float v2033 = v2006[v2012][v2014][(v2015 + 1)];	// L2516
          float v2034 = v2033 * v2022;	// L2517
          float v2035 = v2032 + v2034;	// L2518
          bool v2036 = v2035 > (float)0.000000;	// L2519
          float v2037 = v2036 ? v2035 : (float)0.000000;	// L2520
          float v2038 = ((((-v2012) + (v2009 * -8)) + 63) == 0 && ((-v2011) + 2) == 0 && ((-v2010) + 2) == 0) ? v2037 : v2035;	// L2521
          v2008[v2013][v2014][(v2015 + 1)] = v2038;	// L2522
          float v2039 = v2005[v2013][(v2014 + 1)][v2015];	// L2523
          float v2040 = v2007[v2013][(v2014 + 1)][v2015];	// L2524
          float v2041 = v2008[v2013][(v2014 + 1)][v2015];	// L2525
          float v2042 = (v2012 == 0) ? v2040 : v2041;	// L2526
          float v2043 = ((v2012 + (v2009 * 8)) == 0 && v2011 == 0 && v2010 == 0) ? v2039 : v2042;	// L2527
          float v2044 = v2006[v2012][(v2014 + 1)][v2015];	// L2528
          float v2045 = v2044 * v2022;	// L2529
          float v2046 = v2043 + v2045;	// L2530
          bool v2047 = v2046 > (float)0.000000;	// L2531
          float v2048 = v2047 ? v2046 : (float)0.000000;	// L2532
          float v2049 = ((((-v2012) + (v2009 * -8)) + 63) == 0 && ((-v2011) + 2) == 0 && ((-v2010) + 2) == 0) ? v2048 : v2046;	// L2533
          v2008[v2013][(v2014 + 1)][v2015] = v2049;	// L2534
          float v2050 = v2005[v2013][(v2014 + 1)][(v2015 + 1)];	// L2535
          float v2051 = v2007[v2013][(v2014 + 1)][(v2015 + 1)];	// L2536
          float v2052 = v2008[v2013][(v2014 + 1)][(v2015 + 1)];	// L2537
          float v2053 = (v2012 == 0) ? v2051 : v2052;	// L2538
          float v2054 = ((v2012 + (v2009 * 8)) == 0 && v2011 == 0 && v2010 == 0) ? v2050 : v2053;	// L2539
          float v2055 = v2006[v2012][(v2014 + 1)][(v2015 + 1)];	// L2540
          float v2056 = v2055 * v2022;	// L2541
          float v2057 = v2054 + v2056;	// L2542
          bool v2058 = v2057 > (float)0.000000;	// L2543
          float v2059 = v2058 ? v2057 : (float)0.000000;	// L2544
          float v2060 = ((((-v2012) + (v2009 * -8)) + 63) == 0 && ((-v2011) + 2) == 0 && ((-v2010) + 2) == 0) ? v2059 : v2057;	// L2545
          v2008[v2013][(v2014 + 1)][(v2015 + 1)] = v2060;	// L2546
        }
      }
    }
  }
}

void forward_node132(
  float v2061[64][64][3][3],
  float v2062[8][8],
  int v2063,
  int v2064,
  int v2065,
  int v2066
) {	// L2553
  #pragma HLS inline
  #pragma HLS resource variable=v2062 core=ram_t2p_bram

  for (int v2067 = 0; v2067 < 8; v2067 += 1) {	// L2554
    for (int v2068 = 0; v2068 < 8; v2068 += 1) {	// L2555
      #pragma HLS pipeline II=1
      float v2069 = v2061[(v2067 + (v2065 * 8))][(v2068 + (v2066 * 8))][v2063][v2064];	// L2556
      v2062[v2067][v2068] = v2069;	// L2557
    }
  }
}

void forward_node133(
  float v2070[64][32][32],
  float v2071[8][8][8],
  int v2072,
  int v2073,
  int v2074,
  int v2075,
  int v2076
) {	// L2562
  #pragma HLS inline
  #pragma HLS array_partition variable=v2070 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2070 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2071 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2071 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2071 core=ram_t2p_bram

  for (int v2077 = 0; v2077 < 8; v2077 += 1) {	// L2563
    for (int v2078 = 0; v2078 < 8; v2078 += 2) {	// L2564
      for (int v2079 = 0; v2079 < 8; v2079 += 2) {	// L2565
        #pragma HLS pipeline II=1
        float v2080 = v2070[(v2077 + (v2072 * 8))][(((v2078 + v2073) + (v2074 * 8)) - 1)][(((v2079 + v2075) + (v2076 * 8)) - 1)];	// L2566
        v2071[v2077][v2078][v2079] = v2080;	// L2567
        float v2081 = v2070[(v2077 + (v2072 * 8))][(((v2078 + v2073) + (v2074 * 8)) - 1)][((v2079 + v2075) + (v2076 * 8))];	// L2568
        v2071[v2077][v2078][(v2079 + 1)] = v2081;	// L2569
        float v2082 = v2070[(v2077 + (v2072 * 8))][((v2078 + v2073) + (v2074 * 8))][(((v2079 + v2075) + (v2076 * 8)) - 1)];	// L2570
        v2071[v2077][(v2078 + 1)][v2079] = v2082;	// L2571
        float v2083 = v2070[(v2077 + (v2072 * 8))][((v2078 + v2073) + (v2074 * 8))][((v2079 + v2075) + (v2076 * 8))];	// L2572
        v2071[v2077][(v2078 + 1)][(v2079 + 1)] = v2083;	// L2573
      }
    }
  }
}

void forward_node134(
  float v2084[64][32][32],
  float v2085[8][8][8],
  int v2086,
  int v2087,
  int v2088
) {	// L2579
  #pragma HLS inline
  #pragma HLS array_partition variable=v2084 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2084 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2085 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2085 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2085 core=ram_t2p_bram

  for (int v2089 = 0; v2089 < 8; v2089 += 1) {	// L2580
    for (int v2090 = 0; v2090 < 8; v2090 += 2) {	// L2581
      for (int v2091 = 0; v2091 < 8; v2091 += 2) {	// L2582
        #pragma HLS pipeline II=1
        float v2092 = v2084[(v2089 + (v2086 * 8))][(v2090 + (v2087 * 8))][(v2091 + (v2088 * 8))];	// L2583
        v2085[v2089][v2090][v2091] = v2092;	// L2584
        float v2093 = v2084[(v2089 + (v2086 * 8))][(v2090 + (v2087 * 8))][((v2091 + (v2088 * 8)) + 1)];	// L2585
        v2085[v2089][v2090][(v2091 + 1)] = v2093;	// L2586
        float v2094 = v2084[(v2089 + (v2086 * 8))][((v2090 + (v2087 * 8)) + 1)][(v2091 + (v2088 * 8))];	// L2587
        v2085[v2089][(v2090 + 1)][v2091] = v2094;	// L2588
        float v2095 = v2084[(v2089 + (v2086 * 8))][((v2090 + (v2087 * 8)) + 1)][((v2091 + (v2088 * 8)) + 1)];	// L2589
        v2085[v2089][(v2090 + 1)][(v2091 + 1)] = v2095;	// L2590
      }
    }
  }
}

void forward_node135(
  float v2096[64][32][32],
  float v2097[8][8][8],
  int v2098,
  int v2099,
  int v2100
) {	// L2596
  #pragma HLS inline
  #pragma HLS array_partition variable=v2096 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2096 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2097 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2097 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2097 core=ram_t2p_bram

  for (int v2101 = 0; v2101 < 8; v2101 += 1) {	// L2597
    for (int v2102 = 0; v2102 < 8; v2102 += 2) {	// L2598
      for (int v2103 = 0; v2103 < 8; v2103 += 2) {	// L2599
        #pragma HLS pipeline II=1
        float v2104 = v2096[(v2101 + (v2098 * 8))][(v2102 + (v2099 * 8))][(v2103 + (v2100 * 8))];	// L2600
        v2097[v2101][v2102][v2103] = v2104;	// L2601
        float v2105 = v2096[(v2101 + (v2098 * 8))][(v2102 + (v2099 * 8))][((v2103 + (v2100 * 8)) + 1)];	// L2602
        v2097[v2101][v2102][(v2103 + 1)] = v2105;	// L2603
        float v2106 = v2096[(v2101 + (v2098 * 8))][((v2102 + (v2099 * 8)) + 1)][(v2103 + (v2100 * 8))];	// L2604
        v2097[v2101][(v2102 + 1)][v2103] = v2106;	// L2605
        float v2107 = v2096[(v2101 + (v2098 * 8))][((v2102 + (v2099 * 8)) + 1)][((v2103 + (v2100 * 8)) + 1)];	// L2606
        v2097[v2101][(v2102 + 1)][(v2103 + 1)] = v2107;	// L2607
      }
    }
  }
}

void forward_node129(
  float v2108[64][32][32],
  float v2109[64][64][3][3],
  float v2110[64][32][32],
  float v2111[64][32][32],
  float v2112[64][32][32]
) {	// L2613
  #pragma HLS array_partition variable=v2108 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2108 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2110 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2110 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2111 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2111 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2112 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2112 cyclic factor=2 dim=3

  for (int v2113 = 0; v2113 < 9216; v2113 += 1) {	// L2614
    #pragma HLS dataflow
    int v2114 = (v2113 % 4);	// L2615
    int v2115 = ((v2113 / 4) % 4);	// L2616
    int v2116 = (((v2113 / 4) / 4) % 8);	// L2617
    int v2117 = ((((v2113 / 4) / 4) / 8) % 3);	// L2618
    int v2118 = (((((v2113 / 4) / 4) / 8) / 3) % 3);	// L2619
    int v2119 = (((((v2113 / 4) / 4) / 8) / 3) / 3);	// L2620
    float v2120[8][8];	// L2621
    #pragma HLS resource variable=v2120 core=ram_t2p_bram

    float v2121[8][8][8];	// L2622
    #pragma HLS array_partition variable=v2121 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v2121 cyclic factor=2 dim=3
    #pragma HLS resource variable=v2121 core=ram_t2p_bram

    float v2122[8][8][8];	// L2623
    #pragma HLS array_partition variable=v2122 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v2122 cyclic factor=2 dim=3
    #pragma HLS resource variable=v2122 core=ram_t2p_bram

    float v2123[8][8][8];	// L2624
    #pragma HLS array_partition variable=v2123 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v2123 cyclic factor=2 dim=3
    #pragma HLS resource variable=v2123 core=ram_t2p_bram

    forward_node135(v2110, v2123, v2116, v2115, v2114);	// L2625
    forward_node134(v2111, v2122, v2116, v2115, v2114);	// L2626
    forward_node133(v2108, v2121, v2119, v2118, v2115, v2117, v2114);	// L2627
    forward_node132(v2109, v2120, v2118, v2117, v2116, v2119);	// L2628
    float v2124[8][8][8];	// L2629
    #pragma HLS array_partition variable=v2124 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v2124 cyclic factor=2 dim=3
    #pragma HLS resource variable=v2124 core=ram_t2p_bram

    forward_node131(v2120, v2123, v2121, v2122, v2124, v2119, v2117, v2118);	// L2630
    forward_node130(v2124, v2112, v2116, v2115, v2114);	// L2631
  }
}

void forward_node137(
  float v2125[8][8][8],
  float v2126[64][32][32],
  int v2127,
  int v2128,
  int v2129
) {	// L2635
  #pragma HLS inline
  #pragma HLS array_partition variable=v2125 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2125 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2125 core=ram_t2p_bram

  #pragma HLS array_partition variable=v2126 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2126 cyclic factor=2 dim=3

  for (int v2130 = 0; v2130 < 8; v2130 += 1) {	// L2636
    for (int v2131 = 0; v2131 < 8; v2131 += 2) {	// L2637
      for (int v2132 = 0; v2132 < 8; v2132 += 2) {	// L2638
        #pragma HLS pipeline II=1
        float v2133 = v2125[v2130][v2131][v2132];	// L2639
        v2126[(v2130 + (v2127 * 8))][(v2131 + (v2128 * 8))][(v2132 + (v2129 * 8))] = v2133;	// L2640
        float v2134 = v2125[v2130][v2131][(v2132 + 1)];	// L2641
        v2126[(v2130 + (v2127 * 8))][(v2131 + (v2128 * 8))][((v2132 + (v2129 * 8)) + 1)] = v2134;	// L2642
        float v2135 = v2125[v2130][(v2131 + 1)][v2132];	// L2643
        v2126[(v2130 + (v2127 * 8))][((v2131 + (v2128 * 8)) + 1)][(v2132 + (v2129 * 8))] = v2135;	// L2644
        float v2136 = v2125[v2130][(v2131 + 1)][(v2132 + 1)];	// L2645
        v2126[(v2130 + (v2127 * 8))][((v2131 + (v2128 * 8)) + 1)][((v2132 + (v2129 * 8)) + 1)] = v2136;	// L2646
      }
    }
  }
}

void forward_node138(
  float v2137[8][8][8],
  float v2138[64][32][32],
  int v2139,
  int v2140,
  int v2141
) {	// L2652
  #pragma HLS inline
  #pragma HLS array_partition variable=v2137 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2137 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2137 core=ram_t2p_bram

  #pragma HLS array_partition variable=v2138 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2138 cyclic factor=2 dim=3

  for (int v2142 = 0; v2142 < 8; v2142 += 1) {	// L2653
    for (int v2143 = 0; v2143 < 8; v2143 += 2) {	// L2654
      for (int v2144 = 0; v2144 < 8; v2144 += 2) {	// L2655
        #pragma HLS pipeline II=1
        float v2145 = v2137[v2142][v2143][v2144];	// L2656
        v2138[(v2142 + (v2139 * 8))][(v2143 + (v2140 * 8))][(v2144 + (v2141 * 8))] = v2145;	// L2657
        float v2146 = v2137[v2142][v2143][(v2144 + 1)];	// L2658
        v2138[(v2142 + (v2139 * 8))][(v2143 + (v2140 * 8))][((v2144 + (v2141 * 8)) + 1)] = v2146;	// L2659
        float v2147 = v2137[v2142][(v2143 + 1)][v2144];	// L2660
        v2138[(v2142 + (v2139 * 8))][((v2143 + (v2140 * 8)) + 1)][(v2144 + (v2141 * 8))] = v2147;	// L2661
        float v2148 = v2137[v2142][(v2143 + 1)][(v2144 + 1)];	// L2662
        v2138[(v2142 + (v2139 * 8))][((v2143 + (v2140 * 8)) + 1)][((v2144 + (v2141 * 8)) + 1)] = v2148;	// L2663
      }
    }
  }
}

void forward_node139(
  float v2149[8][8],
  float v2150[8][8][8],
  float v2151[8][8][8],
  float v2152[8][8][8],
  float v2153[8][8][8],
  float v2154[8][8][8],
  float v2155[8][8][8],
  int v2156,
  int v2157,
  int v2158
) {	// L2669
  #pragma HLS inline
  #pragma HLS resource variable=v2149 core=ram_t2p_bram

  #pragma HLS array_partition variable=v2150 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2150 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2150 core=ram_t2p_bram

  #pragma HLS array_partition variable=v2151 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2151 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2151 core=ram_t2p_bram

  #pragma HLS array_partition variable=v2152 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2152 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2152 core=ram_t2p_bram

  #pragma HLS array_partition variable=v2153 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2153 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2153 core=ram_t2p_bram

  #pragma HLS array_partition variable=v2154 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2154 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2154 core=ram_t2p_bram

  #pragma HLS array_partition variable=v2155 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2155 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2155 core=ram_t2p_bram

  for (int v2159 = 0; v2159 < 8; v2159 += 1) {	// L2671
    for (int v2160 = 0; v2160 < 8; v2160 += 1) {	// L2672
      for (int v2161 = 0; v2161 < 8; v2161 += 2) {	// L2673
        for (int v2162 = 0; v2162 < 8; v2162 += 2) {	// L2674
          #pragma HLS pipeline II=1
          float v2163 = v2152[v2160][v2161][v2162];	// L2675
          float v2164 = v2153[v2160][v2161][v2162];	// L2676
          float v2165 = v2155[v2160][v2161][v2162];	// L2677
          float v2166 = (v2159 == 0) ? v2164 : v2165;	// L2678
          float v2167 = ((v2159 + (v2156 * 8)) == 0 && v2158 == 0 && v2157 == 0) ? v2163 : v2166;	// L2679
          float v2168 = v2150[v2159][v2161][v2162];	// L2680
          float v2169 = v2149[v2160][v2159];	// L2681
          float v2170 = v2168 * v2169;	// L2682
          float v2171 = v2167 + v2170;	// L2683
          v2155[v2160][v2161][v2162] = v2171;	// L2684
          float v2172 = v2151[v2160][v2161][v2162];	// L2685
          float v2173 = v2171 + v2172;	// L2686
          bool v2174 = v2173 > (float)0.000000;	// L2687
          float v2175 = v2174 ? v2173 : (float)0.000000;	// L2688
          if ((((-v2159) + (v2156 * -8)) + 63) == 0 && ((-v2158) + 2) == 0 && ((-v2157) + 2) == 0) {	// L2689
            v2154[v2160][v2161][v2162] = v2175;	// L2690
          }
          float v2176 = v2152[v2160][v2161][(v2162 + 1)];	// L2692
          float v2177 = v2153[v2160][v2161][(v2162 + 1)];	// L2693
          float v2178 = v2155[v2160][v2161][(v2162 + 1)];	// L2694
          float v2179 = (v2159 == 0) ? v2177 : v2178;	// L2695
          float v2180 = ((v2159 + (v2156 * 8)) == 0 && v2158 == 0 && v2157 == 0) ? v2176 : v2179;	// L2696
          float v2181 = v2150[v2159][v2161][(v2162 + 1)];	// L2697
          float v2182 = v2181 * v2169;	// L2698
          float v2183 = v2180 + v2182;	// L2699
          v2155[v2160][v2161][(v2162 + 1)] = v2183;	// L2700
          float v2184 = v2151[v2160][v2161][(v2162 + 1)];	// L2701
          float v2185 = v2183 + v2184;	// L2702
          bool v2186 = v2185 > (float)0.000000;	// L2703
          float v2187 = v2186 ? v2185 : (float)0.000000;	// L2704
          if ((((-v2159) + (v2156 * -8)) + 63) == 0 && ((-v2158) + 2) == 0 && ((-v2157) + 2) == 0) {	// L2705
            v2154[v2160][v2161][(v2162 + 1)] = v2187;	// L2706
          }
          float v2188 = v2152[v2160][(v2161 + 1)][v2162];	// L2708
          float v2189 = v2153[v2160][(v2161 + 1)][v2162];	// L2709
          float v2190 = v2155[v2160][(v2161 + 1)][v2162];	// L2710
          float v2191 = (v2159 == 0) ? v2189 : v2190;	// L2711
          float v2192 = ((v2159 + (v2156 * 8)) == 0 && v2158 == 0 && v2157 == 0) ? v2188 : v2191;	// L2712
          float v2193 = v2150[v2159][(v2161 + 1)][v2162];	// L2713
          float v2194 = v2193 * v2169;	// L2714
          float v2195 = v2192 + v2194;	// L2715
          v2155[v2160][(v2161 + 1)][v2162] = v2195;	// L2716
          float v2196 = v2151[v2160][(v2161 + 1)][v2162];	// L2717
          float v2197 = v2195 + v2196;	// L2718
          bool v2198 = v2197 > (float)0.000000;	// L2719
          float v2199 = v2198 ? v2197 : (float)0.000000;	// L2720
          if ((((-v2159) + (v2156 * -8)) + 63) == 0 && ((-v2158) + 2) == 0 && ((-v2157) + 2) == 0) {	// L2721
            v2154[v2160][(v2161 + 1)][v2162] = v2199;	// L2722
          }
          float v2200 = v2152[v2160][(v2161 + 1)][(v2162 + 1)];	// L2724
          float v2201 = v2153[v2160][(v2161 + 1)][(v2162 + 1)];	// L2725
          float v2202 = v2155[v2160][(v2161 + 1)][(v2162 + 1)];	// L2726
          float v2203 = (v2159 == 0) ? v2201 : v2202;	// L2727
          float v2204 = ((v2159 + (v2156 * 8)) == 0 && v2158 == 0 && v2157 == 0) ? v2200 : v2203;	// L2728
          float v2205 = v2150[v2159][(v2161 + 1)][(v2162 + 1)];	// L2729
          float v2206 = v2205 * v2169;	// L2730
          float v2207 = v2204 + v2206;	// L2731
          v2155[v2160][(v2161 + 1)][(v2162 + 1)] = v2207;	// L2732
          float v2208 = v2151[v2160][(v2161 + 1)][(v2162 + 1)];	// L2733
          float v2209 = v2207 + v2208;	// L2734
          bool v2210 = v2209 > (float)0.000000;	// L2735
          float v2211 = v2210 ? v2209 : (float)0.000000;	// L2736
          if ((((-v2159) + (v2156 * -8)) + 63) == 0 && ((-v2158) + 2) == 0 && ((-v2157) + 2) == 0) {	// L2737
            v2154[v2160][(v2161 + 1)][(v2162 + 1)] = v2211;	// L2738
          }
        }
      }
    }
  }
}

void forward_node140(
  float v2212[64][32][32],
  float v2213[8][8][8],
  int v2214,
  int v2215,
  int v2216
) {	// L2746
  #pragma HLS inline
  #pragma HLS array_partition variable=v2212 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2212 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2213 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2213 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2213 core=ram_t2p_bram

  for (int v2217 = 0; v2217 < 8; v2217 += 1) {	// L2747
    for (int v2218 = 0; v2218 < 8; v2218 += 2) {	// L2748
      for (int v2219 = 0; v2219 < 8; v2219 += 2) {	// L2749
        #pragma HLS pipeline II=1
        float v2220 = v2212[(v2217 + (v2214 * 8))][(v2218 + (v2215 * 8))][(v2219 + (v2216 * 8))];	// L2750
        v2213[v2217][v2218][v2219] = v2220;	// L2751
        float v2221 = v2212[(v2217 + (v2214 * 8))][(v2218 + (v2215 * 8))][((v2219 + (v2216 * 8)) + 1)];	// L2752
        v2213[v2217][v2218][(v2219 + 1)] = v2221;	// L2753
        float v2222 = v2212[(v2217 + (v2214 * 8))][((v2218 + (v2215 * 8)) + 1)][(v2219 + (v2216 * 8))];	// L2754
        v2213[v2217][(v2218 + 1)][v2219] = v2222;	// L2755
        float v2223 = v2212[(v2217 + (v2214 * 8))][((v2218 + (v2215 * 8)) + 1)][((v2219 + (v2216 * 8)) + 1)];	// L2756
        v2213[v2217][(v2218 + 1)][(v2219 + 1)] = v2223;	// L2757
      }
    }
  }
}

void forward_node141(
  float v2224[64][64][3][3],
  float v2225[8][8],
  int v2226,
  int v2227,
  int v2228,
  int v2229
) {	// L2763
  #pragma HLS inline
  #pragma HLS resource variable=v2225 core=ram_t2p_bram

  for (int v2230 = 0; v2230 < 8; v2230 += 1) {	// L2764
    for (int v2231 = 0; v2231 < 8; v2231 += 1) {	// L2765
      #pragma HLS pipeline II=1
      float v2232 = v2224[(v2230 + (v2228 * 8))][(v2231 + (v2229 * 8))][v2226][v2227];	// L2766
      v2225[v2230][v2231] = v2232;	// L2767
    }
  }
}

void forward_node142(
  float v2233[64][32][32],
  float v2234[8][8][8],
  int v2235,
  int v2236,
  int v2237,
  int v2238,
  int v2239
) {	// L2772
  #pragma HLS inline
  #pragma HLS array_partition variable=v2233 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2233 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2234 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2234 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2234 core=ram_t2p_bram

  for (int v2240 = 0; v2240 < 8; v2240 += 1) {	// L2773
    for (int v2241 = 0; v2241 < 8; v2241 += 2) {	// L2774
      for (int v2242 = 0; v2242 < 8; v2242 += 2) {	// L2775
        #pragma HLS pipeline II=1
        float v2243 = v2233[(v2240 + (v2235 * 8))][(((v2241 + v2236) + (v2237 * 8)) - 1)][(((v2242 + v2238) + (v2239 * 8)) - 1)];	// L2776
        v2234[v2240][v2241][v2242] = v2243;	// L2777
        float v2244 = v2233[(v2240 + (v2235 * 8))][(((v2241 + v2236) + (v2237 * 8)) - 1)][((v2242 + v2238) + (v2239 * 8))];	// L2778
        v2234[v2240][v2241][(v2242 + 1)] = v2244;	// L2779
        float v2245 = v2233[(v2240 + (v2235 * 8))][((v2241 + v2236) + (v2237 * 8))][(((v2242 + v2238) + (v2239 * 8)) - 1)];	// L2780
        v2234[v2240][(v2241 + 1)][v2242] = v2245;	// L2781
        float v2246 = v2233[(v2240 + (v2235 * 8))][((v2241 + v2236) + (v2237 * 8))][((v2242 + v2238) + (v2239 * 8))];	// L2782
        v2234[v2240][(v2241 + 1)][(v2242 + 1)] = v2246;	// L2783
      }
    }
  }
}

void forward_node143(
  float v2247[64][32][32],
  float v2248[8][8][8],
  int v2249,
  int v2250,
  int v2251
) {	// L2789
  #pragma HLS inline
  #pragma HLS array_partition variable=v2247 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2247 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2248 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2248 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2248 core=ram_t2p_bram

  for (int v2252 = 0; v2252 < 8; v2252 += 1) {	// L2790
    for (int v2253 = 0; v2253 < 8; v2253 += 2) {	// L2791
      for (int v2254 = 0; v2254 < 8; v2254 += 2) {	// L2792
        #pragma HLS pipeline II=1
        float v2255 = v2247[(v2252 + (v2249 * 8))][(v2253 + (v2250 * 8))][(v2254 + (v2251 * 8))];	// L2793
        v2248[v2252][v2253][v2254] = v2255;	// L2794
        float v2256 = v2247[(v2252 + (v2249 * 8))][(v2253 + (v2250 * 8))][((v2254 + (v2251 * 8)) + 1)];	// L2795
        v2248[v2252][v2253][(v2254 + 1)] = v2256;	// L2796
        float v2257 = v2247[(v2252 + (v2249 * 8))][((v2253 + (v2250 * 8)) + 1)][(v2254 + (v2251 * 8))];	// L2797
        v2248[v2252][(v2253 + 1)][v2254] = v2257;	// L2798
        float v2258 = v2247[(v2252 + (v2249 * 8))][((v2253 + (v2250 * 8)) + 1)][((v2254 + (v2251 * 8)) + 1)];	// L2799
        v2248[v2252][(v2253 + 1)][(v2254 + 1)] = v2258;	// L2800
      }
    }
  }
}

void forward_node144(
  float v2259[64][32][32],
  float v2260[8][8][8],
  int v2261,
  int v2262,
  int v2263
) {	// L2806
  #pragma HLS inline
  #pragma HLS array_partition variable=v2259 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2259 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2260 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2260 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2260 core=ram_t2p_bram

  for (int v2264 = 0; v2264 < 8; v2264 += 1) {	// L2807
    for (int v2265 = 0; v2265 < 8; v2265 += 2) {	// L2808
      for (int v2266 = 0; v2266 < 8; v2266 += 2) {	// L2809
        #pragma HLS pipeline II=1
        float v2267 = v2259[(v2264 + (v2261 * 8))][(v2265 + (v2262 * 8))][(v2266 + (v2263 * 8))];	// L2810
        v2260[v2264][v2265][v2266] = v2267;	// L2811
        float v2268 = v2259[(v2264 + (v2261 * 8))][(v2265 + (v2262 * 8))][((v2266 + (v2263 * 8)) + 1)];	// L2812
        v2260[v2264][v2265][(v2266 + 1)] = v2268;	// L2813
        float v2269 = v2259[(v2264 + (v2261 * 8))][((v2265 + (v2262 * 8)) + 1)][(v2266 + (v2263 * 8))];	// L2814
        v2260[v2264][(v2265 + 1)][v2266] = v2269;	// L2815
        float v2270 = v2259[(v2264 + (v2261 * 8))][((v2265 + (v2262 * 8)) + 1)][((v2266 + (v2263 * 8)) + 1)];	// L2816
        v2260[v2264][(v2265 + 1)][(v2266 + 1)] = v2270;	// L2817
      }
    }
  }
}

void forward_node136(
  float v2271[64][32][32],
  float v2272[64][64][3][3],
  float v2273[64][32][32],
  float v2274[64][32][32],
  float v2275[64][32][32],
  float v2276[64][32][32],
  float v2277[64][32][32]
) {	// L2823
  #pragma HLS array_partition variable=v2271 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2271 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2273 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2273 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2274 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2274 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2275 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2275 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2276 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2276 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2277 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2277 cyclic factor=2 dim=3

  for (int v2278 = 0; v2278 < 9216; v2278 += 1) {	// L2824
    #pragma HLS dataflow
    int v2279 = (v2278 % 4);	// L2825
    int v2280 = ((v2278 / 4) % 4);	// L2826
    int v2281 = (((v2278 / 4) / 4) % 8);	// L2827
    int v2282 = ((((v2278 / 4) / 4) / 8) % 3);	// L2828
    int v2283 = (((((v2278 / 4) / 4) / 8) / 3) % 3);	// L2829
    int v2284 = (((((v2278 / 4) / 4) / 8) / 3) / 3);	// L2830
    float v2285[8][8][8];	// L2831
    #pragma HLS array_partition variable=v2285 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v2285 cyclic factor=2 dim=3
    #pragma HLS resource variable=v2285 core=ram_t2p_bram

    float v2286[8][8][8];	// L2832
    #pragma HLS array_partition variable=v2286 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v2286 cyclic factor=2 dim=3
    #pragma HLS resource variable=v2286 core=ram_t2p_bram

    float v2287[8][8];	// L2833
    #pragma HLS resource variable=v2287 core=ram_t2p_bram

    float v2288[8][8][8];	// L2834
    #pragma HLS array_partition variable=v2288 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v2288 cyclic factor=2 dim=3
    #pragma HLS resource variable=v2288 core=ram_t2p_bram

    float v2289[8][8][8];	// L2835
    #pragma HLS array_partition variable=v2289 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v2289 cyclic factor=2 dim=3
    #pragma HLS resource variable=v2289 core=ram_t2p_bram

    float v2290[8][8][8];	// L2836
    #pragma HLS array_partition variable=v2290 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v2290 cyclic factor=2 dim=3
    #pragma HLS resource variable=v2290 core=ram_t2p_bram

    forward_node144(v2273, v2290, v2281, v2280, v2279);	// L2837
    forward_node143(v2275, v2289, v2281, v2280, v2279);	// L2838
    forward_node142(v2274, v2288, v2284, v2283, v2280, v2282, v2279);	// L2839
    forward_node141(v2272, v2287, v2283, v2282, v2281, v2284);	// L2840
    forward_node140(v2271, v2286, v2281, v2280, v2279);	// L2841
    float v2291[8][8][8];	// L2842
    #pragma HLS array_partition variable=v2291 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v2291 cyclic factor=2 dim=3
    #pragma HLS resource variable=v2291 core=ram_t2p_bram

    forward_node139(v2287, v2288, v2286, v2290, v2289, v2285, v2291, v2284, v2282, v2283);	// L2843
    forward_node138(v2291, v2277, v2281, v2280, v2279);	// L2844
    forward_node137(v2285, v2276, v2281, v2280, v2279);	// L2845
  }
}

void forward_node146(
  float v2292[8][8][8],
  float v2293[64][32][32],
  int v2294,
  int v2295,
  int v2296
) {	// L2849
  #pragma HLS inline
  #pragma HLS array_partition variable=v2292 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2292 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2292 core=ram_t2p_bram

  #pragma HLS array_partition variable=v2293 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2293 cyclic factor=2 dim=3

  for (int v2297 = 0; v2297 < 8; v2297 += 1) {	// L2850
    for (int v2298 = 0; v2298 < 8; v2298 += 2) {	// L2851
      for (int v2299 = 0; v2299 < 8; v2299 += 2) {	// L2852
        #pragma HLS pipeline II=1
        float v2300 = v2292[v2297][v2298][v2299];	// L2853
        v2293[(v2297 + (v2294 * 8))][(v2298 + (v2295 * 8))][(v2299 + (v2296 * 8))] = v2300;	// L2854
        float v2301 = v2292[v2297][v2298][(v2299 + 1)];	// L2855
        v2293[(v2297 + (v2294 * 8))][(v2298 + (v2295 * 8))][((v2299 + (v2296 * 8)) + 1)] = v2301;	// L2856
        float v2302 = v2292[v2297][(v2298 + 1)][v2299];	// L2857
        v2293[(v2297 + (v2294 * 8))][((v2298 + (v2295 * 8)) + 1)][(v2299 + (v2296 * 8))] = v2302;	// L2858
        float v2303 = v2292[v2297][(v2298 + 1)][(v2299 + 1)];	// L2859
        v2293[(v2297 + (v2294 * 8))][((v2298 + (v2295 * 8)) + 1)][((v2299 + (v2296 * 8)) + 1)] = v2303;	// L2860
      }
    }
  }
}

void forward_node147(
  float v2304[8][8][8],
  float v2305[8][8],
  float v2306[8][8][8],
  float v2307[8][8][8],
  float v2308[8][8][8],
  int v2309,
  int v2310,
  int v2311
) {	// L2866
  #pragma HLS inline
  #pragma HLS array_partition variable=v2304 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2304 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2304 core=ram_t2p_bram

  #pragma HLS resource variable=v2305 core=ram_t2p_bram

  #pragma HLS array_partition variable=v2306 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2306 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2306 core=ram_t2p_bram

  #pragma HLS array_partition variable=v2307 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2307 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2307 core=ram_t2p_bram

  #pragma HLS array_partition variable=v2308 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2308 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2308 core=ram_t2p_bram

  for (int v2312 = 0; v2312 < 8; v2312 += 1) {	// L2868
    for (int v2313 = 0; v2313 < 8; v2313 += 1) {	// L2869
      for (int v2314 = 0; v2314 < 8; v2314 += 2) {	// L2870
        for (int v2315 = 0; v2315 < 8; v2315 += 2) {	// L2871
          #pragma HLS pipeline II=1
          float v2316 = v2304[v2313][v2314][v2315];	// L2872
          float v2317 = v2307[v2313][v2314][v2315];	// L2873
          float v2318 = v2308[v2313][v2314][v2315];	// L2874
          float v2319 = (v2312 == 0) ? v2317 : v2318;	// L2875
          float v2320 = ((v2312 + (v2311 * 8)) == 0 && v2309 == 0 && v2310 == 0) ? v2316 : v2319;	// L2876
          float v2321 = v2306[v2312][v2314][v2315];	// L2877
          float v2322 = v2305[v2313][v2312];	// L2878
          float v2323 = v2321 * v2322;	// L2879
          float v2324 = v2320 + v2323;	// L2880
          bool v2325 = v2324 > (float)0.000000;	// L2881
          float v2326 = v2325 ? v2324 : (float)0.000000;	// L2882
          float v2327 = ((((-v2312) + (v2311 * -8)) + 63) == 0 && ((-v2309) + 2) == 0 && ((-v2310) + 2) == 0) ? v2326 : v2324;	// L2883
          v2308[v2313][v2314][v2315] = v2327;	// L2884
          float v2328 = v2304[v2313][v2314][(v2315 + 1)];	// L2885
          float v2329 = v2307[v2313][v2314][(v2315 + 1)];	// L2886
          float v2330 = v2308[v2313][v2314][(v2315 + 1)];	// L2887
          float v2331 = (v2312 == 0) ? v2329 : v2330;	// L2888
          float v2332 = ((v2312 + (v2311 * 8)) == 0 && v2309 == 0 && v2310 == 0) ? v2328 : v2331;	// L2889
          float v2333 = v2306[v2312][v2314][(v2315 + 1)];	// L2890
          float v2334 = v2333 * v2322;	// L2891
          float v2335 = v2332 + v2334;	// L2892
          bool v2336 = v2335 > (float)0.000000;	// L2893
          float v2337 = v2336 ? v2335 : (float)0.000000;	// L2894
          float v2338 = ((((-v2312) + (v2311 * -8)) + 63) == 0 && ((-v2309) + 2) == 0 && ((-v2310) + 2) == 0) ? v2337 : v2335;	// L2895
          v2308[v2313][v2314][(v2315 + 1)] = v2338;	// L2896
          float v2339 = v2304[v2313][(v2314 + 1)][v2315];	// L2897
          float v2340 = v2307[v2313][(v2314 + 1)][v2315];	// L2898
          float v2341 = v2308[v2313][(v2314 + 1)][v2315];	// L2899
          float v2342 = (v2312 == 0) ? v2340 : v2341;	// L2900
          float v2343 = ((v2312 + (v2311 * 8)) == 0 && v2309 == 0 && v2310 == 0) ? v2339 : v2342;	// L2901
          float v2344 = v2306[v2312][(v2314 + 1)][v2315];	// L2902
          float v2345 = v2344 * v2322;	// L2903
          float v2346 = v2343 + v2345;	// L2904
          bool v2347 = v2346 > (float)0.000000;	// L2905
          float v2348 = v2347 ? v2346 : (float)0.000000;	// L2906
          float v2349 = ((((-v2312) + (v2311 * -8)) + 63) == 0 && ((-v2309) + 2) == 0 && ((-v2310) + 2) == 0) ? v2348 : v2346;	// L2907
          v2308[v2313][(v2314 + 1)][v2315] = v2349;	// L2908
          float v2350 = v2304[v2313][(v2314 + 1)][(v2315 + 1)];	// L2909
          float v2351 = v2307[v2313][(v2314 + 1)][(v2315 + 1)];	// L2910
          float v2352 = v2308[v2313][(v2314 + 1)][(v2315 + 1)];	// L2911
          float v2353 = (v2312 == 0) ? v2351 : v2352;	// L2912
          float v2354 = ((v2312 + (v2311 * 8)) == 0 && v2309 == 0 && v2310 == 0) ? v2350 : v2353;	// L2913
          float v2355 = v2306[v2312][(v2314 + 1)][(v2315 + 1)];	// L2914
          float v2356 = v2355 * v2322;	// L2915
          float v2357 = v2354 + v2356;	// L2916
          bool v2358 = v2357 > (float)0.000000;	// L2917
          float v2359 = v2358 ? v2357 : (float)0.000000;	// L2918
          float v2360 = ((((-v2312) + (v2311 * -8)) + 63) == 0 && ((-v2309) + 2) == 0 && ((-v2310) + 2) == 0) ? v2359 : v2357;	// L2919
          v2308[v2313][(v2314 + 1)][(v2315 + 1)] = v2360;	// L2920
        }
      }
    }
  }
}

void forward_node148(
  float v2361[64][64][3][3],
  float v2362[8][8],
  int v2363,
  int v2364,
  int v2365,
  int v2366
) {	// L2927
  #pragma HLS inline
  #pragma HLS resource variable=v2362 core=ram_t2p_bram

  for (int v2367 = 0; v2367 < 8; v2367 += 1) {	// L2928
    for (int v2368 = 0; v2368 < 8; v2368 += 1) {	// L2929
      #pragma HLS pipeline II=1
      float v2369 = v2361[(v2367 + (v2365 * 8))][(v2368 + (v2366 * 8))][v2363][v2364];	// L2930
      v2362[v2367][v2368] = v2369;	// L2931
    }
  }
}

void forward_node149(
  float v2370[64][32][32],
  float v2371[8][8][8],
  int v2372,
  int v2373,
  int v2374,
  int v2375,
  int v2376
) {	// L2936
  #pragma HLS inline
  #pragma HLS array_partition variable=v2370 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2370 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2371 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2371 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2371 core=ram_t2p_bram

  for (int v2377 = 0; v2377 < 8; v2377 += 1) {	// L2937
    for (int v2378 = 0; v2378 < 8; v2378 += 2) {	// L2938
      for (int v2379 = 0; v2379 < 8; v2379 += 2) {	// L2939
        #pragma HLS pipeline II=1
        float v2380 = v2370[(v2377 + (v2372 * 8))][(((v2378 + v2373) + (v2374 * 8)) - 1)][(((v2379 + v2375) + (v2376 * 8)) - 1)];	// L2940
        v2371[v2377][v2378][v2379] = v2380;	// L2941
        float v2381 = v2370[(v2377 + (v2372 * 8))][(((v2378 + v2373) + (v2374 * 8)) - 1)][((v2379 + v2375) + (v2376 * 8))];	// L2942
        v2371[v2377][v2378][(v2379 + 1)] = v2381;	// L2943
        float v2382 = v2370[(v2377 + (v2372 * 8))][((v2378 + v2373) + (v2374 * 8))][(((v2379 + v2375) + (v2376 * 8)) - 1)];	// L2944
        v2371[v2377][(v2378 + 1)][v2379] = v2382;	// L2945
        float v2383 = v2370[(v2377 + (v2372 * 8))][((v2378 + v2373) + (v2374 * 8))][((v2379 + v2375) + (v2376 * 8))];	// L2946
        v2371[v2377][(v2378 + 1)][(v2379 + 1)] = v2383;	// L2947
      }
    }
  }
}

void forward_node150(
  float v2384[64][32][32],
  float v2385[8][8][8],
  int v2386,
  int v2387,
  int v2388
) {	// L2953
  #pragma HLS inline
  #pragma HLS array_partition variable=v2384 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2384 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2385 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2385 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2385 core=ram_t2p_bram

  for (int v2389 = 0; v2389 < 8; v2389 += 1) {	// L2954
    for (int v2390 = 0; v2390 < 8; v2390 += 2) {	// L2955
      for (int v2391 = 0; v2391 < 8; v2391 += 2) {	// L2956
        #pragma HLS pipeline II=1
        float v2392 = v2384[(v2389 + (v2386 * 8))][(v2390 + (v2387 * 8))][(v2391 + (v2388 * 8))];	// L2957
        v2385[v2389][v2390][v2391] = v2392;	// L2958
        float v2393 = v2384[(v2389 + (v2386 * 8))][(v2390 + (v2387 * 8))][((v2391 + (v2388 * 8)) + 1)];	// L2959
        v2385[v2389][v2390][(v2391 + 1)] = v2393;	// L2960
        float v2394 = v2384[(v2389 + (v2386 * 8))][((v2390 + (v2387 * 8)) + 1)][(v2391 + (v2388 * 8))];	// L2961
        v2385[v2389][(v2390 + 1)][v2391] = v2394;	// L2962
        float v2395 = v2384[(v2389 + (v2386 * 8))][((v2390 + (v2387 * 8)) + 1)][((v2391 + (v2388 * 8)) + 1)];	// L2963
        v2385[v2389][(v2390 + 1)][(v2391 + 1)] = v2395;	// L2964
      }
    }
  }
}

void forward_node151(
  float v2396[64][32][32],
  float v2397[8][8][8],
  int v2398,
  int v2399,
  int v2400
) {	// L2970
  #pragma HLS inline
  #pragma HLS array_partition variable=v2396 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2396 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2397 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2397 cyclic factor=2 dim=3
  #pragma HLS resource variable=v2397 core=ram_t2p_bram

  for (int v2401 = 0; v2401 < 8; v2401 += 1) {	// L2971
    for (int v2402 = 0; v2402 < 8; v2402 += 2) {	// L2972
      for (int v2403 = 0; v2403 < 8; v2403 += 2) {	// L2973
        #pragma HLS pipeline II=1
        float v2404 = v2396[(v2401 + (v2398 * 8))][(v2402 + (v2399 * 8))][(v2403 + (v2400 * 8))];	// L2974
        v2397[v2401][v2402][v2403] = v2404;	// L2975
        float v2405 = v2396[(v2401 + (v2398 * 8))][(v2402 + (v2399 * 8))][((v2403 + (v2400 * 8)) + 1)];	// L2976
        v2397[v2401][v2402][(v2403 + 1)] = v2405;	// L2977
        float v2406 = v2396[(v2401 + (v2398 * 8))][((v2402 + (v2399 * 8)) + 1)][(v2403 + (v2400 * 8))];	// L2978
        v2397[v2401][(v2402 + 1)][v2403] = v2406;	// L2979
        float v2407 = v2396[(v2401 + (v2398 * 8))][((v2402 + (v2399 * 8)) + 1)][((v2403 + (v2400 * 8)) + 1)];	// L2980
        v2397[v2401][(v2402 + 1)][(v2403 + 1)] = v2407;	// L2981
      }
    }
  }
}

void forward_node145(
  float v2408[64][32][32],
  float v2409[64][64][3][3],
  float v2410[64][32][32],
  float v2411[64][32][32],
  float v2412[64][32][32]
) {	// L2987
  #pragma HLS array_partition variable=v2408 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2408 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2410 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2410 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2411 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2411 cyclic factor=2 dim=3

  #pragma HLS array_partition variable=v2412 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2412 cyclic factor=2 dim=3

  for (int v2413 = 0; v2413 < 9216; v2413 += 1) {	// L2988
    #pragma HLS dataflow
    int v2414 = (v2413 % 4);	// L2989
    int v2415 = ((v2413 / 4) % 4);	// L2990
    int v2416 = (((v2413 / 4) / 4) % 8);	// L2991
    int v2417 = ((((v2413 / 4) / 4) / 8) % 3);	// L2992
    int v2418 = (((((v2413 / 4) / 4) / 8) / 3) % 3);	// L2993
    int v2419 = (((((v2413 / 4) / 4) / 8) / 3) / 3);	// L2994
    float v2420[8][8];	// L2995
    #pragma HLS resource variable=v2420 core=ram_t2p_bram

    float v2421[8][8][8];	// L2996
    #pragma HLS array_partition variable=v2421 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v2421 cyclic factor=2 dim=3
    #pragma HLS resource variable=v2421 core=ram_t2p_bram

    float v2422[8][8][8];	// L2997
    #pragma HLS array_partition variable=v2422 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v2422 cyclic factor=2 dim=3
    #pragma HLS resource variable=v2422 core=ram_t2p_bram

    float v2423[8][8][8];	// L2998
    #pragma HLS array_partition variable=v2423 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v2423 cyclic factor=2 dim=3
    #pragma HLS resource variable=v2423 core=ram_t2p_bram

    forward_node151(v2410, v2423, v2416, v2415, v2414);	// L2999
    forward_node150(v2411, v2422, v2416, v2415, v2414);	// L3000
    forward_node149(v2408, v2421, v2419, v2418, v2415, v2417, v2414);	// L3001
    forward_node148(v2409, v2420, v2418, v2417, v2416, v2419);	// L3002
    float v2424[8][8][8];	// L3003
    #pragma HLS array_partition variable=v2424 cyclic factor=2 dim=2
    #pragma HLS array_partition variable=v2424 cyclic factor=2 dim=3
    #pragma HLS resource variable=v2424 core=ram_t2p_bram

    forward_node147(v2423, v2420, v2421, v2422, v2424, v2418, v2417, v2419);	// L3004
    forward_node146(v2424, v2412, v2416, v2415, v2414);	// L3005
  }
}

void forward_node153(
  float v2425[8][8][8],
  float v2426[64][32][32],
  int v2427,
  int v2428,
  int v2429
) {	// L3009
  #pragma HLS inline
  #pragma HLS resource variable=v2425 core=ram_t2p_bram

  for (int v2430 = 0; v2430 < 8; v2430 += 1) {	// L3010
    for (int v2431 = 0; v2431 < 8; v2431 += 1) {	// L3011
      for (int v2432 = 0; v2432 < 8; v2432 += 1) {	// L3012
        #pragma HLS pipeline II=1
        float v2433 = v2425[v2430][v2431][v2432];	// L3013
        v2426[(v2430 + (v2427 * 8))][(v2431 + (v2428 * 8))][(v2432 + (v2429 * 8))] = v2433;	// L3014
      }
    }
  }
}

void forward_node154(
  float v2434[8][8][8],
  float v2435[8],
  float v2436[8][8],
  float v2437[8][8][8],
  float v2438[8][8][8],
  int v2439,
  int v2440,
  int v2441
) {	// L3020
  #pragma HLS inline
  #pragma HLS resource variable=v2434 core=ram_t2p_bram

  #pragma HLS resource variable=v2435 core=ram_t2p_bram

  #pragma HLS resource variable=v2436 core=ram_t2p_bram

  #pragma HLS resource variable=v2437 core=ram_t2p_bram

  #pragma HLS resource variable=v2438 core=ram_t2p_bram

  for (int v2442 = 0; v2442 < 8; v2442 += 1) {	// L3022
    for (int v2443 = 0; v2443 < 8; v2443 += 1) {	// L3023
      for (int v2444 = 0; v2444 < 8; v2444 += 1) {	// L3024
        #pragma HLS pipeline II=1
        float v2445 = v2434[v2442][v2443][v2444];	// L3025
        float v2446 = v2437[v2442][v2443][v2444];	// L3026
        float v2447 = (v2439 == 0 && v2440 == 0 && v2441 == 0) ? v2445 : v2446;	// L3027
        float v2448 = v2436[v2443][v2444];	// L3028
        float v2449 = v2435[v2442];	// L3029
        float v2450 = v2448 * v2449;	// L3030
        float v2451 = v2447 + v2450;	// L3031
        bool v2452 = v2451 > (float)0.000000;	// L3032
        float v2453 = v2452 ? v2451 : (float)0.000000;	// L3033
        float v2454 = (((-v2439) + 2) == 0 && ((-v2440) + 2) == 0 && ((-v2441) + 2) == 0) ? v2453 : v2451;	// L3034
        v2438[v2442][v2443][v2444] = v2454;	// L3035
      }
    }
  }
}

void forward_node155(
  float v2455[64][3][3][3],
  float v2456[8],
  int v2457,
  int v2458,
  int v2459,
  int v2460
) {	// L3041
  #pragma HLS inline
  #pragma HLS resource variable=v2456 core=ram_t2p_bram

  for (int v2461 = 0; v2461 < 8; v2461 += 1) {	// L3042
    #pragma HLS pipeline II=1
    float v2462 = v2455[(v2461 + (v2460 * 8))][v2457][v2458][v2459];	// L3043
    v2456[v2461] = v2462;	// L3044
  }
}

void forward_node156(
  float v2463[3][32][32],
  float v2464[8][8],
  int v2465,
  int v2466,
  int v2467,
  int v2468,
  int v2469
) {	// L3048
  #pragma HLS inline
  #pragma HLS resource variable=v2464 core=ram_t2p_bram

  for (int v2470 = 0; v2470 < 8; v2470 += 1) {	// L3049
    for (int v2471 = 0; v2471 < 8; v2471 += 1) {	// L3050
      #pragma HLS pipeline II=1
      float v2472 = v2463[v2465][(((v2470 + v2466) + (v2467 * 8)) - 1)][(((v2471 + v2468) + (v2469 * 8)) - 1)];	// L3051
      v2464[v2470][v2471] = v2472;	// L3052
    }
  }
}

void forward_node157(
  float v2473[64][32][32],
  float v2474[8][8][8],
  int v2475,
  int v2476,
  int v2477
) {	// L3057
  #pragma HLS inline
  #pragma HLS resource variable=v2474 core=ram_t2p_bram

  for (int v2478 = 0; v2478 < 8; v2478 += 1) {	// L3058
    for (int v2479 = 0; v2479 < 8; v2479 += 1) {	// L3059
      for (int v2480 = 0; v2480 < 8; v2480 += 1) {	// L3060
        #pragma HLS pipeline II=1
        float v2481 = v2473[(v2478 + (v2475 * 8))][(v2479 + (v2476 * 8))][(v2480 + (v2477 * 8))];	// L3061
        v2474[v2478][v2479][v2480] = v2481;	// L3062
      }
    }
  }
}

void forward_node158(
  float v2482[64][32][32],
  float v2483[8][8][8],
  int v2484,
  int v2485,
  int v2486
) {	// L3068
  #pragma HLS inline
  #pragma HLS resource variable=v2483 core=ram_t2p_bram

  for (int v2487 = 0; v2487 < 8; v2487 += 1) {	// L3069
    for (int v2488 = 0; v2488 < 8; v2488 += 1) {	// L3070
      for (int v2489 = 0; v2489 < 8; v2489 += 1) {	// L3071
        #pragma HLS pipeline II=1
        float v2490 = v2482[(v2487 + (v2484 * 8))][(v2488 + (v2485 * 8))][(v2489 + (v2486 * 8))];	// L3072
        v2483[v2487][v2488][v2489] = v2490;	// L3073
      }
    }
  }
}

void forward_node152(
  float v2491[3][32][32],
  float v2492[64][3][3][3],
  float v2493[64][32][32],
  float v2494[64][32][32],
  float v2495[64][32][32]
) {	// L3079
  for (int v2496 = 0; v2496 < 3456; v2496 += 1) {	// L3080
    #pragma HLS dataflow
    int v2497 = (v2496 % 4);	// L3081
    int v2498 = ((v2496 / 4) % 4);	// L3082
    int v2499 = (((v2496 / 4) / 4) % 8);	// L3083
    int v2500 = ((((v2496 / 4) / 4) / 8) % 3);	// L3084
    int v2501 = (((((v2496 / 4) / 4) / 8) / 3) % 3);	// L3085
    int v2502 = (((((v2496 / 4) / 4) / 8) / 3) / 3);	// L3086
    float v2503[8];	// L3087
    #pragma HLS resource variable=v2503 core=ram_t2p_bram

    float v2504[8][8];	// L3088
    #pragma HLS resource variable=v2504 core=ram_t2p_bram

    float v2505[8][8][8];	// L3089
    #pragma HLS resource variable=v2505 core=ram_t2p_bram

    float v2506[8][8][8];	// L3090
    #pragma HLS resource variable=v2506 core=ram_t2p_bram

    forward_node158(v2493, v2506, v2499, v2498, v2497);	// L3091
    forward_node157(v2494, v2505, v2499, v2498, v2497);	// L3092
    forward_node156(v2491, v2504, v2502, v2501, v2498, v2500, v2497);	// L3093
    forward_node155(v2492, v2503, v2502, v2501, v2500, v2499);	// L3094
    float v2507[8][8][8];	// L3095
    #pragma HLS resource variable=v2507 core=ram_t2p_bram

    forward_node154(v2506, v2503, v2504, v2505, v2507, v2502, v2501, v2500);	// L3096
    forward_node153(v2507, v2495, v2499, v2498, v2497);	// L3097
  }
}

/// This is top function.
void forward(
  float v2508[3][32][32],
  float v2509[10],
  float v2510[64][3][3][3],
  float v2511[64][64][3][3],
  float v2512[64][64][3][3],
  float v2513[64][64][3][3],
  float v2514[64][64][3][3],
  float v2515[128][64][3][3],
  float v2516[128][128][3][3],
  float v2517[128][64],
  float v2518[128][128][3][3],
  float v2519[128][128][3][3],
  float v2520[256][128][3][3],
  float v2521[256][256][3][3],
  float v2522[256][128],
  float v2523[256][256][3][3],
  float v2524[256][256][3][3],
  float v2525[512][256][3][3],
  float v2526[512][512][3][3],
  float v2527[512][256],
  float v2528[512][512][3][3],
  float v2529[512][512][3][3],
  float v2530[512][10],
  float v2531[64][32][32],
  float v2532[64][32][32],
  float v2533[64][32][32],
  float v2534[64][32][32],
  float v2535[64][32][32],
  float v2536[64][32][32],
  float v2537[64][32][32],
  float v2538[64][32][32],
  float v2539[64][32][32],
  float v2540[64][32][32],
  float v2541[64][32][32],
  float v2542[64][32][32],
  float v2543[64][32][32],
  float v2544[64][32][32],
  float v2545[64][32][32],
  float v2546[64][32][32],
  float v2547[64][32][32],
  float v2548[64][32][32],
  float v2549[64][32][32],
  float v2550[64][32][32],
  float v2551[64][32][32],
  float v2552[64][32][32],
  float v2553[64][32][32],
  float v2554[64][32][32],
  float v2555[128][16][16],
  float v2556[128][16][16],
  float v2557[128][16][16],
  float v2558[128][16][16],
  float v2559[128][16][16],
  float v2560[128][16][16],
  float v2561[128][16][16],
  float v2562[128][16][16],
  float v2563[128][16][16],
  float v2564[128][16][16],
  float v2565[128][16][16],
  float v2566[128][16][16],
  float v2567[128][16][16],
  float v2568[128][16][16],
  float v2569[128][16][16],
  float v2570[128][16][16],
  float v2571[128][16][16],
  float v2572[128][16][16],
  float v2573[128][16][16],
  float v2574[128][16][16],
  float v2575[128][16][16],
  float v2576[128][16][16],
  float v2577[128][16][16],
  float v2578[256][8][8],
  float v2579[256][8][8],
  float v2580[256][8][8],
  float v2581[256][8][8],
  float v2582[256][8][8],
  float v2583[256][8][8],
  float v2584[256][8][8],
  float v2585[256][8][8],
  float v2586[256][8][8],
  float v2587[256][8][8],
  float v2588[256][8][8],
  float v2589[256][8][8],
  float v2590[256][8][8],
  float v2591[256][8][8],
  float v2592[256][8][8],
  float v2593[256][8][8],
  float v2594[256][8][8],
  float v2595[256][8][8],
  float v2596[256][8][8],
  float v2597[256][8][8],
  float v2598[256][8][8],
  float v2599[256][8][8],
  float v2600[256][8][8],
  float v2601[512][4][4],
  float v2602[512][4][4],
  float v2603[512][4][4],
  float v2604[512][4][4],
  float v2605[512][4][4],
  float v2606[512][4][4],
  float v2607[512][4][4],
  float v2608[512][4][4],
  float v2609[512][4][4],
  float v2610[512][4][4],
  float v2611[512][4][4],
  float v2612[512][4][4],
  float v2613[512][4][4],
  float v2614[512][4][4],
  float v2615[512][4][4],
  float v2616[512][4][4],
  float v2617[512][4][4],
  float v2618[512][4][4],
  float v2619[512][4][4],
  float v2620[512][4][4],
  float v2621[512][4][4],
  float v2622[512][4][4]
) {	// L3101
  #pragma HLS interface s_axilite port=return bundle=ctrl

  #pragma HLS interface s_axilite port=v2508 bundle=ctrl
  #pragma HLS interface s_axilite port=v2509 bundle=ctrl
  #pragma HLS interface s_axilite port=v2510 bundle=ctrl
  #pragma HLS interface s_axilite port=v2511 bundle=ctrl
  #pragma HLS interface s_axilite port=v2512 bundle=ctrl
  #pragma HLS interface s_axilite port=v2513 bundle=ctrl
  #pragma HLS interface s_axilite port=v2514 bundle=ctrl
  #pragma HLS interface s_axilite port=v2515 bundle=ctrl
  #pragma HLS interface s_axilite port=v2516 bundle=ctrl
  #pragma HLS interface s_axilite port=v2517 bundle=ctrl
  #pragma HLS interface s_axilite port=v2518 bundle=ctrl
  #pragma HLS interface s_axilite port=v2519 bundle=ctrl
  #pragma HLS interface s_axilite port=v2520 bundle=ctrl
  #pragma HLS interface s_axilite port=v2521 bundle=ctrl
  #pragma HLS interface s_axilite port=v2522 bundle=ctrl
  #pragma HLS interface s_axilite port=v2523 bundle=ctrl
  #pragma HLS interface s_axilite port=v2524 bundle=ctrl
  #pragma HLS interface s_axilite port=v2525 bundle=ctrl
  #pragma HLS interface s_axilite port=v2526 bundle=ctrl
  #pragma HLS interface s_axilite port=v2527 bundle=ctrl
  #pragma HLS interface s_axilite port=v2528 bundle=ctrl
  #pragma HLS interface s_axilite port=v2529 bundle=ctrl
  #pragma HLS interface s_axilite port=v2530 bundle=ctrl
  #pragma HLS interface s_axilite port=v2531 bundle=ctrl
  #pragma HLS interface s_axilite port=v2532 bundle=ctrl
  #pragma HLS interface s_axilite port=v2533 bundle=ctrl
  #pragma HLS interface s_axilite port=v2534 bundle=ctrl
  #pragma HLS interface s_axilite port=v2535 bundle=ctrl
  #pragma HLS interface s_axilite port=v2536 bundle=ctrl
  #pragma HLS interface s_axilite port=v2537 bundle=ctrl
  #pragma HLS interface s_axilite port=v2538 bundle=ctrl
  #pragma HLS interface s_axilite port=v2539 bundle=ctrl
  #pragma HLS interface s_axilite port=v2540 bundle=ctrl
  #pragma HLS interface s_axilite port=v2541 bundle=ctrl
  #pragma HLS interface s_axilite port=v2542 bundle=ctrl
  #pragma HLS interface s_axilite port=v2543 bundle=ctrl
  #pragma HLS interface s_axilite port=v2544 bundle=ctrl
  #pragma HLS interface s_axilite port=v2545 bundle=ctrl
  #pragma HLS interface s_axilite port=v2546 bundle=ctrl
  #pragma HLS interface s_axilite port=v2547 bundle=ctrl
  #pragma HLS interface s_axilite port=v2548 bundle=ctrl
  #pragma HLS interface s_axilite port=v2549 bundle=ctrl
  #pragma HLS interface s_axilite port=v2550 bundle=ctrl
  #pragma HLS interface s_axilite port=v2551 bundle=ctrl
  #pragma HLS interface s_axilite port=v2552 bundle=ctrl
  #pragma HLS interface s_axilite port=v2553 bundle=ctrl
  #pragma HLS interface s_axilite port=v2554 bundle=ctrl
  #pragma HLS interface s_axilite port=v2555 bundle=ctrl
  #pragma HLS interface s_axilite port=v2556 bundle=ctrl
  #pragma HLS interface s_axilite port=v2557 bundle=ctrl
  #pragma HLS interface s_axilite port=v2558 bundle=ctrl
  #pragma HLS interface s_axilite port=v2559 bundle=ctrl
  #pragma HLS interface s_axilite port=v2560 bundle=ctrl
  #pragma HLS interface s_axilite port=v2561 bundle=ctrl
  #pragma HLS interface s_axilite port=v2562 bundle=ctrl
  #pragma HLS interface s_axilite port=v2563 bundle=ctrl
  #pragma HLS interface s_axilite port=v2564 bundle=ctrl
  #pragma HLS interface s_axilite port=v2565 bundle=ctrl
  #pragma HLS interface s_axilite port=v2566 bundle=ctrl
  #pragma HLS interface s_axilite port=v2567 bundle=ctrl
  #pragma HLS interface s_axilite port=v2568 bundle=ctrl
  #pragma HLS interface s_axilite port=v2569 bundle=ctrl
  #pragma HLS interface s_axilite port=v2570 bundle=ctrl
  #pragma HLS interface s_axilite port=v2571 bundle=ctrl
  #pragma HLS interface s_axilite port=v2572 bundle=ctrl
  #pragma HLS interface s_axilite port=v2573 bundle=ctrl
  #pragma HLS interface s_axilite port=v2574 bundle=ctrl
  #pragma HLS interface s_axilite port=v2575 bundle=ctrl
  #pragma HLS interface s_axilite port=v2576 bundle=ctrl
  #pragma HLS interface s_axilite port=v2577 bundle=ctrl
  #pragma HLS interface s_axilite port=v2578 bundle=ctrl
  #pragma HLS interface s_axilite port=v2579 bundle=ctrl
  #pragma HLS interface s_axilite port=v2580 bundle=ctrl
  #pragma HLS interface s_axilite port=v2581 bundle=ctrl
  #pragma HLS interface s_axilite port=v2582 bundle=ctrl
  #pragma HLS interface s_axilite port=v2583 bundle=ctrl
  #pragma HLS interface s_axilite port=v2584 bundle=ctrl
  #pragma HLS interface s_axilite port=v2585 bundle=ctrl
  #pragma HLS interface s_axilite port=v2586 bundle=ctrl
  #pragma HLS interface s_axilite port=v2587 bundle=ctrl
  #pragma HLS interface s_axilite port=v2588 bundle=ctrl
  #pragma HLS interface s_axilite port=v2589 bundle=ctrl
  #pragma HLS interface s_axilite port=v2590 bundle=ctrl
  #pragma HLS interface s_axilite port=v2591 bundle=ctrl
  #pragma HLS interface s_axilite port=v2592 bundle=ctrl
  #pragma HLS interface s_axilite port=v2593 bundle=ctrl
  #pragma HLS interface s_axilite port=v2594 bundle=ctrl
  #pragma HLS interface s_axilite port=v2595 bundle=ctrl
  #pragma HLS interface s_axilite port=v2596 bundle=ctrl
  #pragma HLS interface s_axilite port=v2597 bundle=ctrl
  #pragma HLS interface s_axilite port=v2598 bundle=ctrl
  #pragma HLS interface s_axilite port=v2599 bundle=ctrl
  #pragma HLS interface s_axilite port=v2600 bundle=ctrl
  #pragma HLS interface s_axilite port=v2601 bundle=ctrl
  #pragma HLS interface s_axilite port=v2602 bundle=ctrl
  #pragma HLS interface s_axilite port=v2603 bundle=ctrl
  #pragma HLS interface s_axilite port=v2604 bundle=ctrl
  #pragma HLS interface s_axilite port=v2605 bundle=ctrl
  #pragma HLS interface s_axilite port=v2606 bundle=ctrl
  #pragma HLS interface s_axilite port=v2607 bundle=ctrl
  #pragma HLS interface s_axilite port=v2608 bundle=ctrl
  #pragma HLS interface s_axilite port=v2609 bundle=ctrl
  #pragma HLS interface s_axilite port=v2610 bundle=ctrl
  #pragma HLS interface s_axilite port=v2611 bundle=ctrl
  #pragma HLS interface s_axilite port=v2612 bundle=ctrl
  #pragma HLS interface s_axilite port=v2613 bundle=ctrl
  #pragma HLS interface s_axilite port=v2614 bundle=ctrl
  #pragma HLS interface s_axilite port=v2615 bundle=ctrl
  #pragma HLS interface s_axilite port=v2616 bundle=ctrl
  #pragma HLS interface s_axilite port=v2617 bundle=ctrl
  #pragma HLS interface s_axilite port=v2618 bundle=ctrl
  #pragma HLS interface s_axilite port=v2619 bundle=ctrl
  #pragma HLS interface s_axilite port=v2620 bundle=ctrl
  #pragma HLS interface s_axilite port=v2621 bundle=ctrl
  #pragma HLS interface s_axilite port=v2622 bundle=ctrl

  #pragma HLS interface ap_memory port=v2622
  #pragma HLS stable variable=v2622

  #pragma HLS interface ap_memory port=v2621
  #pragma HLS stable variable=v2621
  #pragma HLS array_partition variable=v2621 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2621 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2620
  #pragma HLS stable variable=v2620
  #pragma HLS array_partition variable=v2620 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2620 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2619
  #pragma HLS stable variable=v2619
  #pragma HLS array_partition variable=v2619 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2619 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2618
  #pragma HLS stable variable=v2618
  #pragma HLS array_partition variable=v2618 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2618 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2617
  #pragma HLS stable variable=v2617

  #pragma HLS interface ap_memory port=v2616
  #pragma HLS stable variable=v2616

  #pragma HLS interface ap_memory port=v2615
  #pragma HLS stable variable=v2615
  #pragma HLS array_partition variable=v2615 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2615 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2614
  #pragma HLS stable variable=v2614
  #pragma HLS array_partition variable=v2614 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2614 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2613
  #pragma HLS stable variable=v2613

  #pragma HLS interface ap_memory port=v2612
  #pragma HLS stable variable=v2612

  #pragma HLS interface ap_memory port=v2611
  #pragma HLS stable variable=v2611
  #pragma HLS array_partition variable=v2611 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2611 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2610
  #pragma HLS stable variable=v2610
  #pragma HLS array_partition variable=v2610 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2610 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2609
  #pragma HLS stable variable=v2609
  #pragma HLS array_partition variable=v2609 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2609 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2608
  #pragma HLS stable variable=v2608
  #pragma HLS array_partition variable=v2608 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2607
  #pragma HLS stable variable=v2607
  #pragma HLS array_partition variable=v2607 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2606
  #pragma HLS stable variable=v2606
  #pragma HLS array_partition variable=v2606 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2606 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2605
  #pragma HLS stable variable=v2605
  #pragma HLS array_partition variable=v2605 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2605 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2604
  #pragma HLS stable variable=v2604
  #pragma HLS array_partition variable=v2604 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2604 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2603
  #pragma HLS stable variable=v2603

  #pragma HLS interface ap_memory port=v2602
  #pragma HLS stable variable=v2602
  #pragma HLS array_partition variable=v2602 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2602 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2601
  #pragma HLS stable variable=v2601
  #pragma HLS array_partition variable=v2601 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2600
  #pragma HLS stable variable=v2600

  #pragma HLS interface ap_memory port=v2599
  #pragma HLS stable variable=v2599
  #pragma HLS array_partition variable=v2599 cyclic factor=4 dim=3


  #pragma HLS interface ap_memory port=v2598
  #pragma HLS stable variable=v2598
  #pragma HLS array_partition variable=v2598 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2598 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2597
  #pragma HLS stable variable=v2597
  #pragma HLS array_partition variable=v2597 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2597 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2596
  #pragma HLS stable variable=v2596
  #pragma HLS array_partition variable=v2596 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2596 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2595
  #pragma HLS stable variable=v2595
  #pragma HLS array_partition variable=v2595 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2595 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2594
  #pragma HLS stable variable=v2594

  #pragma HLS interface ap_memory port=v2593
  #pragma HLS stable variable=v2593

  #pragma HLS interface ap_memory port=v2592
  #pragma HLS stable variable=v2592
  #pragma HLS array_partition variable=v2592 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2592 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2591
  #pragma HLS stable variable=v2591
  #pragma HLS array_partition variable=v2591 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2591 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2590
  #pragma HLS stable variable=v2590

  #pragma HLS interface ap_memory port=v2589
  #pragma HLS stable variable=v2589

  #pragma HLS interface ap_memory port=v2588
  #pragma HLS stable variable=v2588
  #pragma HLS array_partition variable=v2588 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2588 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2587
  #pragma HLS stable variable=v2587
  #pragma HLS array_partition variable=v2587 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2587 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2586
  #pragma HLS stable variable=v2586
  #pragma HLS array_partition variable=v2586 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2586 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2585
  #pragma HLS stable variable=v2585
  #pragma HLS array_partition variable=v2585 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2584
  #pragma HLS stable variable=v2584
  #pragma HLS array_partition variable=v2584 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2583
  #pragma HLS stable variable=v2583
  #pragma HLS array_partition variable=v2583 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2583 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2582
  #pragma HLS stable variable=v2582
  #pragma HLS array_partition variable=v2582 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2582 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2581
  #pragma HLS stable variable=v2581
  #pragma HLS array_partition variable=v2581 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2581 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2580
  #pragma HLS stable variable=v2580

  #pragma HLS interface ap_memory port=v2579
  #pragma HLS stable variable=v2579
  #pragma HLS array_partition variable=v2579 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2579 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2578
  #pragma HLS stable variable=v2578
  #pragma HLS array_partition variable=v2578 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2577
  #pragma HLS stable variable=v2577

  #pragma HLS interface ap_memory port=v2576
  #pragma HLS stable variable=v2576
  #pragma HLS array_partition variable=v2576 cyclic factor=4 dim=3


  #pragma HLS interface ap_memory port=v2575
  #pragma HLS stable variable=v2575
  #pragma HLS array_partition variable=v2575 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2575 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2574
  #pragma HLS stable variable=v2574
  #pragma HLS array_partition variable=v2574 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2574 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2573
  #pragma HLS stable variable=v2573
  #pragma HLS array_partition variable=v2573 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2573 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2572
  #pragma HLS stable variable=v2572
  #pragma HLS array_partition variable=v2572 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2572 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2571
  #pragma HLS stable variable=v2571

  #pragma HLS interface ap_memory port=v2570
  #pragma HLS stable variable=v2570

  #pragma HLS interface ap_memory port=v2569
  #pragma HLS stable variable=v2569
  #pragma HLS array_partition variable=v2569 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2569 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2568
  #pragma HLS stable variable=v2568
  #pragma HLS array_partition variable=v2568 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2568 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2567
  #pragma HLS stable variable=v2567

  #pragma HLS interface ap_memory port=v2566
  #pragma HLS stable variable=v2566

  #pragma HLS interface ap_memory port=v2565
  #pragma HLS stable variable=v2565
  #pragma HLS array_partition variable=v2565 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2565 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2564
  #pragma HLS stable variable=v2564
  #pragma HLS array_partition variable=v2564 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2564 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2563
  #pragma HLS stable variable=v2563
  #pragma HLS array_partition variable=v2563 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2563 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2562
  #pragma HLS stable variable=v2562
  #pragma HLS array_partition variable=v2562 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2561
  #pragma HLS stable variable=v2561
  #pragma HLS array_partition variable=v2561 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2560
  #pragma HLS stable variable=v2560
  #pragma HLS array_partition variable=v2560 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2560 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2559
  #pragma HLS stable variable=v2559
  #pragma HLS array_partition variable=v2559 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2559 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2558
  #pragma HLS stable variable=v2558
  #pragma HLS array_partition variable=v2558 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2558 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2557
  #pragma HLS stable variable=v2557

  #pragma HLS interface ap_memory port=v2556
  #pragma HLS stable variable=v2556
  #pragma HLS array_partition variable=v2556 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2556 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2555
  #pragma HLS stable variable=v2555
  #pragma HLS array_partition variable=v2555 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2554
  #pragma HLS stable variable=v2554

  #pragma HLS interface ap_memory port=v2553
  #pragma HLS stable variable=v2553
  #pragma HLS array_partition variable=v2553 cyclic factor=4 dim=3


  #pragma HLS interface ap_memory port=v2552
  #pragma HLS stable variable=v2552
  #pragma HLS array_partition variable=v2552 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2552 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2551
  #pragma HLS stable variable=v2551
  #pragma HLS array_partition variable=v2551 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2551 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2550
  #pragma HLS stable variable=v2550
  #pragma HLS array_partition variable=v2550 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2550 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2549
  #pragma HLS stable variable=v2549
  #pragma HLS array_partition variable=v2549 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2549 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2548
  #pragma HLS stable variable=v2548
  #pragma HLS array_partition variable=v2548 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2548 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2547
  #pragma HLS stable variable=v2547
  #pragma HLS array_partition variable=v2547 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2547 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2546
  #pragma HLS stable variable=v2546
  #pragma HLS array_partition variable=v2546 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2546 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2545
  #pragma HLS stable variable=v2545
  #pragma HLS array_partition variable=v2545 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2545 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2544
  #pragma HLS stable variable=v2544
  #pragma HLS array_partition variable=v2544 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2544 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2543
  #pragma HLS stable variable=v2543
  #pragma HLS array_partition variable=v2543 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2543 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2542
  #pragma HLS stable variable=v2542
  #pragma HLS array_partition variable=v2542 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2542 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2541
  #pragma HLS stable variable=v2541
  #pragma HLS array_partition variable=v2541 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2541 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2540
  #pragma HLS stable variable=v2540
  #pragma HLS array_partition variable=v2540 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2540 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2539
  #pragma HLS stable variable=v2539
  #pragma HLS array_partition variable=v2539 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2539 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2538
  #pragma HLS stable variable=v2538

  #pragma HLS interface ap_memory port=v2537
  #pragma HLS stable variable=v2537

  #pragma HLS interface ap_memory port=v2536
  #pragma HLS stable variable=v2536
  #pragma HLS array_partition variable=v2536 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2536 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2535
  #pragma HLS stable variable=v2535
  #pragma HLS array_partition variable=v2535 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2535 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2534
  #pragma HLS stable variable=v2534
  #pragma HLS array_partition variable=v2534 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2534 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2533
  #pragma HLS stable variable=v2533
  #pragma HLS array_partition variable=v2533 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2533 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2532
  #pragma HLS stable variable=v2532
  #pragma HLS array_partition variable=v2532 cyclic factor=2 dim=2
  #pragma HLS array_partition variable=v2532 cyclic factor=2 dim=3


  #pragma HLS interface ap_memory port=v2531
  #pragma HLS stable variable=v2531

  #pragma HLS interface ap_memory port=v2530
  #pragma HLS stable variable=v2530

  #pragma HLS interface ap_memory port=v2529
  #pragma HLS stable variable=v2529

  #pragma HLS interface ap_memory port=v2528
  #pragma HLS stable variable=v2528

  #pragma HLS interface ap_memory port=v2527
  #pragma HLS stable variable=v2527

  #pragma HLS interface ap_memory port=v2526
  #pragma HLS stable variable=v2526

  #pragma HLS interface ap_memory port=v2525
  #pragma HLS stable variable=v2525

  #pragma HLS interface ap_memory port=v2524
  #pragma HLS stable variable=v2524

  #pragma HLS interface ap_memory port=v2523
  #pragma HLS stable variable=v2523

  #pragma HLS interface ap_memory port=v2522
  #pragma HLS stable variable=v2522

  #pragma HLS interface ap_memory port=v2521
  #pragma HLS stable variable=v2521

  #pragma HLS interface ap_memory port=v2520
  #pragma HLS stable variable=v2520

  #pragma HLS interface ap_memory port=v2519
  #pragma HLS stable variable=v2519

  #pragma HLS interface ap_memory port=v2518
  #pragma HLS stable variable=v2518

  #pragma HLS interface ap_memory port=v2517
  #pragma HLS stable variable=v2517

  #pragma HLS interface ap_memory port=v2516
  #pragma HLS stable variable=v2516

  #pragma HLS interface ap_memory port=v2515
  #pragma HLS stable variable=v2515

  #pragma HLS interface ap_memory port=v2514
  #pragma HLS stable variable=v2514

  #pragma HLS interface ap_memory port=v2513
  #pragma HLS stable variable=v2513

  #pragma HLS interface ap_memory port=v2512
  #pragma HLS stable variable=v2512

  #pragma HLS interface ap_memory port=v2511
  #pragma HLS stable variable=v2511

  #pragma HLS interface ap_memory port=v2510
  #pragma HLS stable variable=v2510

  #pragma HLS interface ap_memory port=v2509
  #pragma HLS stable variable=v2509

  #pragma HLS interface ap_memory port=v2508
  #pragma HLS stable variable=v2508

  float v2738[10] = {(float)-0.027346, (float)-0.042124, (float)0.011725, (float)-0.042708, (float)0.016996, (float)-0.027074, (float)-0.016846, (float)0.034282, (float)0.014162, (float)0.008224};	// L3332
  #pragma HLS resource variable=v2738 core=ram_t2p_bram

  forward_node152(v2508, v2510, v2531, v2538, v2537);	// L3333
  forward_node145(v2539, v2511, v2532, v2542, v2541);	// L3334
  forward_node136(v2540, v2512, v2533, v2543, v2548, v2544, v2547);	// L3335
  forward_node129(v2545, v2513, v2534, v2550, v2549);	// L3336
  forward_node121(v2546, v2514, v2551, v2536, v2552, v2535);	// L3337
  forward_node114(v2553, v2515, v2555, v2562, v2561);	// L3338
  forward_node107(v2563, v2556, v2516, v2565, v2564);	// L3339
  forward_node98(v2554, v2566, v2557, v2517, v2571, v2570, v2567);	// L3340
  forward_node91(v2568, v2518, v2558, v2573, v2572);	// L3341
  forward_node83(v2569, v2574, v2519, v2560, v2559, v2575);	// L3342
  forward_node76(v2520, v2578, v2576, v2585, v2584);	// L3343
  forward_node69(v2586, v2579, v2521, v2588, v2587);	// L3344
  forward_node60(v2522, v2589, v2580, v2577, v2594, v2593, v2590);	// L3345
  forward_node53(v2523, v2591, v2581, v2596, v2595);	// L3346
  forward_node45(v2524, v2592, v2597, v2583, v2598, v2582);	// L3347
  forward_node38(v2599, v2525, v2601, v2608, v2607);	// L3348
  forward_node31(v2602, v2609, v2526, v2611, v2610);	// L3349
  forward_node22(v2600, v2603, v2612, v2527, v2617, v2613, v2616);	// L3350
  forward_node15(v2614, v2604, v2528, v2619, v2618);	// L3351
  forward_node7(v2529, v2615, v2620, v2606, v2621, v2605);	// L3352
  float v2739[512];	// L3353
  #pragma HLS resource variable=v2739 core=ram_t2p_bram

  forward_node4(v2622, v2739);	// L3354
  forward_node0(v2738, v2739, v2530, v2509);	// L3355
}

