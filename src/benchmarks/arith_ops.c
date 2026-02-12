float addf(float a, float b) {
  return a + b;
}

float mulf(float a, float b) {
return a * b;
}

float subf(float a, float b) {
  return a - b;
}

float divf(float a, float b) {
  return a / b;
}

float exp_bb(float a) {
  return a * 2; // putting exp(a) here causes an unsupported instruction to be generated. This is just a placeholder anyways.
}

unsigned short lshift_bb(unsigned short a, unsigned short b) {
  return (unsigned short)(a << (b & 0xF));
}

float addf_ctrl_chain(float a, float b) {
  return a + b;
}

float mulf_ctrl_chain(float a, float b) {
return a * b;
}

float subf_ctrl_chain(float a, float b) {
  return a - b;
}

float divf_ctrl_chain(float a, float b) {
  return a / b;
}

float exp_bb_ctrl_chain(float a) {
  return a * 2; // putting exp(a) here causes an unsupported instruction to be generated. This is just a placeholder anyways.
}

unsigned short lshift_bb_ctrl_chain(unsigned short a, unsigned short b) {
  return (unsigned short)(a << (b & 0xF));
}