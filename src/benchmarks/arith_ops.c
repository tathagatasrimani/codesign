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