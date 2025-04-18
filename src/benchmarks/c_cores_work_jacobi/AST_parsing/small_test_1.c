void compute(int a, int b, int c, int *out) {
    int sum;  // Should be a register
    sum = a + b;  // Catapult normally infers sum as a register
    *out = sum * c;
}
