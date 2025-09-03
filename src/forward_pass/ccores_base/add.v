module add (
    input  [15:0] a, 
    input  [15:0] b, 
    output [15:0] z
);
    assign z = a + b;
endmodule