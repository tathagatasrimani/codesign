module mult (
    input  [15:0] a, 
    input  [15:0] b, 
    input  [15:0] tag,
    output [15:0] z
);
    assign z = a * b;
endmodule