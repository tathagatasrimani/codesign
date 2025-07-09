module ccs_ram_sync_1R1W
#(
    parameter data_width = 8,
    parameter addr_width = 7,
    parameter depth = 128
)(
    input [addr_width-1:0] radr,
    input [addr_width-1:0] wadr,
    input [data_width-1:0] d,
    input we,
    input re,
    input clk,
    output reg [data_width-1:0] q
);

    reg [data_width-1:0] mem [depth-1:0];

    always @(posedge clk) begin
        // Write port 1
        if (we) begin
            mem[wadr] <= d;
        end
        // Read port 1
        if (re) begin
            q <= mem[radr];
        end
    end

endmodule