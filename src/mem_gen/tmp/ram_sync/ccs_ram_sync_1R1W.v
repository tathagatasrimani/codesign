module ccs_ram_sync_1R1W
#(
parameter data_width = 8,
parameter addr_width = 7,
parameter depth = 128
)(
	radr, wadr, d, we, re, clk, q
);

	input [addr_width-1:0] radr;
	input [addr_width-1:0] wadr;
	input [data_width-1:0] d;
	input we;
	input re;
	input clk;
	output[data_width-1:0] q;

   // synopsys translate_off
	reg [data_width-1:0] q;

	reg [data_width-1:0] mem [depth-1:0];
		
	always @(posedge clk) begin
		if (we) begin
			mem[wadr] <= d; // Write port
		end
		if (re) begin
			q <= mem[radr] ; // read port
		end
	end
   // synopsys translate_on

endmodule
