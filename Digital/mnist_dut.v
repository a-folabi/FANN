// ============================================================================
// Top-Level Module: 2-Layer MNIST Accelerator (L1 + L2 Argmax)
// - Layer 1 computes 64 -> 100, ReLU + requant to uint8
// - Layer 2 computes 100 -> 10 int32 logits, adds bias, then argmax
// - Outputs: class_idx[3:0] and class_val[31:0] for debug
// ============================================================================
module mnist2layer_accelerator (
    input                       clk,
    input                       reset,

    // --- Host Control & L1 Requantization ---
    input                       start,
    input       [15:0]          M1,
    input       [4:0]           S1,
    output reg                  busy,
    output reg                  done,

    // --- Progress Monitoring (L1) ---
    output wire                 tile_mac_done_pulse,

    // --- Host Memory Interface ---
    input                       host_wren,
    input       [19:0]          host_addr,
    input       [31:0]          host_din,
    output      [31:0]          host_dout,

    // --- Final classification results (for debug/host) ---
    output reg  [3:0]           class_idx,
    output reg  signed [31:0]   class_val
);

    // -----------------------------
    // Common parameters and bases
    // -----------------------------
    localparam A_IN_BASE        = 10'd0;
    localparam HID_BASE         = 10'd128;
    localparam NUM_TILES        = 10;
    localparam K_L1             = 64;
    localparam NEURONS_PER_TILE = 10;

    // L2 params
    localparam K_L2             = 100;        // hidden size
    localparam L2_OUT           = 10;         // number of classes

    // Host windows kept from L1
    localparam ACT_MEM_H_BASE   = 20'h00000;
    localparam W1_B0_H_BASE     = 20'h01000;  // 10 banks, each 0x1000
    localparam B1_MEM_H_BASE    = 20'h0B000;  // 100 * 32-bit

    // New host windows for L2
    localparam W2_B0_H_BASE     = 20'h0C000;  // 10 banks, each 0x1000, addr = k (0..99)
    localparam B2_MEM_H_BASE    = 20'h16000;  // 10 * 32-bit

    // -----------------------------
    // Controller FSM states
    // -----------------------------
    localparam [4:0]
        S_IDLE           = 5'd0,
        // L1
        S_L1_SETUP_TILE  = 5'd1,
        S_L1_LOAD_BIAS   = 5'd2,
        S_L1_PRIME       = 5'd3,
        S_L1_RUN_K       = 5'd4,
        S_L1_POST        = 5'd5,
        S_L1_WRITE       = 5'd6,
        S_L1_NEXT_TILE   = 5'd7,
        // L2
        S_L2_SETUP       = 5'd8,
        S_L2_LOAD_BIAS   = 5'd9,
        S_L2_PRIME       = 5'd10,
        S_L2_RUN_K       = 5'd11,
        S_L2_POST        = 5'd12,
        S_L2_ARGMAX      = 5'd13,
        // final
        S_DONE           = 5'd14;

    reg [4:0] state, next_state;

    // -----------------------------
    // L1 bookkeeping
    // -----------------------------
    reg [6:0] k1_counter;
    reg [3:0] tile_counter;
    reg [3:0] write_counter;
    reg [3:0] bias1_load_counter;

    wire signed [31:0] pe1_out [0:NEURONS_PER_TILE-1];
    wire               pe1_done[0:NEURONS_PER_TILE-1];
    wire signed [7:0]  pp1_out [0:NEURONS_PER_TILE-1];
    wire               pp1_valid[0:NEURONS_PER_TILE-1];

    reg                pe1_clear;
    reg                pp1_start;

    reg signed [31:0]  bias1_for_pe [0:NEURONS_PER_TILE-1];
    reg [NEURONS_PER_TILE-1:0] pp1_seen;
    wire pp1_valid_all = &pp1_seen;

    // -----------------------------
    // L2 bookkeeping
    // -----------------------------
    reg [6:0]          k2_counter;         // 0..99
    reg [3:0]          bias2_load_counter; // 0..9
    reg                pe2_clear;

    wire signed [31:0] pe2_out [0:L2_OUT-1];
    wire               pe2_done[0:L2_OUT-1];
    reg  signed [31:0] bias2_for_pe [0:L2_OUT-1];

    reg [L2_OUT-1:0]   pe2_seen;
    wire pe2_all_done = &pe2_seen;

    // Argmax scan
    reg [3:0]          amax_idx;
    reg signed [31:0]  amax_val;

    // -----------------------------
    // Shared memories
    // -----------------------------

    // ACT_MEM is shared by L1 and L2. Keep single port and arbitrate by state.
    reg                 act_mem_wren;
    reg  [9:0]          act_mem_addr;
    reg  [7:0]          act_mem_din;
    wire [7:0]          act_mem_dout;

    // L1 weights and bias memories
    reg  [9:0]          w1_mem_addr;
    wire [7:0]          w1_mem_dout [0:NEURONS_PER_TILE-1];
    reg  [6:0]          b1_mem_addr;
    wire [31:0]         b1_mem_dout;

    // L2 weights and bias memories
    reg  [6:0]          w2_mem_addr;   // 0..99
    wire [7:0]          w2_mem_dout [0:L2_OUT-1];
    reg  [3:0]          b2_mem_addr;   // 0..9
    wire [31:0]         b2_mem_dout;

    // Broadcast activation buses
    reg   [7:0]   broadcast_act_l1;
    reg   [7:0]   broadcast_act_l2;

    // Progress pulse for L1
    assign tile_mac_done_pulse = (state == S_L1_POST) && (k1_counter == 0);

    // -----------------------------
    // Memories
    // -----------------------------
    // ACT_MEM: 1KB 8-bit
    spram #( .AWIDTH(10), .NUM_ROWS(1024), .NUM_COLS(8) ) act_mem (
        .clk(clk),
        .addr(busy ? act_mem_addr : host_addr[9:0]),
        .wren(busy ? act_mem_wren : (host_wren && host_addr[19:10] == ACT_MEM_H_BASE[19:10])),
        .data_in(busy ? act_mem_din : host_din[7:0]),
        .data_out(act_mem_dout)
    );
    assign host_dout = {24'b0, act_mem_dout}; // readback of ACT_MEM bytes

    // W1 banks: 10 x 1KB 8-bit
    genvar i;
    generate
        for (i = 0; i < NEURONS_PER_TILE; i = i + 1) begin : w1_banks
            spram #( .AWIDTH(10), .NUM_ROWS(1024), .NUM_COLS(8) ) w1_bank_inst (
                .clk(clk),
                .addr(busy ? w1_mem_addr : host_addr[9:0]),
                .wren(host_wren && (host_addr[19:12] == (W1_B0_H_BASE[19:12] + i))),
                .data_in(host_din[7:0]),
                .data_out(w1_mem_dout[i])
            );
        end
    endgenerate

    // B1 memory: 128 x 32-bit
    spram #( .AWIDTH(7), .NUM_ROWS(128), .NUM_COLS(32) ) b1_mem (
        .clk(clk),
        .addr(busy ? b1_mem_addr : host_addr[6:0]),
        .wren(host_wren && host_addr[19:7] == B1_MEM_H_BASE[19:7]),
        .data_in(host_din),
        .data_out(b1_mem_dout)
    );

    // W2 banks: 10 x 1KB 8-bit window
    generate
        for (i = 0; i < L2_OUT; i = i + 1) begin : w2_banks
            spram #( .AWIDTH(10), .NUM_ROWS(1024), .NUM_COLS(8) ) w2_bank_inst (
                .clk(clk),
                .addr(busy ? {3'b0, w2_mem_addr} : host_addr[9:0]), // K_L2 <= 100
                .wren(host_wren && (host_addr[19:12] == (W2_B0_H_BASE[19:12] + i))),
                .data_in(host_din[7:0]),
                .data_out(w2_mem_dout[i])
            );
        end
    endgenerate

    // B2 memory: 16 x 32-bit window
    spram #( .AWIDTH(4), .NUM_ROWS(16), .NUM_COLS(32) ) b2_mem (
        .clk(clk),
        .addr(busy ? b2_mem_addr : host_addr[3:0]),
        .wren(host_wren && host_addr[19:4] == B2_MEM_H_BASE[19:4]),
        .data_in(host_din),
        .data_out(b2_mem_dout)
    );

    // -----------------------------
    // Compute blocks
    // -----------------------------
    // L1 PEs + post processing
    generate
        for (i = 0; i < NEURONS_PER_TILE; i = i + 1) begin : l1_compute
            processing_element pe1_inst (
                .clk(clk), .reset(reset), .clear(pe1_clear),
                .k_limit(K_L1), .pe_col(i),
                .row_bias(bias1_for_pe[i]),
                .in_act(broadcast_act_l1),
                .in_wt(w1_mem_dout[i]),
                .pe_out(pe1_out[i]), .pe_done(pe1_done[i])
            );
            l1_pp pp1_inst (
                .clk(clk), .reset(reset), .M1(M1), .S1(S1),
                .pe_done(pe1_done[i]),
                .data_in(pe1_out[i]),
                .out_valid(pp1_valid[i]),
                .relu8_u(pp1_out[i])
            );
        end
    endgenerate

    // L2 PEs (bias added in the PE, no ReLU or requant here)
    generate
        for (i = 0; i < L2_OUT; i = i + 1) begin : l2_compute
            processing_element pe2_inst (
                .clk(clk), .reset(reset), .clear(pe2_clear),
                .k_limit(K_L2), .pe_col(i),
                .row_bias(bias2_for_pe[i]),
                .in_act(broadcast_act_l2),
                .in_wt(w2_mem_dout[i]),
                .pe_out(pe2_out[i]), .pe_done(pe2_done[i])
            );
        end
    endgenerate

    // -----------------------------
    // Seen pulse trackers
    // -----------------------------
    // L1 pp_valid collator
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            pp1_seen <= {NEURONS_PER_TILE{1'b0}};
        end else begin
            if (state == S_L1_SETUP_TILE)
                pp1_seen <= {NEURONS_PER_TILE{1'b0}};
            else if (state == S_L1_POST) begin
                integer j;
                for (j = 0; j < NEURONS_PER_TILE; j = j + 1)
                    if (pp1_valid[j]) pp1_seen[j] <= 1'b1;
            end
        end
    end

    // L2 pe_done collator
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            pe2_seen <= {L2_OUT{1'b0}};
        end else begin
            if (state == S_L2_SETUP)
                pe2_seen <= {L2_OUT{1'b0}};
            else if (state == S_L2_POST) begin
                integer j2;
                for (j2 = 0; j2 < L2_OUT; j2 = j2 + 1)
                    if (pe2_done[j2]) pe2_seen[j2] <= 1'b1;
            end
        end
    end

    // -----------------------------
    // Controller FSM
    // -----------------------------
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= S_IDLE;
            busy  <= 1'b0; done <= 1'b0;
            class_idx <= 0; class_val <= 0;
            k1_counter <= 0; tile_counter <= 0; write_counter <= 0; bias1_load_counter <= 0;
            k2_counter <= 0; bias2_load_counter <= 0;
            amax_idx <= 0; amax_val <= 0;
        end else begin
            state <= next_state;

            // Busy/done protocol for the full 2-layer run
            if (state == S_IDLE && next_state != S_IDLE) begin
                busy <= 1'b1; done <= 1'b0;
            end else if (state == S_DONE && next_state == S_IDLE) begin
                busy <= 1'b0; done <= 1'b1;
            end

            // L1 counters
            if (next_state == S_L1_SETUP_TILE) tile_counter <= (state == S_IDLE) ? 0 : tile_counter + 1;
            if (next_state == S_L1_LOAD_BIAS)  bias1_load_counter <= bias1_load_counter + 1; else bias1_load_counter <= 0;
            if (next_state == S_L1_RUN_K)      k1_counter <= k1_counter + 1; else k1_counter <= 0;
            if (state == S_L1_WRITE)           write_counter <= write_counter + 1; else write_counter <= 0;

            // Capture B1 into per-PE hold regs on the cycle after address present
            if (state == S_L1_LOAD_BIAS && bias1_load_counter > 0) begin
                bias1_for_pe[bias1_load_counter-1] <= b1_mem_dout;
            end

            // L2 counters
            if (next_state == S_L2_LOAD_BIAS)  bias2_load_counter <= bias2_load_counter + 1; else bias2_load_counter <= 0;
            if (next_state == S_L2_RUN_K)      k2_counter <= k2_counter + 1; else k2_counter <= 0;

            // Capture B2
            if (state == S_L2_LOAD_BIAS && bias2_load_counter > 0) begin
                bias2_for_pe[bias2_load_counter-1] <= b2_mem_dout;
            end

            // Argmax scan updates
            if (state == S_L2_ARGMAX) begin
                if (amax_idx == 0) begin
                    amax_val <= pe2_out[0];
                    class_idx <= 0;
                    amax_idx <= 1;
                end else begin
                    if ($signed(pe2_out[amax_idx]) > $signed(amax_val)) begin
                        amax_val <= pe2_out[amax_idx];
                        class_idx <= amax_idx[3:0];
                    end
                    amax_idx <= amax_idx + 1;
                end
                // latch final value at exit to DONE
                if (next_state == S_DONE) class_val <= amax_val;
            end else begin
                amax_idx <= 0;
            end
        end
    end

    // -----------------------------
    // FSM outputs and address muxing
    // -----------------------------
    always @(*) begin
        next_state = state;

        // defaults
        pe1_clear = 1'b0; pp1_start = 1'b0;
        pe2_clear = 1'b0;

        act_mem_wren  = 1'b0;
        act_mem_addr  = 10'b0;
        act_mem_din   = 8'b0;

        w1_mem_addr   = 10'b0;
        b1_mem_addr   = 7'b0;
        w2_mem_addr   = 7'd0;
        b2_mem_addr   = 4'd0;

        broadcast_act_l1 = 8'sd0;
        broadcast_act_l2 = 8'sd0;

        case (state)
            // ----------------- IDLE -----------------
            S_IDLE: begin
                if (start) next_state = S_L1_SETUP_TILE;
            end

            // ----------------- L1 -----------------
            S_L1_SETUP_TILE: begin
                pe1_clear   = 1'b1;
                next_state  = S_L1_LOAD_BIAS;
                b1_mem_addr = {3'b0, tile_counter} * NEURONS_PER_TILE;
            end

            S_L1_LOAD_BIAS: begin
                b1_mem_addr = {3'b0, tile_counter} * NEURONS_PER_TILE + {3'b0, bias1_load_counter};
                if (bias1_load_counter == NEURONS_PER_TILE) begin
                    next_state = S_L1_PRIME;
                end
            end

            S_L1_PRIME: begin
                pe1_clear      = 1'b1;
                act_mem_addr   = A_IN_BASE + 10'd0;
                w1_mem_addr    = {3'b0, tile_counter} * K_L1 + 10'd0;
                next_state     = S_L1_RUN_K;
            end

            S_L1_RUN_K: begin
                broadcast_act_l1 = act_mem_dout;
                act_mem_addr   = A_IN_BASE + {3'b0, k1_counter};
                w1_mem_addr    = {3'b0, tile_counter} * K_L1 + {3'b0, k1_counter};
                if (k1_counter == K_L1-1) next_state = S_L1_POST;
            end

            S_L1_POST: begin
                pp1_start = 1'b1;
                broadcast_act_l1 = act_mem_dout;
                if (pp1_valid_all) next_state = S_L1_WRITE;
            end

            S_L1_WRITE: begin
                act_mem_wren = 1'b1;
                act_mem_addr = HID_BASE + ({6'b0, tile_counter} * NEURONS_PER_TILE) + {6'b0, write_counter};
                act_mem_din  = pp1_out[write_counter];
                if (write_counter == NEURONS_PER_TILE - 1) next_state = S_L1_NEXT_TILE;
            end

            S_L1_NEXT_TILE: begin
                if (tile_counter == NUM_TILES - 1) next_state = S_L2_SETUP;
                else next_state = S_L1_SETUP_TILE;
            end

            // ----------------- L2 -----------------
            S_L2_SETUP: begin
                pe2_clear   = 1'b1;
                next_state  = S_L2_LOAD_BIAS;
                b2_mem_addr = 4'd0;
            end

            S_L2_LOAD_BIAS: begin
                b2_mem_addr = bias2_load_counter;
                if (bias2_load_counter == L2_OUT) begin
                    next_state = S_L2_PRIME;
                end
            end

            S_L2_PRIME: begin
                pe2_clear     = 1'b1;
                act_mem_addr  = HID_BASE + 10'd0;
                w2_mem_addr   = 7'd0;
                next_state    = S_L2_RUN_K;
            end

            S_L2_RUN_K: begin
                broadcast_act_l2 = act_mem_dout;
                act_mem_addr  = HID_BASE + {3'b0, k2_counter}; // 0..99
                w2_mem_addr   = k2_counter;
                if (k2_counter == K_L2-1) next_state = S_L2_POST;
            end

            S_L2_POST: begin
                // wait until all ten PEs have raised pe2_done at least once
                if (pe2_all_done) next_state = S_L2_ARGMAX;
            end

            S_L2_ARGMAX: begin
                // we iterate amax_idx from 0..9 inside the sequential block
                if (amax_idx == L2_OUT-1) next_state = S_DONE;
            end

            // ----------------- DONE -----------------
            S_DONE: begin
                next_state = S_IDLE;
            end

            default: next_state = S_IDLE;
        endcase
    end

endmodule

// ============================================================================
// Processing Element (shared by L1 and L2)
// Accumulates int8*int8 over K, then adds bias once, then outputs and pulses pe_done
// ============================================================================
module processing_element (
    input clk, input reset, input clear,
    input [6:0] k_limit, input [3:0] pe_col,
    input signed [31:0] row_bias,
    input        [7:0] in_act, 
    input signed [7:0] in_wt,
    output reg signed [31:0] pe_out, output reg pe_done
);
    reg signed [31:0] acc32;
    reg [6:0] pe_cnt;
    wire signed [15:0] prod;
    qmult mult_u0(.i_multiplicand(in_act), .i_multiplier(in_wt), .o_result(prod));
    wire signed [31:0] prod_ext = {{16{prod[15]}}, prod};
    wire do_mac = (pe_cnt < k_limit);
    always @(posedge clk) begin
        if (reset || clear) begin
            pe_cnt <= 0; acc32 <= 0; pe_done <= 0; pe_out <= 0;
        end else if (do_mac) begin
            acc32 <= acc32 + prod_ext;
            pe_cnt <= pe_cnt + 1;
        end else if (pe_cnt == k_limit) begin
            pe_cnt <= pe_cnt + 1;
            acc32 <= acc32 + row_bias;
        end else begin
            pe_done <= 1; pe_out <= acc32;
        end
    end
endmodule

// ============================================================================
// 8x8 -> 16 unsigned x signed multiplier
// ============================================================================
module qmult(i_multiplicand, i_multiplier, o_result);
    input        [7:0] i_multiplicand;
    input signed [7:0] i_multiplier;
    output signed [15:0] o_result;
    assign o_result = $signed({1'b0, i_multiplicand}) * i_multiplier;
endmodule

// ============================================================================
// Layer 1 post-processing: ReLU, then requantize via (x*M1)>>S1, clamp to [0,255]
// ============================================================================
module l1_pp (
    input               clk,
    input               reset,
    input       [15:0]  M1,
    input       [4:0]   S1,
    input               pe_done,
    input signed [31:0] data_in,
    output reg          out_valid,
    output reg      [7:0] relu8_u
);
    reg signed [31:0] relu_out;
    reg signed [31:0] quant_prod;
    reg phase1 = 1'd0;
    reg phase2 = 1'd0;
    reg pe_done_q;
    reg arm;

    wire signed [31:0] q_shifted = quant_prod >>> S1; // arithmetic shift

    always @(posedge clk) begin
        if (reset) begin
            out_valid <= 0; relu8_u <= 0;
            relu_out <= 0; quant_prod <= 0;
            phase1 <= 0; phase2 <= 0;
            pe_done_q <= 0; arm <= 0;
        end else begin
            pe_done_q <= pe_done;
            out_valid <= 0;

            if (!phase1 && !phase2) begin
                if (arm) begin
                    relu_out <= (data_in < 32'sd0) ? 32'd0 : data_in;
                    arm <= 0;
                    phase1 <= 1;
                end else if (pe_done && !pe_done_q) begin
                    arm <= 1;
                end
            end else if (phase1) begin
                quant_prod <= relu_out * $signed({1'b0, M1}); // M1 is treated as unsigned in spec
                phase1 <= 0; phase2 <= 1;
            end else if (phase2) begin
                phase2 <= 0;
                if (q_shifted > 32'sd255) relu8_u <= 8'd255;
                else if (q_shifted < 0)   relu8_u <= 8'd0;
                else                      relu8_u <= q_shifted[7:0];
                out_valid <= 1;
            end
        end
    end
endmodule

// ============================================================================
// Simple single-port RAM
// ============================================================================
module spram #(
    parameter AWIDTH=6, parameter NUM_ROWS=64, parameter NUM_COLS=32
)(
    input clk, input [AWIDTH-1:0] addr, input wren,
    input [NUM_COLS-1:0] data_in, output reg [NUM_COLS-1:0] data_out
);
    (* ram_style = "block" *) reg [NUM_COLS-1:0] ram[NUM_ROWS-1:0];
    always @(posedge clk) begin
        if (wren) begin
            ram[addr] <= data_in;
        end
        data_out <= ram[addr];
    end
endmodule
