import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Timer
import numpy as np

# --- Constants matching the Verilog module ---
# Memory Addressing
A_IN_BASE = 0
HID_BASE = 128
NUM_NEURONS = 100
K_L1 = 64
NEURONS_PER_TILE = 10
NUM_TILES = 10

# Host Memory Map
ACT_MEM_H_BASE = 0x00000
W1_B0_H_BASE = 0x01000  # Base address for the first weight bank
W1_BANK_ADDR_SPACE = 0x1000  # Each bank occupies 4KB (0x1000)
B1_MEM_H_BASE = 0x0B000

W2_B0_H_BASE = 0x0C000
W2_BANK_ADDR_SPACE = 0x1000
B2_MEM_H_BASE = 0x16000


# --- Helper Functions ---

async def reset_dut(dut):
    dut.reset.value = 1
    await ClockCycles(dut.clk, 2)
    dut.reset.value = 0
    dut.start.value = 0
    dut.host_wren.value = 0
    dut.host_addr.value = 0
    dut.host_din.value = 0
    await ClockCycles(dut.clk, 1)
    dut._log.info("DUT reset complete.")

async def host_write(dut, addr, data):
    await RisingEdge(dut.clk)
    dut.host_wren.value = 1
    dut.host_addr.value = addr
    dut.host_din.value = data
    await RisingEdge(dut.clk)
    dut.host_wren.value = 0

async def host_read(dut, addr):
    await RisingEdge(dut.clk)
    dut.host_wren.value = 0
    dut.host_addr.value = addr
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    return int(dut.host_dout.value)

async def preload_weights_and_biases(dut, w1, b1, w2, b2):
    # Map layer 1 weights to tiles and banks
    for neuron_idx in range(NUM_NEURONS):
        tile_idx = neuron_idx // NEURONS_PER_TILE
        bank_idx = neuron_idx % NEURONS_PER_TILE
        for weight_idx in range(K_L1):
            internal_addr = tile_idx * K_L1 + weight_idx
            bank_base_addr = W1_B0_H_BASE + (bank_idx * W1_BANK_ADDR_SPACE)
            host_addr = bank_base_addr + internal_addr
            data = int(w1[neuron_idx][weight_idx])
            await host_write(dut, host_addr, data)
    dut._log.info(f"Loaded {NUM_NEURONS * K_L1} weights into W1 banks.")

    for i in range(NUM_NEURONS):
        await host_write(dut, B1_MEM_H_BASE + i, int(b1[i]))
    dut._log.info(f"Loaded {NUM_NEURONS} biases into B1 memory.")

    # Layer 2 weights: 10 banks, one per class
    for cls in range(10):
        bank_base_addr = W2_B0_H_BASE + cls * W2_BANK_ADDR_SPACE
        for k in range(NUM_NEURONS):
            await host_write(dut, bank_base_addr + k, int(w2[cls][k]))
    dut._log.info("Loaded 1000 weights into W2 banks.")

    for cls in range(10):
        await host_write(dut, B2_MEM_H_BASE + cls, int(b2[cls]))
    dut._log.info("Loaded 10 biases into B2 memory.")

async def run_one_sample(dut, act_u8):
    # Write 64 bytes of uint8 activations into ACT_MEM
    for i in range(K_L1):
        await host_write(dut, ACT_MEM_H_BASE + A_IN_BASE + i, int(act_u8[i]))

    # Kick DUT
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for completion
    await RisingEdge(dut.done)
    # Optional settle
    await ClockCycles(dut.clk, 2)

    # Read final classification from top level outputs
    pred_idx = int(dut.class_idx.value)
    pred_val = int(dut.class_val.value.signed_integer)
    return pred_idx, pred_val

def ref_int_pipeline_preds(test_u8, w1, b1, w2, b2, M1, S1):
    # Fallback software integer reference if pytorch_test_pred is not present
    acts_u8 = test_u8.astype(np.uint8).astype(np.int32)             # N x 64               
    w1_i32 = w1.astype(np.int32)                                     # 100 x 64
    b1_i32 = b1.astype(np.int32)                                     # 100
    # Layer 1
    y1 = acts_u8 @ w1_i32.T                                          # N x 100
    y1 = y1 + b1_i32                                                 # broadcast
    y1 = np.maximum(y1, 0)                                           # ReLU
    # Requant to uint8 with fixed point
    quant_prod32 = (y1.astype(np.int64) * int(M1)).astype(np.int32)  # wrap to int32
    q_shifted = (quant_prod32 >> int(S1))                             # arithmetic shift
    a1_u8 = np.clip(q_shifted, 0, 255).astype(np.uint8)              # N x 100
    # Layer 2 logits
    logits = a1_u8.astype(np.int32) @ w2.astype(np.int32).T
    logits = logits + b2.astype(np.int32)
    preds = np.argmax(logits, axis=1).astype(np.int64)
    return preds

def calc_m1_s1(s_in, s_w, s_out, s1=24):
    scale = (float(s_in) * float(s_w)) / float(s_out)
    m1 = int(np.round(scale * (1 << s1)))
    # clip to your RTL width if needed
    m1 = max(0, min((1<<16)-1, m1))
    return m1, s1

# --- Main Testbench ---

@cocotb.test()
async def test_full_testset_accuracy(dut):
    """
    Loads weights, biases, and the packed test set from the npz.
    Streams all test samples through the DUT and reports:
      - PyTorch accuracy (from npz)
      - FPGA accuracy
      - Count of samples where FPGA and PyTorch predicted class indices differ
    """
    dut._log.info("Starting test_full_testset_accuracy...")

    # 1) Clock and reset
    clock = Clock(dut.clk, 10, units="ns")  # 100 MHz
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # 2) Load params and test set from npz
    npz = np.load("mnist8x8_int8W_int32B_uint8A.npz")
    w1 = np.array(npz["mlp_0.weight_q"], dtype=np.int8)     # [100, 64]
    b1 = np.array(npz["mlp_0.bias_q"], dtype=np.int32)      # [100]
    w2 = np.array(npz["mlp_1.weight_q"], dtype=np.int8)     # [10, 100]
    b2 = np.array(npz["mlp_1.bias_q"], dtype=np.int32)      # [10]

    if "test_inputs_u8" not in npz or "test_labels" not in npz:
        raise RuntimeError("npz is missing test_inputs_u8 or test_labels")

    test_u8 = np.array(npz["test_inputs_u8"], dtype=np.uint8)  # [N, 64]
    test_labels = np.array(npz["test_labels"], dtype=np.int64) # [N]
    num_samples = int(test_u8.shape[0])
    assert test_u8.shape[1] == K_L1, "Packed test vectors must be 64 long"

    # Optional PyTorch predictions from training script
    pytorch_preds = None
    '''if "pytorch_test_pred" in npz:
        pytorch_preds = np.array(npz["pytorch_test_pred"], dtype=np.int64)
        if pytorch_preds.shape[0] != num_samples:
            dut._log.warning("pytorch_test_pred length does not match test set, will recompute a reference instead")
            pytorch_preds = None'''

    # 3) Preload weights and biases once
    dut._log.info("Preloading network params...")
    await preload_weights_and_biases(dut, w1, b1, w2, b2)

    # 4) Program fixed point params and run all samples
    s_in  = float(npz["mlp_0.in_scale"][0])
    s_w   = float(npz["mlp_0.weight_scale"][0])
    s_out = float(npz["mlp_0.out_scale"][0])
    M1_CONST, S1_CONST = calc_m1_s1(s_in, s_w, s_out)
    dut.M1.value = M1_CONST
    dut.S1.value = S1_CONST

    fpga_preds = np.zeros(num_samples, dtype=np.int64)

    dut._log.info(f"Running {num_samples} test samples through the DUT...")
    for i in range(num_samples):
        pred_idx, pred_val = await run_one_sample(dut, test_u8[i])
        fpga_preds[i] = pred_idx

        # If your done signal is level rather than pulsed and holds high, wait for it to drop
        # so the next start pulse is seen. This loop is safe even if done is a pulse.
        if int(dut.done.value) != 0:
            # Wait until done deasserts or give it a couple of cycles
            for _ in range(4):
                await ClockCycles(dut.clk, 1)
                if int(dut.done.value) == 0:
                    break

        if (i + 1) % 50 == 0 or i == num_samples - 1:
            dut._log.info(f"Progress {i+1}/{num_samples}")

    # 5) Metrics
    fpga_acc = float(np.mean(fpga_preds == test_labels))

    if pytorch_preds is None:
        dut._log.warning("pytorch_test_pred not found. Computing a software integer reference as a proxy for comparison.")
        pytorch_preds = ref_int_pipeline_preds(test_u8, w1, b1, w2, b2, M1_CONST, S1_CONST)

    torch_acc = float(np.mean(pytorch_preds == test_labels))
    disagree_count = int(np.sum(fpga_preds != pytorch_preds))

    # 6) Log results
    dut._log.info(f"Samples tested: {num_samples}")
    dut._log.info(f"PyTorch accuracy: {torch_acc * 100.0:.2f}%")
    dut._log.info(f"FPGA accuracy: {fpga_acc * 100.0:.2f}%")
    dut._log.info(f"Disagreements (FPGA idx vs PyTorch idx): {disagree_count} of {num_samples}")
