# Full Analog Neural Networks
Files for reproducing analog and digital artifacts used in the paper titled "On the Viability of End-to-End Neural Networks in the Analog Domain"

## Analog
- The requirements for running and reproducing analog items are `python 3.10+, torch, numpy, sklearn, matplotlib`
- To view the modeling and class defintions see `/Analog/fg_ml.py`
- To see the training loop see `/Analog/mnist_fpaa.py` and to train a network run `python /Analog/mnist_fpaa.py`
    - For a snapshot of the network trained for the paper see `fg_mnist_weights_v3.npz`
    - Run the training file to generate a new network
- To estimate power consumption run `python /Analog/analog_mnist_power.py`
    - open the file and edit the path to the name of the new file generated if you wish to estimate for a different training run.

## Digital
- The digital accelerator was designed and functionally verified using [cocotb](https://www.cocotb.org/) and [verilator](https://www.veripool.org/verilator/) in an ubuntu docker image. The metrics were estimated by synthesizing the design to a 130nm FPGA architecture using [VPR](https://docs.verilogtorouting.org/en/latest/). Install all 3 tools as well as `python 3.10+, numpy, torch, sklearn`.
- To train an 8-bit, quantization aware, digital network on the `sklearn` MNIST database subset in pytorch run `python /Digital/mnist_quant.py`
    - this should generate an `.npz` file with weights and test set.
- To run the testbench for the verilog design in cocotb, run the makefile `/Digital/Makefile` that specifies `mnist_dut.v` as verilog source and `test_mnist_dut.py` as the test module.
- To generate the metrics for the design on a 130nm architecture with power modeling for the 130nm process, use the VTR flow to run:
    ```
        $VTR_ROOT/vtr_flow/scripts/run_vtr_flow.py mnist_dut.v k6_N10_I40_Fi6_L4_frac0_ff1_130nm.xml --route_chan_width 100 -power -cmos_tech 130nm.xml
    ```
    - Look for max frequency under `temp/vpr.out` power metrics under `temp/mnist_dut.power`