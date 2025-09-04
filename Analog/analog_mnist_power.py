import numpy as np
import os
import matplotlib.pyplot as plt
import fg_ml as fg
import torch

path = os.path.join('fg_mnist_weights_v3.npz')
npz = np.load(path)

layer1_wt = npz['mlp_0_weights']
layer1_bias = npz['mlp_0_bias']
layer2_wt = npz['mlp_1_weights']
layer2_bias = npz['mlp_1_bias']

x_test = npz['test_data']
y_test = npz['labels']

def vmm_max_power():
    l1_current_vec = np.sum(np.abs(layer1_wt), axis=1) + np.abs(layer1_bias)
    l1_curr_scalar = np.sum(l1_current_vec)

    l2_current_vec = np.sum(np.abs(layer2_wt), axis=1) + np.abs(layer2_bias)
    l2_curr_scalar = np.sum(l2_current_vec)
    print(f'VMM max:\nl1 current {l1_curr_scalar*1e6:.3f} uA, l2 current {l2_curr_scalar*1e6:.3f} uA')
    # Pick the TA bias and Fet current from an ideal activation
    # Other neurons will be tuned to produce similar outputs
    act_current = 110*(2*570e-9 + 400e-9) # 2 x TA bias + Fet current
    wta_current = 100e-9
    print(f'110 Neurons: {act_current*1e6:.3f} uA')

    total = (act_current + l1_curr_scalar + l2_curr_scalar + wta_current)*2.5
    print(f'Theoretical max power {total*1e3:.3f} mW')

def power_per_class():
    act_current = 110*(2*570e-9 + 400e-9)
    wta_current = 100e-9
    l1_vmm = fg.fg_vmm()
    l1_act = fg.fg_mlp(64, 100)
    power_class = {}
    avg_power = 0
    for test_val in [0, 1, 2, 3, 4, 5, 6, 7, 8 , 9]:
        mask = (y_test == test_val)
        test_idx = mask.argmax()
        inp_act = torch.tensor(x_test[test_idx])
        inp_act = inp_act.unsqueeze(0)
        l1_vmm.weights = torch.tensor(layer1_wt)
        l1_vmm.bias = torch.tensor(layer1_bias)
        
        pos_curr, neg_curr = l1_vmm.fg_mult(inp_act)
        l1_current = pos_curr.abs().sum() + neg_curr.abs().sum()
        power_class[test_val] = {'l1_curr': l1_current}
        #print(f'Layer 1 VMM current for class {test_val} = {l1_current*1e6:.3f} uA')

        l1_hidden = l1_act.fg_sigmoid(pos_curr, neg_curr)
        l1_vmm.weights = torch.tensor(layer2_wt)
        l1_vmm.bias = torch.tensor(layer2_bias)
        pos_curr, neg_curr = l1_vmm.fg_mult(l1_hidden)
        l2_current = pos_curr.abs().sum() + neg_curr.abs().sum()
        power_class[test_val] = {'l2_curr': l2_current}
        #print(f'Layer 2 VMM current for class {test_val} = {l2_current*1e6:.3f} uA')
        
        est_power = (act_current + l1_current + l2_current + wta_current)*2.5
        avg_power += est_power
        print(f'Class {test_val} Power {est_power*1e3:.3f} mW')
    avg_power = avg_power/10
    print(f'Average class power {avg_power*1e3:.3f} mW')


if __name__ == "__main__":
    # Function to optionally calculate power of the unrealistic case with all max voltage inputs
    #vmm_max_power()
    power_per_class()