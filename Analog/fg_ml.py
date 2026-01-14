import torch
import time
import numpy as np

class fg_vmm(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        pass
    
    def fg_mult_subvt(self, v_s, i_leak=0):
        # Compute scale for each weight
        weights = self.weights
        bias = self.bias
        abs_weights = weights.abs()
        scale = -7.1221 * torch.log10(abs_weights) - 22.422
        
        # Use broadcasting to multiply and exponentiate v_s with scales
        v_s = torch.where(v_s > 2.5, 2.5, v_s)
        v_s = v_s - 2.5
        expanded_v_s = v_s.unsqueeze(1)
        print(f'vs shape {expanded_v_s.shape}')
        expanded_scale = scale.unsqueeze(0)
        print(f'scale shape {expanded_scale.shape}')
        exp_component = torch.exp(expanded_v_s * expanded_scale)
        print(f'exp shape {exp_component.shape}')
        # Compute positive and negative output separately
        pos_mask = weights >= 0
        neg_mask = weights < 0

        # Calculate outputs for each weight and add the leak term
        pos_output_per_weight = weights * pos_mask * exp_component + i_leak * pos_mask
        neg_output_per_weight = weights * neg_mask * exp_component - i_leak * neg_mask
        print(f'weighted shape {pos_output_per_weight.shape}')
        # Add the bias term per row
        pos_output = pos_output_per_weight.sum(dim=2) + (bias * (bias >= 0))
        neg_output = neg_output_per_weight.sum(dim=2) + (bias * (bias < 0))
        
        return pos_output, neg_output
    
    def fg_mult(self, v_s, i_leak=7e-12):
        # Get the weights and biases
        weights = self.weights
        bias = self.bias
        abs_weights = weights.abs()

        spec_curr = (torch.log(abs_weights)* -8.0282e-9) -8.0497e-8
        a = 17.5328
        b = (torch.log(abs_weights)* -1.41837) + 18.4134
        # expand into batched dimension for subsequent operations
        v_s = torch.where(v_s > 2.5, 2.5, v_s)
        expanded_v_s = v_s.unsqueeze(1)
        #expanded_a = a.unsqueeze(0)
        expanded_b = b.unsqueeze(0)
        exp_component = torch.exp(((expanded_v_s * a)) - expanded_b)
        unweighted_exp = torch.square(torch.log(1 + exp_component))
        
        # Compute positive and negative output separately
        pos_mask = weights >= 0
        neg_mask = weights < 0

        # Calculate outputs for each weight and add the leak term
        pos_output_per_weight = spec_curr * pos_mask * unweighted_exp + i_leak * pos_mask
        # spec curr has to be negative because it pulled from an absolute value
        neg_output_per_weight = -spec_curr * neg_mask * unweighted_exp - i_leak * neg_mask

        # Add the bias term per row
        pos_output = pos_output_per_weight.sum(dim=2) + (bias * (bias >= 0))
        neg_output = neg_output_per_weight.sum(dim=2) + (bias * (bias < 0))
        
        return pos_output, neg_output

class fg_mlp(fg_vmm):
    def __init__(self, num_in, num_out, activation='sigmoid'):
        super().__init__()
        self.act = activation
        self.num_in = num_in
        self.num_out = num_out

        #Construct 2D array of weights and vector of biases

        self.weights = torch.ones(num_out, num_in)*10e-9
        rand_init = torch.randn(num_out, num_in)*9
        self.weights = torch.nn.Parameter(self.weights*rand_init)
        self.bias = torch.nn.Parameter(torch.Tensor([10e-9]*num_out)) 
        self.mul_time = 0
        
    def set_weights(self, wts, bias):
        self.weights = torch.nn.Parameter(wts)
        self.bias = torch.nn.Parameter(bias)
    
    def fg_relu_v1(self, current):
        # Separate out the positive and negative currents 
        curr_pos = torch.where(current >= 0, current, 0)
        curr_neg = torch.where(current < 0, abs(current), 0)
        pos_relu = 2.07 + 0.06*torch.exp(-0.8+400000*curr_pos) + 0.26/(1.15+ torch.exp(2.7-1800000*curr_pos))
        # Choosing 1u weight parameters for now
        prog_bias, fall_off = 2.1132, 43.5
        neg_relu = prog_bias/(1 + torch.exp(fall_off + 7*torch.log10(curr_neg)))
    
        # Create a mask to ensure the sum has no overlap in indices
        pos_mask = torch.where(current >= 0, 1, 0)
        neg_mask = torch.where(current < 0, 1, 0)
        ret_val = pos_relu*pos_mask + neg_relu*neg_mask

        return ret_val
    
    def fg_relu(self, current):
        # Separate out the positive and negative currents 
        curr_pos = torch.where(current >= 0, current, 0)
        curr_neg = torch.where(current < 0, abs(current), 0)
        
        # Apply 3 linear piecewise functions for separate positive regions
        pos1 = torch.where((curr_pos < 10**-7.3) & (curr_pos > 0), 0.022*torch.log(curr_pos) + 2.615, 0 )
        pos1 = torch.where((pos1 < 2) & (pos1 > 0), 2, pos1)
        pos2 = torch.where((curr_pos >= 10**-7.3) & ( curr_pos < 10**-6.2), 0.04*torch.log(curr_pos) + 2.92, 0 )
        pos3 = torch.where(curr_pos >= 10**-6.2, 0.08*torch.log(curr_pos) + 3.49, 0 )
        pos3 = torch.where(pos3 > 2.5, 2.5, pos3)
        pos_relu = pos1 + pos2 + pos3

        # Apply the exponential negative fall off, choosing 1u weight parameters for now
        prog_bias, fall_off = 2.1132, 43.5
        neg_relu = prog_bias/(1 + torch.exp(fall_off + 7*torch.log10(curr_neg)))

        # Create a mask to ensure the sum has no overlap in indices
        pos_mask = torch.where(current >= 0, 1, 0)
        neg_mask = torch.where(current < 0, 1, 0)
        ret_val = pos_relu*pos_mask + neg_relu*neg_mask

        return ret_val

    def nfet_itov(self, inp_curr):

        if torch.max(inp_curr) > 0:
            last_tran = 2.152e-6
            last_tran_val = 2.0001
        
        else:
            inp_curr = torch.abs(inp_curr)
            last_tran = 4.67e-6
            last_tran_val = 2.4507
                
        a=-1.7814e29
        b=4.6537e19
        c=5.3942e8 #scaling for x_inpcurr
        d=-0.0182625

        f=-13.2209
        g=3.03607
        h=1.75561
        i=-1.9597*(10**-7) #scaling for x_inpcurr
        j=1500
        k=0.120559
        
        # Apply 2 piecewise functions for separate regions
        
        volt1 = torch.where((inp_curr <= 1.5166e-11), 0, 0 )
        volt2 = torch.where( (inp_curr > 1.5166e-11) & (inp_curr <= 1.622e-10), a*(inp_curr**3)+b*(inp_curr**2)+c*(inp_curr)+d, 0 )
        volt3 = torch.where((inp_curr > 1.622e-10) & (inp_curr <= last_tran),(f/(g+h*torch.log(abs(i+j*inp_curr)))) + k , 0 )
        volt4 = torch.where((inp_curr > last_tran), last_tran_val , 0 )

        volt = volt1+volt2+volt3+volt4
        return volt
        

    def sigmoid_diff(self, pos_inp, neg_inp):
        # Apply 3 piecewise functions for separate regions
        volt1 = torch.where((pos_inp <= 0.38) , (-0.2877195 / ( 1+torch.exp(31.202 * (pos_inp-neg_inp) + (-29.0649*pos_inp+12.7174) )) ) + 2.44313, 0 )
        volt2 = torch.where((pos_inp > 0.38) & (pos_inp <= 1.78) , (0.131407*pos_inp-0.338609) / (1+torch.exp( (-3.98366*pos_inp+32.2026) * (pos_inp-neg_inp) + (-0.926118*pos_inp+2.4177))) + (-0.00157441*pos_inp+2.44385), 0 )
        volt3 = torch.where((pos_inp > 1.78), (0.407395*pos_inp-0.830324)/(1+torch.exp((-32.32*pos_inp+83.1541) * (pos_inp-neg_inp) + -9.69387*pos_inp+17.9666)) + 2.440658, 0 )
        volt = volt1+volt2+volt3
        
        return volt
        
        
    def fg_sigmoid(self, curr_pos, curr_neg):
        p_volt = self.nfet_itov(curr_pos)
        n_volt = self.nfet_itov(curr_neg)

        return self.sigmoid_diff(p_volt, n_volt)
    
    def forward(self, v_s):
        if self.act == 'relu':
            mul_start = time.time()
            p_currs, n_currs = self.fg_mult(v_s)
            mul_end = time.time()
            self.mul_time += mul_end - mul_start
            currs = p_currs + n_currs
            ret_val = self.fg_relu(currs)
            return ret_val
        
        elif self.act == 'sigmoid':
            mul_start = time.time()
            p_currs, n_currs = self.fg_mult(v_s)
            mul_end = time.time()
            self.mul_time += mul_end - mul_start
            ret_val = self.fg_sigmoid(p_currs, n_currs)
            return ret_val

    def show_weights(self):
        for idx in range(self.num_out):
            print(f'Row {idx}, bias: {self.bias[idx]} weights: {self.weights[idx]}')

class fg_wta(fg_vmm):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        # Construct 2D array of weights and vector of biases
        self.weights = torch.ones(num_out, num_in)*1e-9
        rand_init = abs(torch.randn(num_out, num_in)*50)
        self.weights = torch.nn.Parameter(self.weights*rand_init)
        self.bias = torch.nn.Parameter(torch.Tensor([5e-9]*num_out)) #before 5e-9
        self.mul_time = 0
        
    def set_weights(self, wts, bias):
        self.weights = torch.nn.Parameter(wts)
        self.bias = torch.nn.Parameter(bias)
    
    def show_weights(self):
        for idx in range(self.num_out):
            print(f'Row {idx}, bias: {self.bias[idx]} weights: {self.weights[idx]}')
    
    def fg_mult(self, v_s, i_leak=0):
        # This layer should only allow positive weights as input to the WTA
        # Get the weights and biases
        weights = self.weights
        bias = self.bias
        abs_weights = weights.abs()
        # compute variables for each weight
        spec_curr = abs_weights*-0.0065 + 5e-8
        a = abs_weights*632404 + 18.5384
        b = abs_weights*-1.2343e6 + 44.2817
        
        # expand into batched dimension for subsequent operations
        v_s = torch.where(v_s > 2.5, 2.5, v_s)
        expanded_v_s = v_s.unsqueeze(1)
        expanded_a = a.unsqueeze(0)
        expanded_b = b.unsqueeze(0)
        exp_component = torch.exp((expanded_v_s * expanded_a) - expanded_b)
        unweighted_exp = torch.square(torch.log(1 + exp_component))
        
        # Compute positive 
        pos_mask = weights >= 0

        # Calculate outputs for each weight and add the leak term
        pos_output_per_weight = spec_curr * pos_mask * unweighted_exp + i_leak * pos_mask

        # Add the bias term per row
        pos_output = pos_output_per_weight.sum(dim=2) + (bias * (bias >= 0))

        return pos_output

    def nfet_itov(self, inp_curr):
        mask1 = torch.where(inp_curr<1.74e-9, 0.003, 0)
        mask2 = torch.where((inp_curr>=1.74e-9) & (inp_curr<4.9e-7), 1.3703e12*(inp_curr**2) + 4.9357e5*inp_curr -6.2089e-4, 0)
        mask3 = torch.where((inp_curr>=4.9e-7) & (inp_curr<1.77e-6), -1.0314e11*(inp_curr**2) + 4.2304e5*inp_curr +0.3746, 0)
        mask4 = torch.where(inp_curr >= 1.77e-6, 0.9, 0)
        volt = mask1 + mask2 + mask3 + mask4
        return volt

    def wta(self, currs):
        # this function was designed to produce maximum output with a row current of low uA values
        # it puts the current through log to get easier numbers to work with then scales the output to maximize distance between values
        # the logisitic equations ensure that values in the nA or below return 0 while values in the uA return 1. This matches FPAA programming
        val = -23-4*torch.log10(torch.abs(currs))
        ret_val = 10/(1+ torch.exp(val))
        return ret_val

    def forward(self, v_s):
        mul_start = time.time()
        currs = self.fg_mult(v_s)
        mul_end = time.time()
        self.mul_time += mul_end - mul_start
        out = self.wta(currs)
        return out

class fg_nn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = fg_mlp(2, 1)
        self.wta = fg_wta(1, 2)
    
    def forward(self, x):
        layer_0 = self.mlp(x)
        layer_1 = self.wta(layer_0)
        return layer_0, layer_1
    
    def show_weights(self):
        print('MLP weights')
        self.mlp.show_weights()
        print('Softmax weights')
        self.wta.show_weights()
    
    def get_mul_time(self):
        return self.mlp.mul_time + self.wta.mul_time

class fg_nn_conc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_0 = fg_mlp(2, 5, activation="sigmoid")
        self.wta = fg_wta(5, 2)
    
    def forward(self, x):
        layer_0 = self.mlp_0(x)
        layer_2 = self.wta(layer_0)
        return layer_2
    
    def show_weights(self):
        print('MLP 0 weights')
        self.mlp_0.show_weights()
        print('WTA weights')
        self.wta.show_weights()
    
    def show_gradients(self):
        print('WTA Gradients')
        print(f'Weights {self.wta.weights.grad}')
        print(f'Bias {self.wta.bias.grad}')

    def update_weights(self, mlp0, mlp1, mlp2, wta):
        self.mlp_0.set_weights(mlp0['weights'], mlp0['bias'])
        self.wta.set_weights(wta['weights'], wta['bias'])

    def get_mul_time(self):
        return self.mlp_0.mul_time + self.wta.mul_time

class fg_conc_v2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_0 = fg_mlp(2, 4, activation="sigmoid")
        self.mlp_1 = fg_mlp(4, 2, activation="sigmoid")
    
    def forward(self, x):
        layer_0 = self.mlp_0(x)
        layer_1 = self.mlp_1(layer_0)
        layer_1 = 4.0 * (layer_1 - 2.25) / 0.25
        return layer_1
    
    def show_weights(self):
        print('MLP 0 weights')
        self.mlp_0.show_weights()
        print('MLP 1 weights')
        self.mlp_1.show_weights()


    def update_weights(self, mlp0, mlp1, mlp2, wta):
        self.mlp_0.set_weights(mlp0['weights'], mlp0['bias'])
        self.mlp_1.set_weights(mlp1['weights'], mlp1['bias'])

    def get_mul_time(self):
        return self.mlp_0.mul_time + self.mlp_1.mul_time


class fg_mnist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_0 = fg_mlp(64,100, activation="sigmoid")
        self.mlp_1 = fg_mlp(100,10, activation="sigmoid")
    
    def forward(self, x):
        layer_0 = self.mlp_0(x)
        layer_1 = self.mlp_1(layer_0)
        layer_1 = 4 * (layer_1 - 2.25) / 0.25
        return layer_1
    
    def show_weights(self):
        print('MLP 0 weights')
        self.mlp_0.show_weights()
        print('MLP 1 weights')
        self.mlp_1.show_weights()
    
    def get_mul_time(self):
        return self.mlp_0.mul_time + self.mlp_1.mul_time
    

class fg_mnist28(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_0 = fg_mlp(784,64, activation="sigmoid")
        self.mlp_1 = fg_mlp(64,10, activation="sigmoid")
    
    def forward(self, x):
        layer_0 = self.mlp_0(x)
        layer_1 = self.mlp_1(layer_0)
        layer_1 = 4 * (layer_1 - 2.25) / 0.25
        return layer_1
    
    def show_weights(self):
        print('MLP 0 weights')
        self.mlp_0.show_weights()
        print('MLP 1 weights')
        self.mlp_1.show_weights()
    
    def get_mul_time(self):
        return self.mlp_0.mul_time + self.mlp_1.mul_time