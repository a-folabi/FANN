import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------
# Data
# ----------------------------
def load_data():
    digits = load_digits()             # values 0..16
    X = digits.data.astype(np.float32) # [N, 64]
    y = digits.target.astype(np.int64)
    X = X / 16.0
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ----------------------------
# Quant helpers
# ----------------------------
def safe_scale_from_max(max_val, qmax):
    max_val = float(max_val)
    return (max_val / qmax) if max_val > 0 else (1.0 / qmax)

def fake_quant_w_int8_sym(x):
    with torch.no_grad():
        max_abs = x.abs().max()
        s = safe_scale_from_max(max_abs, 127.0)
        q = torch.clamp(torch.round(x / s), -128, 127)
    x_hat = (q * s).detach() - x.detach() + x
    return x_hat, s

def fake_quant_act_uint8(x):
    with torch.no_grad():
        max_val = x.max()
        s = safe_scale_from_max(max_val, 255.0)
        q = torch.clamp(torch.round(x / s), 0, 255)
    x_hat = (q * s).detach() - x.detach() + x
    return x_hat, s

def fake_quant_in_uint8(x):
    x = torch.clamp(x, 0)
    return fake_quant_act_uint8(x)

def fake_quant_bias_int32(b, s_in, s_w):
    s_b = s_in * s_w
    with torch.no_grad():
        q = torch.round(b / s_b)
        q = torch.clamp(q, -(2**31), 2**31 - 1)
    b_hat = (q * s_b).detach() - b.detach() + b
    return b_hat, s_b

# ----------------------------
# Modules
# ----------------------------
class QLinearReLU8(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.register_buffer("in_amax", torch.tensor(0.0))
        self.register_buffer("out_amax", torch.tensor(0.0))

    def forward(self, x):
        x_clamped = torch.clamp(x, 0)
        with torch.no_grad():
            self.in_amax = torch.maximum(self.in_amax, x_clamped.max())
        x_q, s_in = fake_quant_in_uint8(x_clamped)

        w = self.linear.weight
        b = self.linear.bias
        w_q, s_w = fake_quant_w_int8_sym(w)
        b_q, s_b = fake_quant_bias_int32(b, torch.tensor(s_in, device=b.device), torch.tensor(s_w, device=b.device))

        y = F.linear(x_q, w_q, b_q)
        y = F.relu(y)

        with torch.no_grad():
            self.out_amax = torch.maximum(self.out_amax, y.max())
        y_q, s_out = fake_quant_act_uint8(y)

        self.last_s_in = float(s_in)
        self.last_s_w = float(s_w)
        self.last_s_b = float(s_in * s_w)
        self.last_s_out = float(s_out)
        return y_q

class QLinear8(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.register_buffer("in_amax", torch.tensor(0.0))

    def forward(self, x):
        x_clamped = torch.clamp(x, 0)
        with torch.no_grad():
            self.in_amax = torch.maximum(self.in_amax, x_clamped.max())
        x_q, s_in = fake_quant_in_uint8(x_clamped)

        w = self.linear.weight
        b = self.linear.bias
        w_q, s_w = fake_quant_w_int8_sym(w)
        b_q, _ = fake_quant_bias_int32(b, torch.tensor(s_in, device=b.device), torch.tensor(s_w, device=b.device))

        y = F.linear(x_q, w_q, b_q)
        self.last_s_in = float(s_in)
        self.last_s_w = float(s_w)
        self.last_s_b = float(s_in * s_w)
        return y

class TorchMNIST_QAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_0 = QLinearReLU8(64, 100, bias=True)
        self.mlp_1 = QLinear8(100, 10, bias=True)

    def forward(self, x):
        x = self.mlp_0(x)
        x = self.mlp_1(x)
        return x

# ----------------------------
# Train and evaluate
# ----------------------------
def train_model(X_train, X_test, y_train, y_test, epochs=3000, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = TorchMNIST_QAT().train()
    xt_train = torch.as_tensor(X_train, dtype=torch.float32)
    xt_test = torch.as_tensor(X_test, dtype=torch.float32)
    yt_train = torch.as_tensor(y_train, dtype=torch.long)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    st = time.time()
    for t in range(epochs):
        y_pred = model(xt_train)
        loss = criterion(y_pred, yt_train)
        if torch.isnan(loss):
            print(f'Loss is nan at iter {t}')
            break
        if t % 500 == 0:
            print('Loss:')
            print(t, loss.item())
            print(f'Time so far {time.time() - st:.2f}s')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_logits = model(xt_test)
        final_pred = final_logits.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_test, final_pred)
    print("Accuracy with fake-quant pipeline = %2f%%" % (acc * 100.0))
    return model, acc, final_pred

# ----------------------------
# Calibrate and export real ints + test set
# ----------------------------
@torch.no_grad()
def calibrate_activation_peaks(model, X_calib):
    model.eval()
    x = torch.as_tensor(X_calib, dtype=torch.float32)
    _ = model.mlp_0(torch.clamp(x, 0))
    _ = model.mlp_1(torch.clamp(model.mlp_0(torch.clamp(x, 0)), 0))

def _quantize_u8_batch(X_float, s):
    Xc = np.clip(X_float, 0.0, None)
    q = np.clip(np.round(Xc / s), 0, 255).astype(np.uint8)
    return q

def export_int8_int32_npz(model, out_prefix, X_test=None, y_test=None, torch_pred=None):
    model.eval()

    # Layer 0 scales
    l0 = model.mlp_0.linear
    l0_w = l0.weight.detach().cpu()
    l0_b = l0.bias.detach().cpu()
    l0_w_scale = safe_scale_from_max(float(l0_w.abs().max()), 127.0)
    l0_in_scale = safe_scale_from_max(float(model.mlp_0.in_amax.cpu().item()), 255.0)
    l0_out_scale = safe_scale_from_max(float(model.mlp_0.out_amax.cpu().item()), 255.0)
    l0_bias_scale = l0_in_scale * l0_w_scale

    l0_w_q = torch.clamp(torch.round(l0_w / l0_w_scale), -128, 127).to(torch.int8).numpy()
    l0_b_q = torch.round(l0_b / l0_bias_scale).clamp(-(2**31), 2**31 - 1).to(torch.int32).numpy()

    # Layer 1 scales
    l1 = model.mlp_1.linear
    l1_w = l1.weight.detach().cpu()
    l1_b = l1.bias.detach().cpu()
    l1_w_scale = safe_scale_from_max(float(l1_w.abs().max()), 127.0)
    l1_in_scale = safe_scale_from_max(float(model.mlp_1.in_amax.cpu().item()), 255.0)
    l1_bias_scale = l1_in_scale * l1_w_scale

    l1_w_q = torch.clamp(torch.round(l1_w / l1_w_scale), -128, 127).to(torch.int8).numpy()
    l1_b_q = torch.round(l1_b / l1_bias_scale).clamp(-(2**31), 2**31 - 1).to(torch.int32).numpy()

    payload = {
        "mlp_0.weight_q": l0_w_q,
        "mlp_0.weight_scale": np.array([l0_w_scale], dtype=np.float32),
        "mlp_0.bias_q": l0_b_q,
        "mlp_0.bias_scale": np.array([l0_bias_scale], dtype=np.float32),
        "mlp_0.in_scale": np.array([l0_in_scale], dtype=np.float32),
        "mlp_0.out_scale": np.array([l0_out_scale], dtype=np.float32),

        "mlp_1.weight_q": l1_w_q,
        "mlp_1.weight_scale": np.array([l1_w_scale], dtype=np.float32),
        "mlp_1.bias_q": l1_b_q,
        "mlp_1.bias_scale": np.array([l1_bias_scale], dtype=np.float32),
        "mlp_1.in_scale": np.array([l1_in_scale], dtype=np.float32),
    }

    if X_test is not None and y_test is not None:
        # Quantize test set with layer 0 input scale so it is plug and play in the testbench
        test_u8 = _quantize_u8_batch(X_test.astype(np.float32), l0_in_scale)
        payload.update({
            "test_inputs_u8": test_u8.astype(np.uint8),
            "test_labels": y_test.astype(np.int64)
        })
        if torch_pred is not None:
            payload["pytorch_test_pred"] = np.asarray(torch_pred, dtype=np.int64)

    np.savez(out_prefix + ".npz", **payload)
    print(f"Saved {out_prefix}.npz")
    print("Export summary:")
    print(f"  mlp_0 - in_scale={l0_in_scale:.6f}, w_scale={l0_w_scale:.6f}, bias_scale={l0_bias_scale:.6f}, out_scale={l0_out_scale:.6f}")
    print(f"  mlp_1 - in_scale={l1_in_scale:.6f}, w_scale={l1_w_scale:.6f}, bias_scale={l1_bias_scale:.6f}")
    if "test_inputs_u8" in payload:
        print(f"  Packed test set: {payload['test_inputs_u8'].shape[0]} samples")

# ----------------------------
# Reference integer math for deployment
# ----------------------------
def reference_int_pipeline_note():
    print("""
Deployment math per layer:
  Given A_uint8, W_int8, B_int32
  y_int32 = A_uint8 @ W_int8^T + B_int32
  y_float = (s_in * s_w) * y_int32
  For hidden layers: y_relu = max(0, y_float); A_next_uint8 = round(y_relu / s_out) clamped to [0,255]
  Next layer uses A_next_uint8 as input with its own s_in = s_out from previous layer
    """)

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    print("\nTrain with int8 W, int32 bias, uint8 activations simulated")
    model, acc, torch_test_pred = train_model(X_train, X_test, y_train, y_test, epochs=3000, lr=1e-3, seed=0)

    print("\nCalibrate activation peaks for export")
    calibrate_activation_peaks(model, X_train)

    print("\nExport real int8 weights, int32 bias, scales, and test set")
    export_int8_int32_npz(
        model,
        out_prefix="mnist8x8_int8W_int32B_uint8A",
        X_test=X_test,
        y_test=y_test,
        torch_pred=torch_test_pred
    )

    reference_int_pipeline_note()
