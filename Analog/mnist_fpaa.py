import torch
import time
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import fg_ml as fg
import os
import numpy as np

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.25)

class torch_mnist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_0 = torch.nn.Linear(64,100)
        self.mlp_1 = torch.nn.Linear(100,10)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        layer_0 = self.mlp_0(x)
        layer_0 = self.relu(layer_0)
        
        layer_1 = self.mlp_1(layer_0)
        #layer_1 = self.relu(layer_1)
        
        return layer_1

def train_torch_mnist():
    model = torch_mnist()
    xt_train = torch.as_tensor(X_train, dtype=torch.float)
    xt_test = torch.as_tensor(X_test, dtype=torch.float)
    yt_train = torch.as_tensor(y_train, dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    st = time.time()
    
    # Training loop
    for t in range(2000):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(xt_train)
        # Compute and print loss
        loss = criterion(y_pred, yt_train)

        if torch.isnan(loss) == True:
            print(f'Loss is nan at iter {t}')
            exit()
        if t % 500 == 0:
            print('Loss:')
            print(t, loss.item())
            train_elapsed = time.time()
            print(f'Time so far {train_elapsed - st}')
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    final_pred = model(xt_test).argmax(dim=1).numpy()
    print("Accuracy of model = %2f%%" % (accuracy_score(y_test, final_pred)*100))
    return model

def predict_single(model, idx=0, xt_test=X_test):
    try:
        while True:
            idx = int(input('Enter data index: '))
            x = torch.tensor(xt_test[idx], dtype=torch.float)
            with torch.no_grad():
                pred = model(x).argmax().item()

            plt.imshow(xt_test[idx].reshape(8, 8), cmap='gray_r')
            plt.axis('off')
            plt.title(f"Predicted: {pred}  |  True: {y_test[idx]}")
            plt.show()
    except KeyboardInterrupt:
        print('Done')
    return pred

def predict_single_fg(model, xt_test=X_test, yt_test=y_test):
    pred = model(xt_test)
    try:
        while True:
            idx = int(input('Enter data index: '))
            pred_single = pred[idx].argmax().numpy()
            plt.imshow(xt_test[idx].reshape(8, 8), cmap='gray_r')
            plt.axis('off')
            plt.title(f"Predicted: {pred_single}  |  True: {yt_test[idx]}")
            plt.show()
    except KeyboardInterrupt:
        print('Done')
    return pred

def train_fg_mnist_npz_store():
    # Rescale data
    arr = 1.8 + (digits.data) * 0.7 / 16.0
    X_train, X_test, y_train, y_test = train_test_split(arr, digits.target, test_size=0.25)

    # setup my model
    model = fg.fg_mnist()
    xt_train = torch.as_tensor(X_train, dtype=torch.float)
    xt_test = torch.as_tensor(X_test, dtype=torch.float)
    yt_train = torch.as_tensor(y_train, dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-9)
    st = time.time()

    # Training loop
    for t in range(100):
        y_pred = model(xt_train)
        loss = criterion(y_pred, yt_train)

        if torch.isnan(loss):
            print(f'Loss is nan at iter {t}')
            return

        if t % 100 == 0:
            print('Loss:')
            print(t, loss.item())
            train_elapsed = time.time()
            print(f'Time so far {train_elapsed - st}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    el = time.time() - st
    final_pred = model(xt_test).argmax(dim=1).cpu().numpy()
    print("Accuracy of model = %2f%%" % (accuracy_score(y_test, final_pred) * 100))
    mul_time = model.get_mul_time()
    print(f'Full training took {el} seconds and VMM alone took {mul_time} which is {(mul_time / el) * 100}%')

    # Collect weights to save
    weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            key = name.replace('.', '_')
            weights[key] = param.detach().cpu().numpy()

    # Save everything in one NPZ at the same location root
    base_dir = os.path.join('.')
    path = os.path.join(base_dir, 'fg_mnist_weights_new_v1.npz')

    np.savez(
        path,
        test_data=X_test,
        labels=y_test,
        param_names=np.array(list(weights.keys())),
        **weights
    )

    predict_single_fg(model, xt_test=xt_test, yt_test=y_test)


if __name__ == "__main__":
    train_fg_mnist_npz_store()