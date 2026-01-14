import torch
import time
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import fg_ml as fg
import os
import numpy as np

import torchvision
from torch.utils.data import TensorDataset, DataLoader


#######################  MNIST 28x28 dataset #########################

def mnist28():
    # Download the raw datasets (Train=60k, Test=10k)
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

    # The raw data is originally integers 0-255 with shape (N, 28, 28)
    X_train = trainset.data
    y_train = trainset.targets
    X_test = testset.data
    y_test = testset.targets

    # Scaling the inputs to match hardware input range
    X_train = 1.8 + (X_train.float()) * 0.6 / 255.0
    X_test = 1.8 + (X_test.float()) * 0.6 / 255.0

    # Change shape from (N, 28, 28) -> (N, 784)
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = mnist28()


###################################################################


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

class torch_mnist28(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_0 = torch.nn.Linear(784,128)
        self.mlp_1 = torch.nn.Linear(128,10)
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

def predict_single_fg_28(model, xt_test, yt_test):
    # SETUP DEVICE
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device}")
    
    model.to(device)
    model.eval()

    #  PREDICT IN BATCHES
    batch_size = 1000 
    all_preds = []
    
    # Ensure xt_test is a tensor
    if isinstance(xt_test, np.ndarray):
        xt_tensor = torch.from_numpy(xt_test).float()
    else:
        xt_tensor = xt_test.float()
    
    n_samples = len(xt_tensor)
    print(f"Predicting {n_samples} images in batches...")

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_data = xt_tensor[i : i + batch_size]
            
            batch_data = batch_data.to(device)
            
            # Predict
            logits = model(batch_data)
            
            # Move result back to CPU and store
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
    
    # Combine all into one big array
    final_predictions = torch.cat(all_preds).numpy()
    print("Prediction complete.")

    # INTERACTIVE LOOP
    try:
        while True:
            val = input('Enter data index (or q to quit): ')
            if val.lower() == 'q': 
                break
            
            idx = int(val)
            
            # Lookup result from our big array
            pred_single = final_predictions[idx]
            true_label = yt_test[idx]

            # Get image for plotting
            img_to_show = xt_test[idx]
            if not isinstance(img_to_show, np.ndarray):
                img_to_show = img_to_show.cpu().numpy()

            plt.imshow(img_to_show.reshape(28, 28), cmap='gray_r')
            plt.axis('off')
            plt.title(f"Predicted: {pred_single}  |  True: {true_label}")
            plt.show()

    except KeyboardInterrupt:
        print('\nDone')
    except Exception as e:
        print(f"Error: {e}")
        
    return final_predictions


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



def train_fg_mnist28_npz_store():
    # SETUP DEVICE (Critical for speed)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS) acceleration.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    

    # PREPARE DATA
    xt_train, xt_test, yt_train, y_test = mnist28()
    
    # Convert to Tensors
    xt_train_t = torch.as_tensor(xt_train, dtype=torch.float32)
    yt_train_t = torch.as_tensor(yt_train, dtype=torch.long)
    xt_test_t = torch.as_tensor(xt_test, dtype=torch.float32)

    # Create a DataLoader 
    batch_size = 128
    train_data = TensorDataset(xt_train_t, yt_train_t)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # SETUP MODEL
    model = fg.fg_mnist28()
    #model = torch_mnist28()
    model.to(device) 
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-10) 

    st = time.time()
    
    # TRAINING LOOP (Epochs + Batches)
    epochs = 50 # Run through the dataset epochs times
    
    model.train() 
    for epoch in range(epochs):
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            # Move batch to device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if torch.isnan(loss):
                print(f'Loss is nan at epoch {epoch}, batch {i}')
                return

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Print average loss per epoch
        print(f'Epoch {epoch+1}, Avg Loss: {running_loss / len(train_loader)}')

    el = time.time() - st
    print(f'Training complete in {el:.2f} seconds')

    # EVALUATION
    model.eval() 
    
    # Process test set in batches
    test_loader = DataLoader(TensorDataset(xt_test_t), batch_size=1000)
    all_preds = []
    
    with torch.no_grad():
        for (inputs,) in test_loader:
            inputs = inputs.to(device)
            preds = model(inputs).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            
    final_pred = np.concatenate(all_preds)
    
    print("Accuracy of model = %2f%%" % (accuracy_score(y_test, final_pred) * 100))

    try:
        mul_time = model.get_mul_time() 
        print(f'VMM time reported: {mul_time}')
    except:
        pass

    # SAVE WEIGHTS
    weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            key = name.replace('.', '_')
            weights[key] = param.detach().cpu().numpy()

    base_dir = os.path.join('.')
    path = os.path.join(base_dir, 'fg_mnist_weights_new_v1.npz')

    np.savez(
        path,
        test_data=xt_test, 
        labels=y_test,
        param_names=np.array(list(weights.keys())),
        **weights
    )

    predict_single_fg_28(model, xt_test=xt_test, yt_test=y_test)


if __name__ == "__main__":
    train_fg_mnist28_npz_store()
