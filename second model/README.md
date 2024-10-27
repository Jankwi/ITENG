## The second model (moving on to PyTorch)

After my exploits in C++ didn't bring much success, I decided to move on to PyTorch.
i started learning Python, and I created this model after a few days.
Though the testing loss was simmilar to the C++ one, the scode got smaller, which was definietely the way forward.

## The model

The model is a very simple 4-layer feed forward neural network, with PyTorch as an activation function (and a sigmoid as the final one). The weights are saved as torch tensors in the parameters dictionary - just like in Andrew Ng's Coursera Neural Networks course.

```python
def iteration(X, Y, params, L, step_size, print_cost):
    params = grads_on(params)
    A = X
    m = X.shape[0]
    for l in range(1, L):
        Z = A@params['W' + str(l)] + params['b' + str(l)]
        A = torch.tanh(Z)
    Z = A@params['W' + str(L)] + params['b' + str(L)]
    A = sigmoid(Z)
    A = torch.clamp(A, min=1e-8)
    A = torch.clamp(A, max=1 - 1e-8)
    loss_table = -(Y*torch.log(A) + (1-Y)*torch.log(1-A))
    cost = loss_table.sum()/m
    if print_cost:
        print("THE CURRENT COST IS ", cost.item())
   # with torch.autograd.detect_anomaly():
    cost.backward()
    return update_params(params, step_size)
```

## The data
The data is stored in a csv file, which represents the one hot encoding of every word. It is imported using pandas.
```python
train_X = import_data("train_X.csv")
train_Y = import_data("train_Y.csv")
test_X = import_data("test_X.csv")
test_Y = import_data("test_Y.csv")
```

