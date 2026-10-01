# Softmax regression for a multi-label classification

## 1. Objective

- Multinomial logistic regression problem (objective function, gradient, classification criteria)
- PyTorch library (Tensor, Dataset, DataLoader)
- Neural Network (matrix multiplication, activiation function)
- Stochastic Gradient Descent (mini-batch size, learning rate, number of epochs)

## 2. Baseline notebook code

- [assignment_02.ipynb](assignment_02.ipynb)

## 3. Utility codes

#### You have to complete the following codes:

- [MyModel.py](MyModel.py)
- [MyOptim.py](MyOptim.py)

#### You do not have to modify the following codes:

- [MyDataset.py](MyDataset.py)
- [MyEval.py](MyEval.py)
- [MyResult.py](MyResult.py)

## 4. Data

- mnist images of size $16 \times 16$ for digits 0, 1, 2, 3, 4
  - train ($300 \times 5$ classes)
  - test ($900 \times 5$ classes)

## 5. Linear layer

- neural network $f_w(x)$ can consist of a sequence of the composition of linear functions and activation functions
- a linear function defines a linear transformation $\mathbb{R}^p \mapsto \mathbb{R}^q$
- a linear transformation can be obtained by a matrix multiplication:

> $$
> y = A x,
> $$
>
> where $A \in \mathbb{R}^{q \times p}$, $x \in \mathbb{R}^p$ and $y \in \mathbb{R}^q$

- a neural network for input $x$ can be defined by:

> $$
> z = f_w(x) = A_k \, \sigma( A_{k-1} \cdots \sigma( A_2 \, \sigma( A_1 \, x ))),
> $$
>
> where $z = (z_0, z_1, \cdots, z_{K-1}) \in \mathbb{R}^K$ for $K$ number of lables, $w$ denotes weights in the constituent linear layers and $\sigma$ denotes activation function such as sigmoid function as defined by:
>
> $$
> \sigma(z) = \frac{1}{1 + \exp(-z)}
> $$

- prediction $h_k = \sigma(z)_k$ from the neural network $f_w(x)$ for the $k$-th label for input $x$ is defined by softmax as defined by:

> $$
> h_k = \sigma(z)_k = \frac{\exp(z_k)}{\sum_{j=0}^{K-1} \exp(z_j)}
> $$
>
> where $h = (h_0, h_1, \cdots, h_{K-1})$, $K$ is the number of labels and $k = 0, 1, \cdots, K-1$ denotes the index of label

- softmax function can be defined by the shifted output vector $\tilde{z} = z - \max(z)$ in order to avoid overflow of the exponential function as defined by:

> $$
> \begin{aligned}
> \tilde{z} &= z - \max{(z)}\\ 
> h_k &= \sigma(\tilde{z})_k = \frac{\exp(\tilde{z}_k)}{\sum_{j=0}^{K-1} \exp(\tilde{z}_j)}
> \end{aligned}
> $$
>
> where $h = (h_0, h_1, \cdots, h_{K-1})$, $K$ is the number of labels and $k = 0, 1, \cdots, K-1$ denotes the index of label

- classification criteria are defined by:

> $$
> l(x) = \arg\max_k (h_0, h_1, \cdots, h_k, \cdots, h_{K-1})
> $$
>
> where $l(x)$ denotes a label function that determines the class of $x$

## 6. Objective function

- objective function is defined by cross entropy:

> $$
> \mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^{n} \ell_i(w),
> $$
>
> where $\ell_i(w)$ denotes the cross entropy as defined by:
>
> $$
> \ell_i(w) = - \sum_{k=0}^{K-1} y_k \log{h_k}
> $$
>
> where label vector $y = (y_0, y_1, \cdots, y_{K-1})$ is defined in the form of one-hot encoding and prediction vector $h = (h_0, h_1, \cdots, h_{K-1})$ indicates the probability for each class labels

## 7. Gradient

- gradient of the loss $\mathcal{L}(w)$ with respect to the weight $w$ is defined by:

> $$
> \nabla \mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^{n} \nabla \ell_i(w),
> $$
>
> where the gradient of $\ell_i(w)$ for each pair of input $(x_i, y_i)$ is defined by:
>
> $$
> \begin{aligned}
> \nabla \ell_i(w) & = - \sum_{k=0}^{K-1} y_k \frac{1}{h_k} \frac{\partial h_k}{\partial w}\\
> \frac{\partial h_k}{\partial z_k} & = \frac{\partial}{\partial z_k} \left( \frac{\exp(z_k)}{\sum_{j=0}^{K-1} \exp(z_j)} \right)\\
> & = \frac{\exp(z_k) \sum_{j=0}^{K-1} \exp(z_j) - \exp(z_k) \exp(z_k)}{\left(\sum_{j=0}^{K-1} \exp(z_j)\right)^2}\\
> & = \left(\frac{\exp(z_k)}{\sum_{j=0}^{K-1} \exp(z_j)}\right) \left(\frac{\sum_{j=0}^{K-1} \exp(z_j) - \exp(z_k)}{\sum_{j=0}^{K-1} \exp(z_j)}\right)\\
> & = h_k (1 - h_k)\\
> \frac{\partial h_k}{\partial z_p} & = \frac{\partial}{\partial z_p} \left( \frac{\exp(z_k)}{\sum_{j=0}^{K-1} \exp(z_j)} \right), \quad (p \neq k)\\
> & = \frac{- \exp(z_k) \exp(z_p)}{\left(\sum_{j=0}^{K-1} \exp(z_j)\right)^2}\\
> & = \left(\frac{\exp(z_k)}{\sum_{j=0}^{K-1} \exp(z_j)}\right) \left(\frac{- \exp(z_p)}{\sum_{j=0}^{K-1} \exp(z_j)}\right)\\
> & = - h_k \, h_p
> \end{aligned}
> $$

## 8. Optimization by Stochastic Gradient Descent

- gradient descent step with a mini-batch is given as follows:

> $$
> \begin{aligned}
> w^{(t+1)} & = w^{(t)} - \eta \nabla \mathcal{L}(w)\\
> & = w^{(t)} - \eta \frac{1}{n} \sum_{i=1}^{n} \nabla \ell_i(w)
> \end{aligned}
> $$
>
> where $\eta$ denotes the learning rate and $n$ denotes the number of training data

## 9. Configuration

- batch size
- learning rate
- number of epochs
- initialization of model parameters
- number of layers
- activation functions
- dimension of the linear maps
- softmax
- gradient of softmax
- classification criteria

## 10. Development Environment

### Create Python Virtual Environment

```console
$ python -m venv env_XX
```

### Activate a Virtual Environment

```console
$ source env_XX/bin/activate
```

### Install necessary Packages

```console
$ pip install -r requirements.txt
```

### Clone the repository

```console
$ git clone path_to_repository
```

### Edit the codes

- use any development software (e.g. Visual Studio Code)

### Add all the result files (e.g. `model.pth`, `result.csv`)

```console
$ git add .
```

### Commit the codes

```console
$ git commit -am "commit message"
```

### Create and switch to your branch

```console
$ git checkout -b student/your-github-account
```

### Push the codes

```console
$ git push -u origin student/your-github-account
```

### Deactivate a Virtual Environment

```console
$ deactivate
```
