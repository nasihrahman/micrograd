# micrograd — A Tiny Autograd Engine 🧮

A minimalistic, educational implementation of an automatic differentiation engine, capable of backpropagation through a computational graph. Originally created by **Andrej Karpathy** as a learning resource for understanding the fundamentals behind deep learning frameworks.

---

## 🔎 Overview

`micrograd` is a tiny scalar-valued autograd engine with a small neural network library on top. It is designed for educational purposes to:

* Build computation graphs.
* Perform forward and backward passes.
* Understand gradient-based optimization at the most basic level.

This repo contains:

* Core autograd engine (`engine.py`)
* A small neural network library (`nn.py`)
* Example scripts and notebooks

---

## 📊 Features

* Reverse-mode automatic differentiation
* Scalar-based `Value` class with operator overloading
* Supports common mathematical operations (+, -, \*, /, \*\*)
* Minimal MLP implementation using `nn.py`
* Under \~100 lines for the core engine

---

## ⚙️ Requirements

* Python 3.7+
* NumPy (optional)

Install dependencies:

```bash
pip install numpy
```

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/micrograd.git
cd micrograd
```

2. Try the demo:

```bash
python demo.py
```

3. Explore Jupyter notebooks in `notebooks/` for hands-on learning.

---



---

## 🧪 Example

```python
from engine import Value

a = Value(2.0)
b = Value(-3.0)
c = a * b
c.backward()

print(a.grad, b.grad)  # Gradients for a and b
```

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

Based on **Andrej Karpathy’s micrograd**, a compact and powerful educational resource for understanding automatic differentiation and backpropagation.
