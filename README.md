# 🤖 Machine Learning Algorithms From Scratch

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Algorithms](https://img.shields.io/badge/Algorithms-From%20Scratch-green)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)
![GitHub Stars](https://img.shields.io/github/stars/ranjiGT/ml_algorithms?style=social)

> **Learn the mathematics. Understand the algorithms. Build machine learning from scratch.**

Welcome to **Machine Learning Algorithms From Scratch**, an educational repository dedicated to understanding and implementing fundamental machine learning algorithms using Python.

The goal is to explore how machine learning algorithms work internally, rather than simply calling functions from existing ML libraries.

This repository contains Python implementations, sample datasets, and algorithm-specific exercises designed to support hands-on learning.

---

## 🎯 Project Objectives

- Understand the mathematical foundations of machine learning.
- Implement core algorithms using Python.
- Explore supervised learning and optimization techniques.
- Practice working with real and synthetic datasets.
- Learn how model parameters are updated during training.
- Build a foundation for advanced machine learning and deep learning.

## 📂 Repository Structure

```text
ml_algorithms/
│
├── decision_tree/
│   ├── decision_tree.py
│   ├── car.csv
│   └── output.xml
│
├── gradient_descent/
│   ├── Assignment1.py
│   ├── output.csv
│   ├── random.csv
│   ├── test.csv
│   └── yacht.csv
│
├── perceptron/
│   ├── Assignment3.py
│   ├── Example.tsv
│   ├── Gauss2.tsv
│   └── test.csv
│
└── README.md
```

## 🧠 Algorithms Covered

### 1. Decision Tree

📁 [Explore Decision Tree](./decision_tree)

A decision tree is a supervised learning algorithm that recursively partitions data based on feature values.

**Key concepts:**

- Entropy and information gain
- Feature selection
- Recursive partitioning
- Classification
- Tree structure and decision rules

**Entropy:**

$$
H(S)=-\sum_{i=1}^{n}p_i\log_2(p_i)
$$

**Information Gain:**

$$
IG(S,A)=H(S)-\sum_{v\in Values(A)}
\frac{|S_v|}{|S|}H(S_v)
$$

Decision trees are commonly used for classification, interpretable predictions, and rule-based decision making.

### 2. Gradient Descent

📁 [Explore Gradient Descent](./gradient_descent)

Gradient descent is an iterative optimization algorithm used to minimize a differentiable objective function.

It is a fundamental building block of many machine learning models, including linear regression and neural networks.

**Parameter update rule:**

$$
\theta_{t+1}=\theta_t-\alpha\nabla J(\theta_t)
$$

Where:

- $\theta$ represents model parameters.
- $\alpha$ represents the learning rate.
- $J(\theta)$ represents the cost function.
- $\nabla J(\theta)$ represents the gradient.

**Learning objectives:**

- Understand cost functions.
- Explore the role of learning rates.
- Observe iterative parameter updates.
- Understand convergence.
- Experiment with numerical datasets.

### 3. Perceptron

📁 [Explore Perceptron](./perceptron)

The perceptron is one of the earliest artificial neural network models, designed for binary classification.

It learns a linear decision boundary by updating its weights when predictions are incorrect.

**Linear combination:**

$$
z=\sum_{i=1}^{n}w_ix_i+b
$$

**Activation function:**

$$
\hat{y}=
\begin{cases}
1 & z\geq 0\\
0 & z<0
\end{cases}
$$

**Weight update rule:**

$$
w_i \leftarrow w_i+\eta(y-\hat{y})x_i
$$

Where:

- $w_i$ represents the weight.
- $x_i$ represents an input feature.
- $\eta$ represents the learning rate.
- $y$ represents the actual class.
- $\hat{y}$ represents the predicted class.

**Learning objectives:**

- Understand artificial neurons.
- Explore binary classification.
- Learn about activation functions.
- Understand perceptron training.
- Explore linear decision boundaries.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Git
- Basic Python programming knowledge
- Familiarity with introductory linear algebra

### 1. Clone the Repository

```bash
git clone https://github.com/ranjiGT/ml_algorithms.git
cd ml_algorithms
```

### 2. Switch to the Development Branch

```bash
git checkout Dev
```

### 3. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies

The scripts may require additional Python packages. Review their imports and install the corresponding dependencies in your virtual environment.

### 5. Explore the Algorithms

For example:

```bash
cd gradient_descent
python Assignment1.py
```

**Note:** Some implementations may expect particular input files, dependencies, or command-line arguments. Review the source code before execution.

---

## 📊 Learning Roadmap

| Topic | Focus | Repository |
|---|---|---|
| Decision Tree | Classification and information gain | [View](./decision_tree) |
| Gradient Descent | Numerical optimization | [View](./gradient_descent) |
| Perceptron | Linear classification | [View](./perceptron) |

### Future Enhancements

- [ ] Add detailed algorithm-specific documentation.
- [ ] Include visualizations of training and convergence.
- [ ] Add Jupyter Notebook tutorials.
- [ ] Introduce unit tests.
- [ ] Add performance comparisons with established ML libraries.
- [ ] Include more machine learning algorithms.
- [ ] Provide step-by-step mathematical derivations.
- [ ] Add interactive learning examples.

---

## 🤝 Contributing

Contributions are welcome!

Whether you are a beginner exploring machine learning or an experienced developer, you can contribute by improving implementations, adding documentation, writing tests, or introducing new examples.

**Contribution workflow:**

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Test your implementation.
5. Submit a pull request with a clear description.

Please avoid committing sensitive information, large generated files, or unnecessary dependencies.

---

## 🌟 Support the Project

If this repository helps you understand machine learning, consider giving it a ⭐ on GitHub.

Your support helps make educational resources more discoverable.

You can also share the repository with students, developers, and machine learning enthusiasts.

---

**⭐ Learn • Code • Experiment • Contribute**
