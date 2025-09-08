# Neural Network from Scratch 🧠

<img width="1400" height="700" alt="image" src="https://github.com/user-attachments/assets/9b953978-e219-4e23-b07e-cb55016f96f2" />

This repository contains my implementation of a neural network from scratch in Python, inspired by the book **"Neural Networks from Scratch" (NNFS)** by Harrison Kinsley and Daniel Kukieła.

## 📌 Goal
The main goal of this project is to **understand the mathematical and computational foundations of neural networks** without relying on frameworks such as TensorFlow or PyTorch — only Python and basic libraries like `numpy` and `matplotlib`.

## 🚀 Progress
- [x] First neuron (simple dense layer)  
- [x] Multiple neurons implementation  
- [x] Activation functions  
- [x] Full forward pass  
- [x] Backpropagation  
- [x] Training on a real dataset  

## 🚀 Features
- Dense (fully connected) layers
- ReLU and Softmax activation functions
- Categorical Cross-Entropy loss
- Backpropagation implementation
- SGD (Stochastic Gradient Descent) optimizer
- Training loop with accuracy and loss tracking
- Model saving and loading with `pickle`
- Visualization:
 - Loss & accuracy curves
 - Decision boundary plot
---

## 📊 Example Results

### Loss & Accuracy over Training
![Loss and Accuracy](Figure_1.png)

### Decision Boundary
![Decision Boundary](Figure_2.png)

---

## 🛠️ Technologies
- **Python 3.10+**
- **NumPy**
- **Matplotlib** (for visualization later)

## 📂 Project Structure

├── main.py # Main implementation
├── model.pkl # Saved model (generated after training)
├── Figure_1.png # Loss/Accuracy curves
├── Figure_2.png # Decision boundary
└── README.md # Project documentation

---

## ⚡ How to Run

1. Clone the repository:

   git clone https://github.com/yourusername/Neural-Network-From-Scratch.git
   cd Neural-Network-From-Scratch


2.Install dependencies: 

   pip install numpy matplotlib nnfs

3. Run the project:

   python main.py

4. After training, you will see:

   Printed loss and accuracy during epochs

   Two plots (loss/accuracy + decision boundary)

   A saved model file model.pkl

---

## 🎯 Next Steps

Add support for more activation functions (Sigmoid, Tanh, Leaky ReLU)

Implement more advanced optimizers (Adam, RMSProp, Momentum)

Expand dataset experiments beyond the spiral dataset

Build a small framework for experimenting with different architectures


## 📚 Acknowledgments

Inspired by the book Neural Networks from Scratch
 by Harrison Kinsley (Sentdex) and Daniel Kukieła.
