# My_neural_network
Neural Network from scratch in Python

If we call:

- Output: $Y$
- Input: $X$
- Weight: $W$
- Bias: $B$
- Activation function: $f(z = W\cdot X + B)$
- Cost function: $C$

## Forward Propagation
In forward propagation, for each layer:

\begin{aligned}
\displaystyle X &\rightarrow \boxed{\pmb{layer}} \rightarrow Y \\[10pt]
Y &= f(W \cdot X + B)
\end{aligned}

The output for one layer is the input for the next layer.

**At the end of the forward propagation, we can calculate the error as the derivative of the cost function with respect to the output for the neural network model output. We will call it the output gradient $\frac{\partial C}{\partial Y}$ .** 

## Backward Propagation
In backward propagation, for each layer:

\begin{aligned}
\displaystyle\frac{\partial C}{\partial X} &\leftarrow \boxed{\pmb{layer}} \leftarrow \frac{\partial C}{\partial Y} \\[10pt]
\frac{\partial C}{\partial z} &= f'(z) \odot \frac{\partial C}{\partial Y} \\[10pt]
\frac{\partial C}{\partial w} &= \frac{\partial C}{\partial z} \cdot X^T \\[10pt]
\frac{\partial C}{\partial B} &= \frac{\partial C}{\partial Y} \\[10pt]
\frac{\partial C}{\partial X} &= w^T \cdot \frac{\partial C}{\partial z} \quad (\text{the new output gradient for the previous layer})
\end{aligned} 


