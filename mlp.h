#ifndef NN_H
#define NN_H

#include <stdio.h>

#include "mat.h"

#define MLP_OUTPUT(m) (m)->As[(m)->num_layers]
#define MLP_INPUT(m)  (m)->As[0]

typedef double (*loss_function_t)(Mat *, Mat *);

// Multi-Layer Perceptron
typedef struct MLP {
  Mat **Ws;             // Weights - n elements
  Mat **As;             // Perceptrons Output (Input included) - n + 1 elements
  Mat **bs;             // Biases - n elements

  size_t num_layers;    // Layers count (Input excluded)
  double lr;            // Learning rate
}
MLP;

MLP *mlp_new(size_t *arch, size_t num_layers, double lr);
double mlp_cost (MLP *m, Mat *X, Mat *y, loss_function_t loss);

// Finite difference fitting
void mlp_finite_diff(MLP *grad, MLP *m, Mat *X, Mat *y, double eps);                  
void mlp_fit(MLP *m, Mat *X, Mat *y, size_t epochs, double eps);

void mlp_predict(Mat *pred, MLP *m, Mat *X);
void mlp_forward(MLP *m);

void mlp_dump(MLP *m);
MLP *mlp_copy(MLP *m);
void mlp_destroy(MLP *m);

#endif//NN_H