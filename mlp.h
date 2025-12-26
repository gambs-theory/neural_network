#ifndef NN_H
#define NN_H

#include <stdio.h>

#include "mat.h"

typedef double (*loss_function_t)(Mat *, Mat *);

// Multi-Layer Perceptron
typedef struct MLP {
  Mat **layers;         // Layers (Input included)
  Mat **biases;         // Biases
  size_t num_layers;    // Layers count (Input excluded)
  double lr;            // Learning rate
}
MLP;

MLP *mlp_new(size_t *arch, size_t num_layers, double lr);
double mlp_cost (MLP *m, Mat *X, Mat *Y, loss_function_t loss);

// Finite difference fitting
void mlp_finite_diff_fit(MLP *m, Mat *X, Mat *Y, size_t epochs, double eps);                  

Mat *mlp_forward(MLP *m, Mat *X);

void mlp_dump(MLP *m);
MLP *mlp_copy(MLP *m);
void mlp_destroy(MLP *m);

#endif//NN_H