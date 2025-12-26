#include "mlp.h"
#include "mat.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

// num_layers must include the dimension of the input
MLP *mlp_new(size_t *arch, size_t num_layers, double lr) {
  MLP *m = (MLP *)malloc(sizeof(MLP));

  m->num_layers = num_layers - 1;
  m->layers = (Mat **)malloc(m->num_layers * sizeof(Mat *));
  m->biases = (Mat **)malloc(m->num_layers * sizeof(Mat *));
  m->lr = lr;

  for (size_t l = 0; l < m->num_layers; l++) {
    size_t in  = arch[l];
    size_t out = arch[l + 1];

    m->layers[l] = mat_new_random(out, in);   // W
    m->biases[l] = mat_new(out, 1);           // b
  }

  return m;
}

double mlp_cost (MLP *m, Mat *X, Mat *Y, loss_function_t loss) {
  Mat *y_pred = mat_new(X->rows, 1);
  for (int i = 0; i < X->rows; i++) {
    Mat *x = mat_get_row (X, i);
    Mat *x_t = mat_transpose(x);

    Mat *y = mlp_forward(m, x_t);
    y_pred->mat[i][0] = y->mat[0][0];

    mat_delete(x);
    mat_delete(x_t);
    mat_delete(y);
  }

  double cost = loss(y_pred, Y);
  mat_delete(y_pred);

  return cost;
}

// Learning through finite diff strategy
/**
 * Slow
 */
void mlp_finite_diff_fit(MLP *m, Mat *X, Mat *Y, size_t epochs, double eps) {
  for (int epoch = 0; epoch < epochs; epoch++) {
    Mat *y_pred = mat_new(X->rows, 1);

    for (int i = 0; i < X->rows; i++) {
      Mat *x = mat_get_row(X, i);
      Mat *x_t = mat_transpose(x);
      Mat *y = mlp_forward(m, x_t);

      y_pred->mat[i][0] = y->mat[0][0];

      // Free
      mat_delete(x);
      mat_delete(x_t);
      mat_delete(y);
    }

    // Report error
    double err = mlp_cost (m, X, Y, se);
    printf("epoch %d - loss: %.5lf\n", epoch, err);

    for (int l = 0; l < m->num_layers; l++) {
      // Backpropagation
      for (int i = 0; i < m->layers[l]->rows; i++) {
        for (int j = 0; j < m->layers[l]->cols; j++) {
          // Perturbation
          MLP *m_eps = mlp_copy(m);
          m_eps->layers[l]->mat[i][j] += eps;
          
          // Prediction
          double cost = mlp_cost(m_eps, X, Y, se);
          double dW = (cost - err)/ eps;

          m->layers[l]->mat[i][j] -= m->lr * dW;
          // Free memory
          mlp_destroy(m_eps);
        }
      }

      // Bias
      for (int i = 0; i < m->biases[l]->rows; i++) {
        for (int j = 0; j < m->biases[l]->cols; j++) {
          // Perturbation
          MLP *m_eps = mlp_copy(m);
          m_eps->biases[l]->mat[i][j] += eps;
          
          // Prediction
          double cost = mlp_cost(m_eps, X, Y, se);
          double db = (cost - err)/ eps;

          // Learning
          m->biases[l]->mat[i][j] -= m->lr * db;

          // Free memory
          mlp_destroy(m_eps);
        }
      }
    }
  }
}        

Mat *mlp_forward(MLP *m, Mat *X) {
  Mat *Z = X;

  for (int l = 0; l < m->num_layers; l++) {
    Mat *A = mat_mult(m->layers[l], Z);
    
    Z = mat_add(A, m->biases[l]);

    mat_apply(Z, sigmoid, 1);

    mat_delete(A);
  }

  return Z;
}

void mlp_dump(MLP *m) {
  char buffer[256] = {0};
  for (int l = 0; l < m->num_layers; l++) {
    snprintf(buffer, sizeof(buffer), "W[%d]", l);
    mat_print(buffer, m->layers[l]);
    snprintf(buffer, sizeof(buffer), "b[%d]", l);
    mat_print(buffer, m->biases[l]);
  }
}

MLP *mlp_copy (MLP *m) {
  MLP *ret = (MLP *)malloc(sizeof(MLP));

  ret->num_layers = m->num_layers;
  ret->layers = (Mat **)malloc(ret->num_layers * sizeof(Mat *));
  ret->biases = (Mat **)malloc(ret->num_layers * sizeof(Mat *));

  for (size_t l = 0; l < ret->num_layers; l++) {
    ret->layers[l] = mat_copy(m->layers[l]);
    ret->biases[l] = mat_copy(m->biases[l]);
  }

  return ret;
}

void mlp_destroy(MLP *m) {
  for (int l = 0; l < m->num_layers; l++) {
    mat_delete(m->layers[l]);
    mat_delete(m->biases[l]);
  }

  free(m->layers);
  free(m->biases);

  free(m);
}