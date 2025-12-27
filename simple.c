// Exemplo simples para portas lógicas: OR, AND

#include "metrics.h"
#include "utils.h"
#include "mat.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define TRUE 1
#define FALSE 0

typedef float row_3[3];
typedef float row_2[2];

// 2 input gates
// row_3 train[] = {
//   // a  b  y 
//   {0, 0, 0},
//   {0, 1, 1},
//   {1, 0, 1},
//   {1, 1, 1}
// };

// not gate
row_2 train[] = {
  {0, 1},
  {1, 0}
};


// Predict function
Mat *predict (Mat *W, Mat *X, Mat *b) {
  // NN allocation
  Mat *Z = mat_alloc(1, 1);
  Mat *Y_pred = mat_alloc(2, 1);

  for (int i = 0; i < LEN(train); i++) {
    MAT_AT(X, 0, 0) = train[i][0];

    mat_mult(Z, W, X);
    mat_add(Z, Z, b);
    mat_apply(Z, Z, sigmoid);

    double y_pred = MAT_AT(Z, 0, 0);
    
    MAT_AT(Y_pred, i, 0) = y_pred;
  }

  mat_delete(Z);
  return Y_pred;
}

int main(void) {
  // srand(time(0));
  srand(42);
  
  Mat *W = mat_alloc (1, 1); // Not gate
  mat_fill_rand_range(W, -5, 5);

  Mat *b = mat_alloc (1, 1); // Starts with null vector

  Mat *X = mat_alloc (1, 1);

  Mat *Y_true = mat_alloc(2, 1);
  MAT_AT(Y_true, 0, 0) = 1;
  MAT_AT(Y_true, 1, 0) = 0;

  MAT_PRINT (W);
  MAT_PRINT (b);

  // Learning Rate
  double eta = 1e-0;

  for (int it = 0; it < 2000; it++) {
    
    Mat *Y_pred = predict(W, X, b);

    // Error
    double err = mse(Y_pred, Y_true);
    printf ("Error at iteration %d = %.3lf\n", it, err);

    // Backpropagation - Learning
    // W <- W - eta * dC/dW
    Mat *W_eps = mat_alloc(W->rows, W->cols);
    Mat *b_eps = mat_alloc(b->rows, b->cols);

    double eps = 1e-1;
    for (int i = 0; i < W->rows; i++) {
      for (int j = 0; j < W->cols; j++) {
        // Perturbation
        mat_copy(W_eps, W);

        MAT_AT(W_eps, i, j) += eps;
        
        // Prediction
        Mat *Y_eps = predict(W_eps, X, b);

        // Derivate of error
        double dW = (mse(Y_eps, Y_true) - err)/ eps;
        
        // Learning
        MAT_AT(W, i, j) -= eta * dW;

        // Free memory
        mat_delete(Y_eps);
      }
    }

    // Bias
    for (int i = 0; i < b->rows; i++) {
      for (int j = 0; j < b->cols; j++) {
        // Perturbation
        mat_copy(b_eps, b);
        MAT_AT(b_eps, i, j) += eps;
        
        // Prediction
        Mat *Y_eps = predict(W, X, b_eps);

        // Derivate of error
        double db = (mse(Y_eps, Y_true) - err)/ eps;
        
        // Learning
        MAT_AT(b, i, j) -= eta * db;

        // Free memory
        mat_delete(Y_eps);
      }
    }

    // Free mem
    mat_delete(Y_pred);
  }

  MAT_PRINT (W);
  MAT_PRINT (b);

  Mat *Y_pred = predict(W, X, b);
  MAT_PRINT(Y_pred);
  mat_delete(Y_pred);

  mat_delete(Y_true);
  mat_delete(W);
  mat_delete(b);
  mat_delete(X);
}