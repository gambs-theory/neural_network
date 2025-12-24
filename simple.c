// Exemplo simples para portas lógicas: OR, AND

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
  Mat *Y_pred = mat_new(2, 1);

  for (int i = 0; i < LEN(train); i++) {
    X->mat[0][0] = train[i][0];
    // X->mat[1][0] = train[i][1];
    
    Mat *A = mat_mult(W, X);
    Mat *Z = mat_add(A, b);
    
    // mat_print(R);
    Mat *Y = mat_apply(Z, sigmoid, FALSE);
    double y_pred = Y->mat[0][0];
    
    Y_pred->mat[i][0] = y_pred;

    mat_delete(A);
    mat_delete(Z);
    mat_delete(Y);
  }

  return Y_pred;
}

int main(void) {
  // srand(time(0));
  srand(42);
  
  Mat *W = mat_new_random (1, 1); // Not gate

  Mat *b = mat_new (1, 1); // Starts with null vector

  Mat *X = mat_new (1, 1);

  Mat *Y_true = mat_new(2, 1);
  Y_true->mat[0][0] = 1;
  Y_true->mat[1][0] = 0;

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
    double eps = 1e-1;
    for (int i = 0; i < W->rows; i++) {
      for (int j = 0; j < W->cols; j++) {
        // Perturbation
        Mat *W_eps = mat_copy(W);
        W_eps->mat[i][j] += eps;
        
        // Prediction
        Mat *Y_eps = predict(W_eps, X, b);

        // Derivate of error
        double dW = (mse(Y_eps, Y_true) - err)/ eps;
        
        // Learning
        W->mat[i][j] -= eta * dW;

        // Free memory
        mat_delete(W_eps);
        mat_delete(Y_eps);
      }
    }

    // Bias
    for (int i = 0; i < b->rows; i++) {
      for (int j = 0; j < b->cols; j++) {
        // Perturbation
        Mat *b_eps = mat_copy(b);
        b_eps->mat[i][j] += eps;
        
        // Prediction
        Mat *Y_eps = predict(W, X, b_eps);

        // Derivate of error
        double db = (mse(Y_eps, Y_true) - err)/ eps;
        
        // Learning
        b->mat[i][j] -= eta * db;

        // Free memory
        mat_delete(b_eps);
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