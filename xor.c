// Exemplo simples para portas lógicas: OR, AND

#include "utils.h"
#include "mat.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define TRUE 1
#define FALSE 0

typedef float row_3[3];

row_3 train[] = {
  // a  b  y 
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 0}
};

// Predict function
Mat *predict (Mat *X, Mat *W_1, Mat *b_1, Mat *W_2, Mat *b_2) {
  Mat *Y_pred = mat_new(LEN(train), 1);

  for (int i = 0; i < LEN(train); i++) {
    X->mat[0][0] = train[i][0];
    X->mat[1][0] = train[i][1];
    
    Mat *A_1 = mat_mult(W_1, X);
    Mat *Z_1 = mat_add(A_1, b_1);
    
    // mat_print(R);
    // mat_apply(Z_1, sigmoid, TRUE);
    mat_apply(Z_1, sigmoid, 1);
    
    Mat *A_2 = mat_mult(W_2, Z_1);
    Mat *Y = mat_add(A_2, b_2);
    
    // mat_print(R);
    // mat_apply(Y, sigmoid, TRUE);
    mat_apply(Y, sigmoid, 1);

    double y_pred = Y->mat[0][0];
    
    Y_pred->mat[i][0] = y_pred;

    mat_delete(A_1);
    mat_delete(Z_1);

    mat_delete(A_2);
    mat_delete(Y);
  }

  return Y_pred;
}

int main(void) {
  // srand(time(0));
  srand(42);
  
  Mat *W_1 = mat_new_random (2, 2); // Not gate
  Mat *b_1 = mat_new (2, 1); // Starts with null vector

  Mat *W_2 = mat_new_random (1, 2); // Not gate
  Mat *b_2 = mat_new (1, 1); // Starts with null vector

  Mat *X = mat_new (2, 1);

  Mat *Y_true = mat_new(LEN(train), 1);
  for (int i = 0; i < LEN(train); i++) {
    Y_true->mat[i][0] = train[i][2];
  }

  // Learning Rate
  double eta = 1e-0;

  for (int it = 0; it < 2000; it++) {
    
    Mat *Y_pred = predict(X, W_1, b_1, W_2, b_2);

    // Error
    // double err = mse(Y_pred, Y_true);
    // double err = rmse(Y_pred, Y_true);
    double err = se(Y_pred, Y_true);

    printf ("Error at iteration %d = %lf\n", it, err);

    // Backpropagation - Learning
    // W <- W - eta * dC/dW

    // ======================== FIRST LAYER ===================================
    double eps = 1e-1;
    for (int i = 0; i < W_1->rows; i++) {
      for (int j = 0; j < W_1->cols; j++) {
        // Perturbation
        Mat *W_1_eps = mat_copy(W_1);
        W_1_eps->mat[i][j] += eps;
        
        // Prediction
        Mat *Y_eps = predict(X, W_1_eps, b_1, W_2, b_2);

        // Derivate of error
        // double dW = (mse(Y_eps, Y_true) - err)/ eps;
        // double dW = (rmse(Y_eps, Y_true) - err)/ eps;
        double dW = (se(Y_eps, Y_true) - err)/ eps;

        // Learning
        W_1->mat[i][j] -= eta * dW;

        // Free memory
        mat_delete(W_1_eps);
        mat_delete(Y_eps);
      }
    }

    // Bias
    for (int i = 0; i < b_1->rows; i++) {
      for (int j = 0; j < b_1->cols; j++) {
        // Perturbation
        Mat *b_1_eps = mat_copy(b_1);
        b_1_eps->mat[i][j] += eps;
        
        // Prediction
        Mat *Y_eps = predict(X, W_1, b_1_eps, W_2, b_2);

        // Derivate of error
        // double db = (mse(Y_eps, Y_true) - err)/ eps;
        // double db = (rmse(Y_eps, Y_true) - err)/ eps;
        double db = (se(Y_eps, Y_true) - err)/ eps;

        // Learning
        b_1->mat[i][j] -= eta * db;

        // Free memory
        mat_delete(b_1_eps);
        mat_delete(Y_eps);
      }
    }

    // ======================== SECOND LAYER ===================================
    for (int i = 0; i < W_2->rows; i++) {
      for (int j = 0; j < W_2->cols; j++) {
        // Perturbation
        Mat *W_2_eps = mat_copy(W_2);
        W_2_eps->mat[i][j] += eps;
        
        // Prediction
        Mat *Y_eps = predict(X, W_1, b_1, W_2_eps, b_2);

        // Derivate of error
        // double dW = (mse(Y_eps, Y_true) - err)/ eps;
        // double dW = (rmse(Y_eps, Y_true) - err)/ eps;
        double dW = (se(Y_eps, Y_true) - err)/ eps;

        // Learning
        W_2->mat[i][j] -= eta * dW;

        // Free memory
        mat_delete(W_2_eps);
        mat_delete(Y_eps);
      }
    }

    // Bias
    for (int i = 0; i < b_2->rows; i++) {
      for (int j = 0; j < b_2->cols; j++) {
        // Perturbation
        Mat *b_2_eps = mat_copy(b_2);
        b_2_eps->mat[i][j] += eps;
        
        // Prediction
        Mat *Y_eps = predict(X, W_1, b_1, W_2, b_2_eps);

        // Derivate of error
        // double db = (mse(Y_eps, Y_true) - err)/ eps;
        // double db = (rmse(Y_eps, Y_true) - err)/ eps;
        double db = (se(Y_eps, Y_true) - err)/ eps;
        
        // Learning
        b_2->mat[i][j] -= eta * db;

        // Free memory
        mat_delete(b_2_eps);
        mat_delete(Y_eps);
      }
    }

    // Free mem
    mat_delete(Y_pred);
  }

  MAT_PRINT (W_1);
  MAT_PRINT (b_1);

  MAT_PRINT (W_2);
  MAT_PRINT (b_2);

  Mat *Y_pred = predict(X, W_1, b_1, W_2, b_2);
  MAT_PRINT(Y_pred);
  mat_delete(Y_pred);

  mat_delete(Y_true);
  mat_delete(W_1);
  mat_delete(b_1);
  
  mat_delete(W_2);
  mat_delete(b_2);
  mat_delete(X);
}