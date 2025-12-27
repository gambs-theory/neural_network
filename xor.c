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

row_3 train[] = {
  // a  b  y 
  {0, 0, 0},
  {0, 1, 1},
  {1, 0, 1},
  {1, 1, 0}
};

// Predict function
Mat *predict (Mat *X, Mat *W_1, Mat *b_1, Mat *W_2, Mat *b_2) {
  Mat *Y_pred = mat_alloc(LEN(train), 1);
  Mat *a_1 = mat_alloc(2, 1);
  Mat *a_2 = mat_alloc(1, 1);

  for (int i = 0; i < LEN(train); i++) {
    MAT_AT(X, 1, 0) = train[i][1];
    MAT_AT(X, 0, 0) = train[i][0];
    
    mat_mult(a_1, W_1, X);
    mat_add(a_1, a_1, b_1);
    mat_apply(a_1, a_1, sigmoid);
    
    mat_mult(a_2, W_2, a_1);
    mat_add(a_2, a_2, b_2);
    mat_apply(a_2, a_2, sigmoid);

    double y_pred = MAT_AT(a_2, 0, 0);
    
    MAT_AT(Y_pred, i, 0) = y_pred;
  }

  mat_delete(a_1);
  mat_delete(a_2);

  return Y_pred;
}

int main(void) {
  // srand(time(0));
  srand(42);
  
  Mat *W_1 = mat_alloc (2, 2); // Not gate
  Mat *b_1 = mat_alloc (2, 1); // Starts with null vector

  Mat *W_2 = mat_alloc (1, 2); // Not gate
  Mat *b_2 = mat_alloc (1, 1); // Starts with null vector

  Mat *X = mat_alloc (2, 1);

  mat_fill_rand_range(W_1, 0, 1);
  mat_fill_rand_range(W_2, 0, 1);

  Mat *Y_true = mat_alloc(LEN(train), 1);
  for (int i = 0; i < LEN(train); i++) {
    MAT_AT(Y_true, i, 0) = train[i][2];
  }

  // Learning Rate
  double eta = 1e-0;

  for (int it = 0; it < 10000; it++) {
    
    Mat *Y_pred = predict(X, W_1, b_1, W_2, b_2);

    // Error
    // double err = mse(Y_pred, Y_true);
    // double err = rmse(Y_pred, Y_true);
    double err = se(Y_pred, Y_true);

    printf ("Error at iteration %d = %lf\n", it, err);

    // Backpropagation - Learning
    // W <- W - eta * dC/dW

    // ======================== FIRST LAYER ===================================
    Mat *W_1_eps = mat_alloc(W_1->rows, W_1->cols);
    Mat *b_1_eps = mat_alloc(b_1->rows, b_1->cols);

    Mat *W_2_eps = mat_alloc(W_2->rows, W_2->cols);
    Mat *b_2_eps = mat_alloc(b_2->rows, b_2->cols);

    double eps = 1e-1;
    for (int i = 0; i < W_1->rows; i++) {
      for (int j = 0; j < W_1->cols; j++) {
        // Perturbation
        mat_copy(W_1_eps, W_1);
        MAT_AT(W_1_eps, i, j) += eps; 

        // Prediction
        Mat *Y_eps = predict(X, W_1_eps, b_1, W_2, b_2);

        // Derivate of error
        // double dW = (mse(Y_eps, Y_true) - err)/ eps;
        // double dW = (rmse(Y_eps, Y_true) - err)/ eps;
        double dW = (se(Y_eps, Y_true) - err)/ eps;

        // Learning
        MAT_AT(W_1, i, j) -= eta * dW;

        // Free memory
        mat_delete(Y_eps);
      }
    }

    // Bias
    for (int i = 0; i < b_1->rows; i++) {
      for (int j = 0; j < b_1->cols; j++) {
        // Perturbation
        mat_copy(b_1_eps, b_1);
        MAT_AT(b_1_eps, i, j) += eps;
        
        // Prediction
        Mat *Y_eps = predict(X, W_1, b_1_eps, W_2, b_2);

        // Derivate of error
        // double db = (mse(Y_eps, Y_true) - err)/ eps;
        // double db = (rmse(Y_eps, Y_true) - err)/ eps;
        double db = (se(Y_eps, Y_true) - err)/ eps;

        // Learning
        MAT_AT(b_1, i, j) -= eta * db;

        // Free memory
        mat_delete(Y_eps);
      }
    }

    // ======================== SECOND LAYER ===================================
    for (int i = 0; i < W_2->rows; i++) {
      for (int j = 0; j < W_2->cols; j++) {
        // Perturbation
        mat_copy(W_2_eps, W_2);
        MAT_AT(W_2_eps, i, j) += eps;
        
        // Prediction
        Mat *Y_eps = predict(X, W_1, b_1, W_2_eps, b_2);

        // Derivate of error
        // double dW = (mse(Y_eps, Y_true) - err)/ eps;
        // double dW = (rmse(Y_eps, Y_true) - err)/ eps;
        double dW = (se(Y_eps, Y_true) - err)/ eps;

        // Learning
        MAT_AT(W_2, i, j) -= eta * dW;

        // Free memory
        mat_delete(Y_eps);
      }
    }

    // Bias
    for (int i = 0; i < b_2->rows; i++) {
      for (int j = 0; j < b_2->cols; j++) {
        // Perturbation
        mat_copy(b_2_eps, b_2);
        MAT_AT(b_2_eps, i, j) += eps;
        
        // Prediction
        Mat *Y_eps = predict(X, W_1, b_1, W_2, b_2_eps);

        // Derivate of error
        // double db = (mse(Y_eps, Y_true) - err)/ eps;
        // double db = (rmse(Y_eps, Y_true) - err)/ eps;
        double db = (se(Y_eps, Y_true) - err)/ eps;
        
        // Learning
        MAT_AT(b_2, i, j) -= eta * db;

        // Free memory
        mat_delete(Y_eps);
      }
    }

    // Free mem
    mat_delete(W_1_eps);
    mat_delete(b_1_eps);
    
    mat_delete(W_2_eps);
    mat_delete(b_2_eps);
    
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