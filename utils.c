#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "mat.h"
#include "utils.h"

// y = σ(W'*X + b)
// MSE: (1/n) sum (y - y*)^2
// optimizer (grad): W(t+1) = W(t) - ε grad(MSE(W(t)))

// Generate random float numbers
float randf(void) {
  return (float)rand()/(float)(RAND_MAX);
}

double relu(double x) {
  return MAX(0, x);
}

double softplus (double x) {
  return log(1. + exp(x));
}

double sigmoid (double x) {
  return 1./ (1. + exp(-x));
}

// Loss function

// Square Residual or Square Error
double se (Mat *y_pred, Mat *y_true) {
  if (y_pred->rows != y_true->rows) {
    perror("Dimesions conflict y_pred vs y_true\n");
    return 1e9 + 7;
  }

  double cum_se = 0.0;
  for (int i = 0; i < y_pred->rows; i++) {
    cum_se += pow((y_pred->mat[i][0] - y_true->mat[i][0]), 2);
  }

  return cum_se;
}

// Mean Squared Error
double mse(Mat *y_pred, Mat *y_true) {
  if (y_pred->rows != y_true->rows) {
    perror("Dimesions conflict y_pred vs y_true\n");
    return 1e9 + 7;
  }
  return se(y_pred, y_true)/ y_pred->rows;
}

// Root Mean Squared Error
double rmse(Mat *y_pred, Mat *y_true) {
  if (y_pred->rows != y_true->rows) {
    perror("Dimesions conflict y_pred vs y_true\n");
    return 1e9 + 7;
  }

  return sqrt(mse(y_pred, y_true));
}