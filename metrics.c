#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "metrics.h"
#include "mat.h"

// Square Residual or Square Error
double se (Mat *y_pred, Mat *y_true) {
  if (!mat_check_dims(y_pred, y_true)) {
    return 1e9 + 7;
  }

  double cum_sum = 0.0;
  for (int i = 0; i < y_pred->rows; i++) {
    cum_sum += pow((MAT_AT(y_pred, i , 0) - MAT_AT(y_true, i, 0)), 2);
  }

  return cum_sum;
}

// Mean Squared Error
double mse(Mat *y_pred, Mat *y_true) {
  if (!mat_check_dims(y_pred, y_true)) {
    return 1e9 + 7;
  }
  return se(y_pred, y_true)/ y_pred->rows;
}

// Root Mean Squared Error
double rmse(Mat *y_pred, Mat *y_true) {
  if (!mat_check_dims(y_pred, y_true)) {
    return 1e9 + 7;
  }

  return sqrt(mse(y_pred, y_true));
}

double mae(Mat *y_pred, Mat *y_true) {
  if (!mat_check_dims(y_pred, y_true)) {
    return 1e9 + 7;
  }

  double cum_sum = 0.0;
  for (int i = 0; i < y_pred->rows; i++) {
    cum_sum += abs(MAT_AT(y_pred, i, 0) - MAT_AT(y_true, i, 0));
  }

  return cum_sum/ y_pred->rows;
}