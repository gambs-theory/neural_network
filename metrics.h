#ifndef METRICS_H
#define METRICS_H

#include "mat.h"

// Loss function - Column Matrix
double se  (Mat *y_pred, Mat *y_true);  // Squared Error
double mse (Mat *y_pred, Mat *y_true);  // Mean Squared Error
double rmse(Mat *y_pred, Mat *y_true);  // Root Mean Squared Error
double mae (Mat *y_pred, Mat *y_true);  // Mean Absolute Error

#endif//METRICS_H