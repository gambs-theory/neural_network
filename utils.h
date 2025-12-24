#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#define MAX(x, y) x > y? x : y 
#define MIN(x, y) x < y? x : y

#define LEN(arr) sizeof(arr)/ sizeof(arr[0])

#include "mat.h"

// Generate random float values 
float randf(void);

// Activation function (double)
double relu    (double x);
double softplus(double x);
double sigmoid (double x);

// Loss function - Column Matrix
double se  (Mat *y_pred, Mat *y_true);
double mse (Mat *y_pred, Mat *y_true);
double rmse(Mat *y_pred, Mat *y_true);

#endif // PERCEPTRON_H