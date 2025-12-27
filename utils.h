#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#define MAX(x, y) x > y? x : y 
#define MIN(x, y) x < y? x : y

#define LEN(arr) sizeof(arr)/ sizeof(arr[0])

#include "mat.h"

// Generate random float values 
float randf(void);
double randlf(void);

float randf_range(float low, float high);
double randlf_range(double low, double high);

// Activation function (double)
double relu    (double x);
double softplus(double x);
double sigmoid (double x);

#endif // PERCEPTRON_H