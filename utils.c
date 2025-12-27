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

double randlf(void) {
  return (double)rand()/(double)(RAND_MAX);
}

float randf_range(float low, float high) {
  return (high - low) * randf() + low;
}

double randlf_range(double low, double high) {
  return (high - low) * randlf() + low;
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
