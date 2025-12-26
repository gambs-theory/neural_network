#ifndef MAT_H
#define MAT_H

#include <stdio.h>
#include <stdbool.h>

#define MAT_PRINT(var) mat_print(#var, var)

// f: R->R ~ rr function
typedef double (*rr_function)(double);

typedef struct {
  double **mat;
  size_t rows, cols;
}
Mat;

Mat *mat_new (size_t rows, size_t cols);                  // Return 0 matrix
Mat *mat_new_random (size_t rows, size_t cols);           // Return random matrix
Mat *mat_ones(size_t rows, size_t cols);                  // Return ones matrix
Mat *mat_identity(size_t order);                          // Return identity matrix
Mat *mat_diag(double *arr, size_t n);                     // Return a diagonal matrix

Mat *mat_apply(Mat *m, rr_function func, bool inplace);   // Apply a function for each element (double -> double)
Mat *mat_mult(Mat *A, Mat *B);                            // Multiply two matrices
Mat *mat_scalar_mult(Mat *A, double cte, bool inplace);   // Multiply a matrix by a scalar
Mat *mat_add(Mat *A, Mat *B);                             // Add two matrix
Mat *mat_transpose(Mat *A);                               // Transpose
Mat *mat_hadamard(Mat *A, Mat *B);                        // Hadamard or Schur product (A O B) ~ diag(A) * B

Mat *mat_copy(Mat *m);                                    // Copy a matrix
Mat *mat_get_row(Mat *m, size_t row);                     // Row getter
Mat *mat_get_col(Mat *m, size_t col);                     // Col getter
void mat_print(const char *name, Mat *m);                 // Print the matrix
void mat_delete(Mat *m);                                  // Free the memory

#endif // MAT_H