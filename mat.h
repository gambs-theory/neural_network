#ifndef MAT_H
#define MAT_H

#include <stdio.h>
#include <stdbool.h>

#define MAT_AT(m, i, j) m->mat[(i)*m->cols + (j)]
#define MAT_PRINT(var) mat_print(#var, var)

// f: R->R ~ rr function
typedef double (*rr_function_t)(double);

typedef struct {
  double *mat;
  size_t rows, cols;
}
Mat;

Mat *mat_alloc (size_t rows, size_t cols);                  // Return allocated space for matrix
void mat_fill(Mat *m, double k);                            // Fill a Matrix with the value k
void mat_fill_rand (Mat *m);                                // Return random matrix
void mat_fill_rand_range(Mat *m, double low, double high);

int mat_check_dims (Mat *A, Mat *B);                        // Return 1 if the matrices has the same dimensions

Mat *mat_identity(size_t order);                            // Return identity matrix
Mat *mat_diag(double *arr, size_t n);                       // Return a diagonal matrix

void mat_apply(Mat *dst, Mat *m, rr_function_t func);       // Apply a function for each element (double -> double)
void mat_mult(Mat *dst, Mat *A, Mat *B);                    // Multiply two matrices
void mat_scalar_mult(Mat *dst, Mat *m, double cte);         // Multiply a matrix by a scalar
void mat_add(Mat *dst, Mat *A, Mat *B);                     // Add two matrix
void mat_transpose(Mat *m);                                 // Transpose the matrix
void mat_hadamard(Mat *dst, Mat *A, Mat *B);                // Hadamard or Schur product (A O B) ~ diag(A) * B

void mat_copy(Mat *dst, Mat *m);                            // Copy a matrix
Mat *mat_get_row(Mat *m, size_t row);                       // Row getter
Mat *mat_get_col(Mat *m, size_t col);                       // Col getter
void mat_print(const char *name, Mat *m);                   // Print the matrix
void mat_delete(Mat *m);                                    // Free the memory

#endif // MAT_H