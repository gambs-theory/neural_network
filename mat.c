#include "mat.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


// Allocate a Mat "object"
Mat *mat_alloc (size_t rows, size_t cols) {
  Mat *mat = (Mat *)malloc(sizeof(Mat));
  mat->rows = rows;
  mat->cols = cols;

  mat->mat = (double *)calloc((rows * cols), sizeof(double));
  return mat;
}

void mat_fill(Mat *m, double k) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      MAT_AT(m, i, j) = k;
    }
  }
}

void mat_fill_rand (Mat *m) {
  for (int i = 0; i < m->rows; i++) {
    // Fill random values
    for (int j = 0; j < m->cols; j++)
      MAT_AT(m, i, j) = randlf();
  }
}

void mat_fill_rand_range (Mat *m, double low, double high) {
  for (int i = 0; i < m->rows; i++) {
    // Fill random values
    for (int j = 0; j < m->cols; j++)
      // mat->mat[i][j] += randf();
      MAT_AT(m, i, j) = randlf_range(low, high);
  }
}

int mat_check_dims (Mat *A, Mat *B) {
  return (A->rows == B->rows && A->cols == B->cols);
}

Mat *mat_identity(size_t order) {
  Mat *m = mat_alloc(order, order);
  for (int i = 0; i < order; i++) {
    MAT_AT(m, i, i) = 1;
  }
  return m;
}

Mat *mat_diag(double *arr, size_t n) {
  Mat *m = mat_alloc (n, n);
  for (int i = 0; i < n; i++) {
    MAT_AT(m, i, i) = arr[i];
  }
  return m;
}

// Apply a function to each element in the mat->mat
void mat_apply(Mat *dst, Mat *m, rr_function_t func) {
  if (mat_check_dims(dst, m)) {
    for (int i = 0; i < m->rows; i++) {
      for (int j = 0; j < m->cols; j++) {
        // ret->mat[i][j] = func(m->mat[i][j]);
        MAT_AT(dst, i, j) = func(MAT_AT(m, i, j));
      }
    }
  } else {
    printf ("Operation not possible - function mat_apply\n");
  }
}

void mat_mult(Mat *dst, Mat *A, Mat *B) {
  if (A->cols == B->rows && dst->rows == A->rows && dst->cols == B->cols) {
    // Temporary matrix - With dst is the same as A or B, this is needed
    Mat *t = mat_alloc(A->rows, B->cols);
    for (int i = 0; i < A->rows; i++) {
      for (int j = 0; j < B->cols; j++) {
        for (int k = 0; k < A->cols; k++) {
          MAT_AT(t, i, j) += MAT_AT(A, i, k) * MAT_AT(B, k, j);
        }
      }
    }  
    mat_copy(dst, t);
    mat_delete(t);
  } else {
    printf("Operation not possible - function mat_mult\n");
  }
}

void mat_scalar_mult(Mat *dst, Mat *m, double cte) {
  if (mat_check_dims(dst, m)) {
    for (int i = 0; i < dst->rows; i++) {
      for (int j = 0; j < dst->cols; j++) {
        MAT_AT(dst, i, j) = cte * MAT_AT(m, i, j);
      }
    }
  } else {
    printf("Operation not possible - function mat_scalar_mult\n");
  }
}

void mat_add(Mat *dst, Mat *A, Mat *B) {
  if (mat_check_dims(A, B) && mat_check_dims(dst, A)) {
    size_t r = A->rows;
    size_t c = A->cols;

    for (int i = 0; i < r; i++) {
      for (int j = 0; j < c; j++) {
        MAT_AT(dst, i, j) = MAT_AT(A, i, j) + MAT_AT(B, i, j);
      }
    }
  }
  else {
    printf("Operation not possible - function mat_add\n");
  }
}

void mat_transpose(Mat *m) {
  // Swap
  size_t t = m->rows;
  m->rows = m->cols;
  m->cols = t;
}

void mat_hadamard(Mat *dst, Mat *A, Mat *B) {
  if (mat_check_dims(A, B) && mat_check_dims(dst, A)) {
    size_t r = A->rows;
    size_t c = A->cols;

    for (int i = 0; i < r; i++) {
      for (int j = 0; j < c; j++) {
        MAT_AT(dst, i, j) = MAT_AT(A, i, j) * MAT_AT(B, i, j);
      }
    }
  }
  else {
    printf("Operation not possible - function mat_hadamard\n");
  }
}

void mat_copy(Mat *dst, Mat *m) {
  if (mat_check_dims(dst, m)) {
    for (int i = 0; i < m->rows; i++) {
      for (int j = 0; j < m->cols; j++) {
        MAT_AT(dst, i, j) = MAT_AT(m, i, j);
      }
    }
  }
  else {
    printf("Operation not possible - function mat_copy\n");
  }
}

Mat *mat_get_row(Mat *m, size_t row) {
  Mat *ret = mat_alloc(1, m->cols);
  for (int i = 0; i < m->cols; i++) {
    MAT_AT(ret, 0, i) = MAT_AT(m, row, i);
  }

  return ret;
}

Mat *mat_get_col(Mat *m, size_t col) {
  Mat *ret = mat_alloc(m->rows, 1);
  for (int i = 0; i < m->rows; i++) {
    MAT_AT(ret, i, 0) = MAT_AT(m, i, col);
  }

  return ret;
}

void mat_print (const char *name, Mat *m) {
  printf ("%s = [\n", name);
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      printf ("\t%.4lf", MAT_AT(m, i, j));
    }
    printf ("\n");
  }
  printf ("]\n");
}

void mat_delete(Mat *m) {
  free(m->mat);
  free(m);
}