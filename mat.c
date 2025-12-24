#include "mat.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Allocate a Mat "object"
Mat *mat_new (size_t rows, size_t cols) {
  Mat *mat = (Mat *)malloc(sizeof(Mat));
  mat->rows = rows;
  mat->cols = cols;

  mat->mat = (double **)malloc(rows * sizeof(double *));
  for (int i = 0; i < rows; i++) {
    mat->mat[i] = (double *)calloc(cols, sizeof(double));
  }
  return mat;
}

Mat *mat_ones(size_t rows, size_t cols) {
  Mat *mat = (Mat *)malloc(sizeof(Mat));
  mat->rows = rows;
  mat->cols = cols;

  mat->mat = (double **)malloc(rows * sizeof(double *));
  for (int i = 0; i < rows; i++) {
    mat->mat[i] = (double *)calloc(cols, sizeof(double));

    // Fill 1 values
    for (int j = 0; j < cols; j++)
      mat->mat[i][j] = 1;
  }
  return mat;
}

Mat *mat_identity(size_t order) {
  Mat *A = mat_new(order, order);
  for (int i = 0; i < order; i++) {
    A->mat[i][i] = 1;
  }
  return A;
}

Mat *mat_diag(double *arr, size_t n) {
  Mat *C = mat_new (n, n);
  for (int i = 0; i < n; i++) {
    C->mat[i][i] = arr[i];
  }
  return C;
}

Mat *mat_new_random (size_t rows, size_t cols) {
  Mat *mat = (Mat *)malloc(sizeof(Mat));
  mat->rows = rows;
  mat->cols = cols;

  mat->mat = (double **)malloc(rows * sizeof(double *));
  for (int i = 0; i < rows; i++) {
    mat->mat[i] = (double *)calloc(cols, sizeof(double));

    // Fill random values
    for (int j = 0; j < cols; j++)
      mat->mat[i][j] += randf();
  }
  return mat;
}

// Apply a function to each element in the mat->mat
Mat *mat_apply(Mat *m, rr_function func, bool inplace) {
  Mat *ret = inplace? m : mat_new(m->rows, m->cols);
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      ret->mat[i][j] = func(m->mat[i][j]);
    }
  }
  return ret;
}

Mat *mat_mult(Mat *A, Mat *B) {
  if (A->cols != B->rows) {
    perror("Multiplication error: Invalid dimensions\n");
    return NULL; // (void *)0
  }

  Mat *C = mat_new (A->rows, B->cols);

  for (int i = 0; i < A->rows; i++) {
    for (int j = 0; j < B->cols; j++) {
      for (int k = 0; k < A->cols; k++) {
        C->mat[i][j] += A->mat[i][k] * B->mat[k][j];
      }
    }
  }

  return C;
}

Mat *mat_scalar_mult(Mat *A, double cte, bool inplace) {
  Mat *C = inplace? A : mat_new (A->rows, A->cols);

  for (int i = 0; i < C->rows; i++) {
    for (int j = 0; j < C->cols; j++) {
      C->mat[i][j] = cte * A->mat[i][j];
    }
  }

  return C;
}

Mat *mat_add(Mat *A, Mat *B) {
  if (A->rows == B->rows && A->cols == B->cols) {
    size_t r = A->rows;
    size_t c = A->cols;

    Mat *C = mat_new(r, c);

    for (int i = 0; i < r; i++) {
      for (int j = 0; j < c; j++) {
        C->mat[i][j] = A->mat[i][j] + B->mat[i][j];
      }
    }

    return C;
  }
  else {
    perror("Add failure: Different number of dimensions\n");
    return NULL;
  }
}

Mat *mat_transpose(Mat *A) {
  Mat *C = mat_new (A->cols, A->rows);
  for (int i = 0; i < C->rows; i++) {
    for (int j = 0; j < C->cols; j++) {
      C->mat[i][j] = A->mat[j][i];
    }
  }
  return C;
}

Mat *mat_hadamard(Mat *A, Mat *B) {
  if (A->rows == B->rows && A->cols == B->cols) {
    size_t r = A->rows;
    size_t c = A->cols;

    Mat *C = mat_new(r, c);

    for (int i = 0; i < r; i++) {
      for (int j = 0; j < c; j++) {
        C->mat[i][j] = A->mat[i][j] * B->mat[i][j];
      }
    }

    return C;
  }
  else {
    return NULL;
  }
}

Mat *mat_copy(Mat *m) {
  Mat *copy = mat_new (m->rows, m->cols);

  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      copy->mat[i][j] = m->mat[i][j];
    }
  }

  return copy;
}

Mat *mat_get_row(Mat *m, size_t row) {
  Mat *ret = mat_new(1, m->cols);
  for (int i = 0; i < m->cols; i++) {
    ret->mat[0][i] = m->mat[row][i];
  }

  return ret;
}

Mat *mat_get_col(Mat *m, size_t col) {
  Mat *ret = mat_new(m->rows, 1);
  for (int i = 0; i < m->rows; i++) {
    ret->mat[i][0] = m->mat[i][col];
  }

  return ret;
}

void mat_print (const char *name, Mat *m) {
  printf ("%s = [\n", name);
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      printf ("\t%.2f", m->mat[i][j]);
    }
    printf ("\n");
  }
  printf ("]\n");
}

void mat_delete(Mat *m) {
  for (int i = 0; i < m->rows; i++) {
    free(m->mat[i]);
  }

  free(m->mat);
  free(m);
}