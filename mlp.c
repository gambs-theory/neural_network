#include "metrics.h"
#include "mlp.h"
#include "mat.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

// num_layers must include the dimension of the input
MLP *mlp_new(size_t *arch, size_t num_layers, double lr) {
  MLP *m = (MLP *)malloc(sizeof(MLP));

  m->num_layers = num_layers - 1;
  m->Ws = (Mat **)malloc(m->num_layers * sizeof(Mat *));
  m->As = (Mat **)malloc((m->num_layers + 1) * sizeof(Mat *));
  m->bs = (Mat **)malloc(m->num_layers * sizeof(Mat *));

  m->lr = lr;

  m->As[0] = mat_alloc(1, arch[0]); // Input
  for (size_t l = 0; l < m->num_layers; l++) {
    size_t in  = arch[l];
    size_t out = arch[l + 1];

    m->Ws[l] = mat_alloc(in, out);    // W
    m->bs[l] = mat_alloc(1, out);     // b
    m->As[l + 1] = mat_alloc(1, out); // Hidden Layer and output

    //Initialization
    mat_fill_rand(m->Ws[l]);
    // mat_fill_rand_range(m->Ws[l], -5, 5);
  }

  return m;
}

// Total cost for all training data
double mlp_cost (MLP *m, Mat *X, Mat *y, loss_function_t loss) {
  Mat *y_pred = mat_alloc(X->rows, 1);

  mlp_predict(y_pred, m, X);

  double cost = loss(y_pred, y);
  mat_delete(y_pred);
  return cost;
}

// Calculate finite difference of a Multi-Layer Perceptron
void mlp_finite_diff(MLP *grad, MLP *m, Mat *X, Mat *y, double eps) {
  double err = mlp_cost(m, X, y, se);
  for (int l = 0; l < m->num_layers; l++) {
    // Backpropagation
    for (int i = 0; i < m->Ws[l]->rows; i++) {
      for (int j = 0; j < m->Ws[l]->cols; j++) {
        // Perturbation
        MLP *m_eps = mlp_copy(m);
        MAT_AT(m_eps->Ws[l], i, j) += eps;
        
        // Prediction
        double cost = mlp_cost(m_eps, X, y, se);
        double dW = (cost - err)/ eps;

        MAT_AT(grad->Ws[l], i, j) = dW;
        
        // Free memory
        mlp_destroy(m_eps);
      }
    }

    // Bias
    for (int i = 0; i < m->bs[l]->rows; i++) {
      for (int j = 0; j < m->bs[l]->cols; j++) {
        // Perturbation
        MLP *m_eps = mlp_copy(m);
        MAT_AT(m_eps->bs[l], i, j) += eps;
        
        // Prediction
        double cost = mlp_cost(m_eps, X, y, se);
        double db = (cost - err)/ eps;

        MAT_AT(grad->bs[l], i, j) = db;
        
        // Free memory
        mlp_destroy(m_eps);
      }
    }
  }
}        

void mlp_fit(MLP *m, Mat *X, Mat *y, size_t epochs, double eps) {
  Mat *y_pred = mat_alloc(X->rows, 1); 
  MLP *grad = mlp_copy(m);

  for (int epoch = 0; epoch < epochs; epoch++) {
    mlp_predict(y_pred, m, X);

    // Cost report
    double cost = se(y_pred, y);
    printf ("[+] epoch = %d - loss func regression: %.5lf\n", epoch, cost);

    mlp_finite_diff(grad, m, X, y, eps);
    
    // Learn
    for (int l = 0; l < m->num_layers; l++) {
      // Weights
      for (int i = 0; i < m->Ws[l]->rows; i++) {
        for (int j = 0; j < m->Ws[l]->cols; j++) {
          MAT_AT(m->Ws[l], i, j) -= m->lr * MAT_AT(grad->Ws[l], i, j);
        }
      }

      // Bias
      for (int i = 0; i < m->bs[l]->rows; i++) {
        for (int j = 0; j < m->bs[l]->cols; j++) {
          MAT_AT(m->bs[l], i, j) -= m->lr * MAT_AT(grad->bs[l], i, j);
        }
      }
    }
  }

  mat_delete(y_pred);
  mlp_destroy(grad);
}

void mlp_predict(Mat *pred, MLP *m, Mat *X) {
  if (pred->rows == X->rows) {
    for (int i = 0; i < X->rows; i++) {
      Mat *x = mat_get_row(X, i);
      
      mat_copy(MLP_INPUT(m), x);
      mlp_forward(m);

      MAT_AT(pred, i, 0) = MAT_AT(MLP_OUTPUT(m), 0, 0);

      mat_delete(x);
    }

  } else {
    printf ("prediction failure\n");
  }
}

void mlp_forward(MLP *m) {
  for (int l = 0; l < m->num_layers; l++) {
    mat_mult(m->As[l + 1], m->As[l], m->Ws[l]);       // Multiply
    mat_add(m->As[l + 1], m->As[l + 1], m->bs[l]);    // Add
    mat_apply(m->As[l + 1], m->As[l + 1], sigmoid);   // Activate
  }
}

void mlp_dump(MLP *m) {
  char buffer[256] = {0};
  for (int l = 0; l < m->num_layers; l++) {
    snprintf(buffer, sizeof(buffer), "W[%d]", l);
    mat_print(buffer, m->Ws[l]);
    snprintf(buffer, sizeof(buffer), "b[%d]", l);
    mat_print(buffer, m->bs[l]);

    // Hidden activation
    snprintf(buffer, sizeof(buffer), "a[%d]", l + 1);
    mat_print(buffer, m->As[l + 1]);
  }
}

MLP *mlp_copy (MLP *m) {
  MLP *ret = (MLP *)malloc(sizeof(MLP));

  ret->num_layers = m->num_layers;
  ret->Ws = (Mat **)malloc(ret->num_layers * sizeof(Mat *));
  ret->As = (Mat **)malloc((ret->num_layers + 1) * sizeof(Mat *));
  ret->bs = (Mat **)malloc(ret->num_layers * sizeof(Mat *));

  ret->As[0] = mat_alloc(m->As[0]->rows, m->As[0]->cols);
  mat_copy(ret->As[0], m->As[0]);
  
  for (size_t l = 0; l < ret->num_layers; l++) {
    ret->Ws[l] = mat_alloc(m->Ws[l]->rows, m->Ws[l]->cols);
    ret->As[l + 1] = mat_alloc(m->As[l + 1]->rows, m->As[l + 1]->cols);
    ret->bs[l] = mat_alloc(m->bs[l]->rows, m->bs[l]->cols);

    mat_copy(ret->Ws[l], m->Ws[l]);
    mat_copy(ret->As[l + 1], m->As[l + 1]);
    mat_copy(ret->bs[l], m->bs[l]);
  }

  return ret;
}

void mlp_destroy(MLP *m) {
  mat_delete(m->As[0]);
  for (int l = 0; l < m->num_layers; l++) {
    mat_delete(m->Ws[l]);
    mat_delete(m->bs[l]);
    mat_delete(m->As[l + 1]);
  }

  free(m->Ws);
  free(m->bs);
  free(m->As);

  free(m);
}