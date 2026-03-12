#include <Rcpp.h>
using namespace Rcpp;

// This function calculates a new vector based on the input parameters
// without modifying the original 'p' vector.
// [[Rcpp::export]]
NumericVector SampLatent(int N, NumericVector p, NumericMatrix Y, NumericMatrix Z,
                         NumericVector U, NumericVector se, NumericVector sp, int na) {
  NumericVector p_temp = clone(p); // Create a copy of 'p'
  NumericVector WW(N); // Initialize the result vector
  for(int i = 0; i < N; i++) {
    float pi1 = p_temp(i);
    float pi2 = 1 - pi1;
    int np = Y(i, 2 - 1);
    for(int l = 0; l < np; l++) {
      int j = Y(i, (l + 2));
      int Zj = Z(j - 1, 0);
      int cj = Z(j - 1, 1);
      int tid = Z(j - 1, 2);
      float sej = se(tid - 1);
      float spj = sp(tid - 1);
      int ybar = 0;
      Y(i, 1 - 1) = 0;
      for(int t = 0; t < cj; t++) {
        int id = Z(j - 1, (t + 3));
        ybar = ybar + Y(id - 1, 1 - 1);
      }
      pi1 = pi1 * (sej * Zj + (1 - sej) * (1 - Zj));
      if(ybar > 0) {
        pi2 = pi2 * (sej * Zj + (1 - sej) * (1 - Zj));
      } else {
        pi2 = pi2 * ((1 - spj) * Zj + spj * (1 - Zj));
      }
    }
    float pistar = (pi1 / (pi1 + pi2));
    if(U(i) < pistar) {
      Y(i, 1 - 1) = 1;
    } else {
      Y(i, 1 - 1) = 0;
    }
    WW(i) = Y(i, 1 - 1);
  }  
  return WW;
}
