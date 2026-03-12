#include <Rcpp.h>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace Rcpp;

// [[Rcpp::export]]
double loglik_cpp(NumericMatrix C, 
                  NumericMatrix B, 
                  NumericMatrix Z,
                  NumericVector p, 
                  NumericVector Se, 
                  NumericVector Sp, 
                  NumericVector k_test) {
    
    // prob.Z <- rep(NA,length(k.test))
    NumericVector prob_Z(k_test.size(), NA_REAL);
    
    // for(k in k.test){
    for(int k_idx = 0; k_idx < k_test.size(); ++k_idx) {
        int k = k_test[k_idx] - 1; // Adjust for 0-based indexing in C++
        
        // ind <- C[k,2:(C[k,1]+1)]
        int group_size = C(k, 0);
        IntegerVector ind(group_size);
        for (int i = 0; i < group_size; ++i) {
            ind[i] = C(k, i + 1) - 1; // Adjust for 0-based indexing
        }
        
        // ptemp <- p[ind]
        NumericVector ptemp(ind.size());
        for(int i = 0; i < ind.size(); ++i) {
            ptemp[i] = p[ind[i]];
        }
        
        // cj <- length(ind)
        int cj = ind.size();
        
        // tmat <- expand.grid(replicate(cj, 0:1, simplify = FALSE))
        int num_combinations = std::pow(2, cj);
        std::vector<std::vector<int>> tmat(num_combinations, std::vector<int>(cj));
        for(int i = 0; i < num_combinations; ++i) {
            for(int j = 0; j < cj; ++j) {
                tmat[i][j] = (i >> j) & 1;
            }
        }
        
        // prod.pi <- apply(t(t(tmat)*ptemp) + t(t(abs(1-tmat))*(1-ptemp)),1,prod)
        std::vector<double> prod_pi(num_combinations);
        for(int i = 0; i < num_combinations; ++i) {
            double prod = 1.0;
            for(int j = 0; j < cj; ++j) {
                prod *= tmat[i][j] == 1 ? ptemp[j] : 1 - ptemp[j];
            }
            prod_pi[i] = prod;
        }
        
        // SeSp <- 1
        std::vector<double> SeSp(num_combinations, 1.0);
        
        // for(t in 1:B[k,1]){
        for(int t = 0; t < B(k, 0); ++t) {
            // Zd <- Z[B[k,(t+1)],1]
            int test_idx = B(k, t + 1) - 1;
            int Zd = Z(test_idx, 0);
            int Yid_size = Z(test_idx, 1);
            
            // Yid <- Z[B[k,(t+1)],4:(Z[B[k,(t+1)],2]+3)]
            IntegerVector Yid(Yid_size);
            for (int i = 0; i < Yid_size; ++i) {
                Yid[i] = Z(test_idx, i + 3) - 1; // Adjust for 0-based indexing
            }
            
            // id <- match(Yid, ind)
            std::vector<int> id(Yid.size());
            for(int i = 0; i < Yid.size(); ++i) {
                id[i] = std::distance(ind.begin(), std::find(ind.begin(), ind.end(), Yid[i]));
            }
            
            // Zt <- apply(as.matrix(tmat[,id]),1,sum)>0
            for(int i = 0; i < num_combinations; ++i) {
                bool Zt = false;
                for(size_t j = 0; j < id.size(); ++j) {
                    Zt = Zt || (tmat[i][id[j]] == 1);
                }
                
                // SeSp <- (Zt*Zd*Se[Z[B[k,(t+1)],3]] + Zt*(1-Zd)*(1-Se[Z[B[k,(t+1)],3]]) 
                //          + (1-Zt)*Zd*(1-Sp[Z[B[k,(t+1)],3]]) +(1-Zt)*(1-Zd)*Sp[Z[B[k,(t+1)],3]])*SeSp
                double SeSp_val = 0.0;
                int se_sp_idx = Z(test_idx, 2) - 1; // Adjust for 0-based indexing
                if(Zt) {
                    SeSp_val = Zd ? Se[se_sp_idx] : 1 - Se[se_sp_idx];
                } else {
                    SeSp_val = Zd ? 1 - Sp[se_sp_idx] : Sp[se_sp_idx];
                }
                SeSp[i] *= SeSp_val;
            }
        }
        
        // prob.Z[k] <- sum(SeSp*prod.pi)
        double prob_sum = 0.0;
        for(int i = 0; i < num_combinations; ++i) {
            prob_sum += SeSp[i] * prod_pi[i];
        }
        prob_Z[k_idx] = prob_sum;
    }
    
    // loglik <- sum(log(na.omit(prob.Z)))
    double loglik = 0.0;
    for(int i = 0; i < prob_Z.size(); ++i) {
        if(!NumericVector::is_na(prob_Z[i])) {
            loglik += std::log(prob_Z[i]);
        }
    }
    
    return loglik;
}
