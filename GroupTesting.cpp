// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

#include <RcppArmadillo.h>
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
                  IntegerVector k_test) {
    
    NumericVector prob_Z(k_test.size(), NA_REAL);
    for(int k_idx = 0; k_idx < k_test.size(); ++k_idx) {
        int k0 = k_test[k_idx] - 1;
        int group_size = C(k0, 0);
        std::vector<int> ind(group_size);
        for (int i = 0; i < group_size; ++i) ind[i] = C(k0, i+1) - 1;
        
        std::vector<double> ptemp(group_size);
        for(int i = 0; i < group_size; ++i) ptemp[i] = p[ind[i]];
        
        int cj = group_size;
        int num_comb = 1 << cj;
        
        std::vector<double> prod_pi(num_comb, 1.0), SeSp(num_comb, 1.0);
        
        for(int i = 0; i < num_comb; ++i) {
            double pr = 1.0;
            for(int j = 0; j < cj; ++j) {
                if((i >> j) & 1) pr *= ptemp[j];
                else             pr *= (1.0 - ptemp[j]);
            }
            prod_pi[i] = pr;
        }
        
        int Bk = B(k0,0);
        for(int t = 0; t < Bk; ++t) {
            int test_idx = B(k0,t+1) - 1;
            int Zd       = Z(test_idx,0);
            int Yid_size = Z(test_idx,1);
            
            std::vector<int> Yid(Yid_size);
            for(int i=0; i<Yid_size; i++) Yid[i] = Z(test_idx, i+3)-1;
            
            std::vector<int> id(Yid_size);
            for(int i=0; i<Yid_size; i++)
                id[i] = std::distance(ind.begin(), 
                                      std::find(ind.begin(), ind.end(), Yid[i]));
            
            int spi = Z(test_idx,2)-1;
            for(int i = 0; i < num_comb; ++i) {
                bool Zt = false;
                for(int j : id) if((i >> j) & 1) { Zt = true; break; }
                
                double factor;
                if(Zt) factor = Zd ? Se[spi] : (1-Se[spi]);
                else   factor = Zd ? (1-Sp[spi]) : Sp[spi];
                
                SeSp[i] *= factor;
            }
        }
        
        double sum_p = 0.0;
        for(int i = 0; i < num_comb; ++i)
            sum_p += SeSp[i] * prod_pi[i];
        prob_Z[k_idx] = sum_p;
    }
    
    double out = 0.0;
    for(double v : prob_Z) if(!NumericVector::is_na(v)) out += std::log(v);
    return out;
}


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


static void initialize_parameters_cpp(
    int input_size, const IntegerVector& nodes, int layers,
    arma::field<arma::mat>& Ws, arma::field<arma::rowvec>& Bs) {
    
    Ws = arma::field<arma::mat>(layers);
    Bs = arma::field<arma::rowvec>(layers);
    
    int current_input_size = input_size;
    for (int i = 0; i < layers; ++i) {
        int current_output_size = nodes[i];
        double limit = std::sqrt(6.0 / (current_input_size + current_output_size));
        
        Ws(i) = arma::randu<arma::mat>(current_input_size, current_output_size);
        Ws(i) = Ws(i) * 2.0 * limit - limit;
        Bs(i) = arma::zeros<arma::rowvec>(current_output_size);
        
        current_input_size = current_output_size;
    }
}



// [[Rcpp::export]]
List train_neural_network_loglik(
    NumericMatrix X_train, NumericMatrix Y_t,
    NumericMatrix X_val, NumericMatrix C_val, NumericMatrix B_val, NumericMatrix Z_val,
    NumericVector se, NumericVector sp, int layers, IntegerVector nodes, 
    CharacterVector activations, double learning_rate, int epochs,
    Nullable<List> initial_model = R_NilValue,
    Nullable<NumericVector> power = R_NilValue) { 
    
    arma::mat Xtr = as<arma::mat>(X_train);
    arma::mat Ytr = as<arma::mat>(Y_t);
    arma::mat Xval = as<arma::mat>(X_val);
    IntegerVector k_val = seq_len(B_val.nrow());
    
    arma::vec power_train;
    if (power.isNotNull()) {
        power_train = as<arma::vec>(power);
    } else {
        power_train = arma::ones<arma::vec>(Ytr.n_elem);
    }
    
    int input_size = Xtr.n_cols;
    arma::field<arma::mat> Ws;
    arma::field<arma::rowvec> Bs;
    
    if (initial_model.isNotNull()) {
        List model = as<List>(initial_model);
        List w_list = as<List>(model["weights"]);
        List b_list = as<List>(model["biases"]);
        Ws = arma::field<arma::mat>(layers);
        Bs = arma::field<arma::rowvec>(layers);
        for(int i = 0; i < layers; ++i) {
            Ws(i) = as<arma::mat>(w_list[i]);
            Bs(i) = as<arma::rowvec>(b_list[i]);
        }
    } else {
        initialize_parameters_cpp(input_size, nodes, layers, Ws, Bs);
    }
    
    std::vector<int> act_code(layers);
    for(int i = 0; i < layers; ++i) {
        if(activations[i] == "sigmoid") act_code[i] = 0;
        else if(activations[i] == "relu") act_code[i] = 1;
        else stop("Unsupported activation code.");
    }
    
    double best_lv = -1e9;
    int best_epoch = 0;
    arma::field<arma::mat> best_Ws = Ws;
    arma::field<arma::rowvec> best_Bs = Bs;
    
    arma::field<arma::mat> inputs(layers + 1);
    arma::field<arma::mat> Zmat(layers);
    
    arma::mat A_val, Zmat_val;
    arma::vec p_vec;
    NumericVector p;
    double curr_lv;
    
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // --- forward pass ---
        inputs(0) = Xtr;
        for (int i = 0; i < layers; ++i) {
            Zmat(i) = inputs(i) * Ws(i);
            Zmat(i).each_row() += Bs(i);
            if (act_code[i] == 1) inputs(i+1) = arma::clamp(Zmat(i), 0.0, arma::datum::inf);
            else if (act_code[i] == 0) inputs(i+1) = 1.0 / (1.0 + arma::exp(-Zmat(i)));
        }
        
        // --- backward pass ---
        arma::mat YPred = inputs(layers);
        arma::mat deriv = (-(Ytr % (1.0/YPred) - (1.0 - Ytr) % (1.0/(1.0 - YPred))) % power_train) / Ytr.n_elem;
        
        arma::field<arma::mat> deltas(layers);
        for (int i = layers - 1; i >= 0; --i) {
            arma::mat back;
            if (i == layers - 1) back = deriv;
            else back = deltas(i+1) * Ws(i+1).t();
            
            arma::mat deriv_act;
            if (act_code[i] == 1) {
                deriv_act = arma::conv_to<arma::mat>::from(Zmat(i) > 0);
            } else if (act_code[i] == 0) {
                arma::mat S = 1.0 / (1.0 + arma::exp(-Zmat(i)));
                deriv_act = S % (1.0 - S);
            }
            deltas(i) = back % deriv_act;
        }
        
        // --- update parameters ---
        for (int i = 0; i < layers; ++i) {
            Ws(i) -= inputs(i).t() * deltas(i) * learning_rate;
            Bs(i) -= arma::sum(deltas(i), 0) * learning_rate;
        }
        
        // --- validation ---
        A_val = Xval;
        for (int i = 0; i < layers; ++i) {
            Zmat_val = A_val*Ws(i);
            Zmat_val.each_row() += Bs(i);
            if (act_code[i] == 1) A_val = arma::clamp(Zmat_val, 0.0, arma::datum::inf);
            else if (act_code[i] == 0) A_val = 1.0/(1.0 + arma::exp(-Zmat_val));
        }
        p_vec = A_val.col(0);
        p = wrap(p_vec);
        
        curr_lv = loglik_cpp(C_val, B_val, Z_val, p, se, sp, k_val);
        
        if (curr_lv > best_lv) {
            best_lv = curr_lv;
            best_epoch = epoch;
            best_Ws = Ws;
            best_Bs = Bs;
        }
        
        if (epoch % 2000 == 0) {
            Rcpp::Rcout << "Epoch:" << epoch << " lv:" << curr_lv
                        << " best_lv:" << best_lv << " best_epoch:" << best_epoch << "\n";
        }
    }
    
    List best_model = List::create(
        Named("weights") = wrap(best_Ws),
        Named("biases") = wrap(best_Bs),
        Named("activations") = activations
    );
    return best_model;
}