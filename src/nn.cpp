// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace Rcpp;
double loglik_cpp_in(NumericMatrix C, 
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
      for(int j = 0; j < cj; ++j)
        pr *= ((i>>j)&1) ? ptemp[j] : (1-ptemp[j]);
      prod_pi[i] = pr;
    }
    
    int Bk = B(k0,0);
    for(int t = 0; t < Bk; ++t) {
      int test_idx = B(k0,t+1) -1;
      int Zd       = Z(test_idx,0);
      int Yid_size = Z(test_idx,1);
      
      std::vector<int> Yid(Yid_size);
      for(int i=0;i<Yid_size;i++) Yid[i] = Z(test_idx, i+3)-1;
      
      std::vector<int> id(Yid_size);
      for(int i=0;i<Yid_size;i++)
        id[i] = std::distance(ind.begin(), 
                              std::find(ind.begin(), ind.end(), Yid[i]));
      
      int spi = Z(test_idx,2)-1;
      for(int i = 0; i < num_comb; ++i) {
        bool Zt = false;
        for(int j : id) if((i>>j)&1) { Zt = true; break; }
        
        double factor;
        if(Zt)      factor = Zd ? Se[spi] : (1-Se[spi]);
        else        factor = Zd ? (1-Sp[spi]) : Sp[spi];
        
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
static void initialize_parameters_cpp(
    int input_size,
    const IntegerVector& nodes,
    int layers,
    arma::field<arma::mat>& Ws,
    arma::field<arma::rowvec>& Bs) {
  Ws = arma::field<arma::mat>(layers);
  Bs = arma::field<arma::rowvec>(layers);
  
  Ws(0) = arma::randn(input_size, nodes[0]) * std::sqrt(1.0 / nodes[0]);
  Bs(0) = arma::rowvec(nodes[0], arma::fill::zeros);
  
  for (int i = 1; i < layers; ++i) {
    Ws(i) = arma::randn(nodes[i-1], nodes[i]) * std::sqrt(2.0 / nodes[i-1]);
    Bs(i) = arma::rowvec(nodes[i], arma::fill::zeros);
  }
}

static NumericVector mat_to_NumericVector(const arma::mat& M) {
  NumericVector v( M.n_elem );
  std::copy( M.begin(), M.end(), v.begin() );
  if (M.n_cols > 1)    
    v.attr("dim") = Dimension( M.n_rows, M.n_cols );
  return v;
}

List train_neural_network_loglik(
    NumericMatrix X_train,
    NumericVector Y_t,
    NumericMatrix X_val,
    NumericMatrix C_val,
    NumericMatrix B_val,
    NumericMatrix Z_val,
    NumericVector se,
    NumericVector sp,
    int layers,
    IntegerVector nodes,
    CharacterVector activations,
    double learning_rate,
    int epochs,
    List initial_model) {
  arma::mat Xtr  = as<arma::mat>(X_train);
  arma::vec Ytr = as<arma::vec>(Y_t);
  arma::mat Xval  = as<arma::mat>(X_val);
  IntegerVector k_val = seq_len(B_val.nrow());
  
  std::vector<int> act_code(layers);
  for (int i = 0; i < layers; ++i) {
    std::string s = as<std::string>(activations[i]);
    act_code[i] = (s == "sigmoid" ? 0 : (s == "relu" ? 1 : -1));
  }
  
  arma::field<arma::mat> Ws(layers), inputs(layers+1), etas(layers), deltas(layers), best_Ws(layers);
  arma::field<arma::rowvec> Bs(layers), best_Bs(layers);
  arma::mat ZL, deriv, S, back, A_val, A_train, Zmat_val, Zmat_train, grad;
  arma::vec YPred, loss_grad, p_vec;
  NumericVector p;
  double curr_lv;
  int best_epoch = 0;
  inputs(0) = Xtr;
  
  if (initial_model.size()) {
    Ws = as<arma::field<arma::mat>>(initial_model["weights"]);
    Bs = as<arma::field<arma::rowvec>>(initial_model["biases"]);
  } else {
    Rcpp::Rcout << "Xtr.n_cols = " << Xtr.n_cols << std::endl;
    initialize_parameters_cpp(Xtr.n_cols, nodes, layers, Ws, Bs);
  }
  best_Ws = Ws;
  best_Bs = Bs;
  
  A_val = Xval;
  for (int i = 0; i < layers; ++i) {
    Zmat_val = A_val*Ws(i);
    Zmat_val.each_row() += Bs(i);
    if (act_code[i] == 1) A_val = arma::clamp(Zmat_val, 0.0, arma::datum::inf); // relu
    else if (act_code[i] == 0)A_val= 1.0/(1.0 + arma::exp(-Zmat_val)); //sigmoid
    else stop("Unsupported activation code: " + std::to_string(act_code[i]));
  }
  p_vec = A_val.col(0);
  p = mat_to_NumericVector(p_vec);  
  curr_lv = loglik_cpp_in(C_val, B_val, Z_val, p, se, sp, k_val);
  double best_lv = curr_lv;

  for (int epoch = 1; epoch <= epochs; ++epoch) {
    // --- feedforward ---
    A_train = Xtr;
    for (int i = 0; i < layers; ++i) {
      Zmat_train = A_train * Ws(i);
      Zmat_train.each_row() += Bs(i);
      etas(i) = Zmat_train;
      if (act_code[i] == 1) A_train  = arma::clamp(Zmat_train, 0.0, arma::datum::inf); // relu
      else if (act_code[i] == 0) A_train = 1.0/(1.0 + arma::exp(-Zmat_train));//sigmoid
      else stop("Unsupported activation code: " + std::to_string(act_code[i]));
      inputs(i+1) = A_train;
    }
    
    // --- backpropagation ---
    // output layer
    YPred = arma::clamp(A_train, 1e-15, 1.0 - 1e-15);
    loss_grad = -(Ytr % (1.0/YPred) - (1.0 - Ytr) % (1.0/(1.0 - YPred))) / Ytr.n_elem;
    ZL = etas(layers-1);
    if (act_code[layers-1] == 1) {// ReLU
      deriv = arma::conv_to<arma::mat>::from(ZL > 0.0);
    }
    else if (act_code[layers-1] == 0){// sigmoid
      S = 1.0 / (1.0 + arma::exp(-ZL));  
      deriv = S % (1.0 - S);
    }
    else {
      stop("Unsupported activation code: " + std::to_string(act_code[layers-1]));
    }
    deltas(layers-1) = loss_grad % deriv;
    // hidden layers
    for (int i = layers - 2; i >= 0; --i) {// 
      back = deltas(i+1) * Ws(i+1).t();
      if (act_code[i] == 1) {// ReLU 
        deriv = arma::conv_to<arma::mat>::from(etas(i) > 0.0);
      }
      else if (act_code[i] == 0) {// sigmoid 
        S = 1.0 / (1.0 + arma::exp(-etas(i)));
        deriv = S % (1.0 - S);
      }
      else {
        stop("Unsupported activation code: " + std::to_string(act_code[i]));
      }
      deltas(i) = back % deriv;
    }
    
    // --- parameter updates ---
    for (int i = 0; i < layers; ++i) {
      Ws(i) -= inputs(i).t() * deltas(i) * learning_rate;
      Bs(i) -= arma::sum(deltas(i), 0) * learning_rate;
    }
    
    // --- lv ---
    A_val = Xval;
    for (int i = 0; i < layers; ++i) {
      Zmat_val = A_val*Ws(i);
      Zmat_val.each_row() += Bs(i);
      if (act_code[i] == 1) A_val = arma::clamp(Zmat_val, 0.0, arma::datum::inf); // relu
      else if (act_code[i] == 0) A_val= 1.0/(1.0 + arma::exp(-Zmat_val)); //sigmoid
      else stop("Unsupported activation code: " + std::to_string(act_code[i]));
    }
    p_vec = A_val.col(0);
    p = mat_to_NumericVector(p_vec);  
    curr_lv = loglik_cpp_in(C_val, B_val, Z_val, p, se, sp, k_val);
    
    if (curr_lv > best_lv) {
      best_lv    = curr_lv;
      best_epoch = epoch;
      best_Ws = Ws;
      best_Bs = Bs;
    }
    
    // ---check and print---
    if (epoch % 200 == 0) {
      Rcpp::Rcout << "Epoch:"      << epoch
                  << " lv:"        << curr_lv
      //<< " lt:"        << curr_lt
        << " best_lv:"   << best_lv
        << " best_epoch:"<< best_epoch
        << "\n";
    }
  }
  List best_model = List::create(
    Named("weights")     = wrap(best_Ws),
    Named("biases")      = wrap(best_Bs),
    Named("activations") = activations
  );
  return best_model;
}

