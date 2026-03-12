# utils.R
library(Rcpp)
library(RcppArmadillo)


sourceCpp("src/loglik.cpp")
sourceCpp("src/nn.cpp")
sourceCpp("src/SampLatent.cpp")


Dorfman.decode.diff.error <- function(Y.true, Se, Sp, c) {
  N <- length(Y.true)
  Jmax <- N + N / c
  J <- 1
  
  Y <- matrix(-99, nrow = N, ncol = 4)
  Z <- matrix(-99, nrow = Jmax, ncol = c + 3)
  
  for (j in 1:(N / c)) {
    prob <- ifelse(sum(Y.true[((j-1)*c+1):(j*c)]) > 0, Se[1], 1 - Sp[1])
    Z[J, 1] <- rbinom(n = 1, size = 1, prob = prob)
    Z[J, 2] <- c
    Z[J, 3] <- 1
    Z[J, 4:(c+3)] <- ((j-1)*c+1):(j*c)
    Y[((j-1)*c+1):(j*c), 1] <- 0
    Y[((j-1)*c+1):(j*c), 2] <- 1
    Y[((j-1)*c+1):(j*c), 3] <- J
    J <- J + 1
    if (Z[J-1, 1] == 1) {
      for (k in ((j-1)*c+1):(j*c)) {
        prob <- ifelse(Y.true[k] > 0, Se[2], 1 - Sp[2])
        Z[J, 1] <- rbinom(n = 1, size = 1, prob = prob)
        Z[J, 2] <- 1
        Z[J, 3] <- 2
        Z[J, 4] <- k
        Y[k, 1] <- Z[J, 1]
        Y[k, 2] <- 2
        Y[k, 4] <- J
        J <- J + 1
      }
    }
  }
  
  J <- J - 1
  Z <- Z[1:J,]
  
  return(list("Z" = Z, "Y" = Y))
}


generate_group_data <- function(N, c, se, sp, data_model, algorithm) {
  

  X1 <- runif(N, -pi/2, pi/2)
  X2 <- rbinom(N, size = 1, prob = 0.5)
  X3 <- rbinom(N, size = 1, prob = 0.5)
  X4 <- rbinom(N, size = 1, prob = 0.5)
  
  if (algorithm %in% c("L1", "NN")){
    X  <- cbind(X1, X2, X3, X4)
  } else if (algorithm == "L2"){
    X <- matrix(cbind(
      X1, X2, X3, X4, 
      X1 * X2, 
      X1 * X3, 
      X1 * X4, 
      X2 * X3, 
      X2 * X4, 
      X3 * X4,
      X1 * X2 * X3, 
      X1 * X2 * X4, 
      X1 * X3 * X4, 
      X2 * X3 * X4,
      X1 * X2 * X3 * X4
    ), ncol = 15)
  }

  if (data_model == "M1") {
    p_t1 <- -5 + X1 + 0.5*X2 + 1.5*X3 + 2*X4
  } else if (data_model == "M2") {
    p_t1 <- -4.5 + 3 * X1 * (-1.5 + 0.5 * X2 + 1 * X3 + 2 * X4)
  } else if (data_model == "M3") {
    p_t1 <- -4 + 3 * sin(2*X1) * (-1.5 + 0.5 * X2 + 1 * X3 + 2 * X4)
  } else {
    stop("Unknown data_model! Please specify 'M1', 'M2', or 'M3'.")
  }
  
  
  Real_p <- exp(p_t1) / (1 + exp(p_t1))            
  Y_true <- rbinom(N, size = 1, prob = Real_p)
  
  
  group_data <- Dorfman.decode.diff.error(Y_true, se, sp, c)
  Z <- group_data$Z    
  Y <- group_data$Y    
  
  
  K <- N / c
  if (K != floor(K)) stop("N must be a multiple of c")
  C <- matrix(NA, nrow = K, ncol = c + 1)
  for (k in seq_len(K)) {
    C[k, ] <- c(c, ((k-1)*c + 1):(k*c))
  }
  
  B <- matrix(NA, nrow = K, ncol = N)  
  for (k in seq_len(K)) {
    mps <- C[k, 1]
    inds <- C[k, 2:(mps+1)]
    temp <- unlist(lapply(inds, function(i) {
      np <- Y[i, 2]
      Y[i, 3:(np+2)]
    }))
    temp <- unique(temp)
    nt   <- length(temp)
    B[k, 1:(nt+1)] <- c(nt, temp)
  }
  
  B <- B[, colMeans(is.na(B)) < 1, drop = FALSE]
  
  
  list(
    X      = X,
    Z      = Z,
    Y      = Y,
    C      = C,
    B      = B,
    Real_p = Real_p 
  )
}


logistic_regression_opt <- function(X, Ei, theta0, b0) {
  # Combine theta and b into a single vector
  initial_params <- c(b0, theta0)
  
  # Objective function for logistic regression with intercept
  obj <- function(params, X, Ei) {
    b <- params[1]
    theta <- params[-1]
    p <- exp(X %*% theta + b) / (1 + exp(X %*% theta + b))
    res <- -sum(Ei * log(p) + (1 - Ei) * log(1 - p))
    return(res)
  }
  
  # Run the optimization
  result <- optim(initial_params, obj, X = X, Ei = Ei)
  
  # Return the optimized parameters
  b <- result$par[1]
  theta <- result$par[-1]
  return(list(theta = theta, b = b))
}



#### Forward propagation function ####
relu <- function(x) {
  return(matrix(pmax(0, x), nrow = nrow(x), ncol = ncol(x)))
}
sigmoid <- function(x) {
  return(1 / (1 + exp(-x)))
}
forward_propagation <- function(input_data, model) {
  inputs <- list(input_data)
  eta <- list()
  layers <- length(model$weights)
  input_data_size <- nrow(input_data)
  
  for (i in 1:layers) {
    layer_input <- inputs[[i]] %*% model$weights[[i]] + t(matrix(model$biases[[i]], ncol = input_data_size, nrow = ncol(model$biases[[i]])))
    eta[[i]] <- layer_input
    if (model$activations[i] == "sigmoid") {
      layer_output <- sigmoid(layer_input)
    } else if (model$activations[i] == "relu") {
      layer_output <- relu(layer_input)
    } else if (model$activations[i] == "tanh") {
      layer_output <- tanh(layer_input)
    }
    inputs[[i + 1]] <- layer_output
  }
  
  return(list(inputs = inputs, eta = eta, output = inputs[[layers + 1]]))
}