source("utils.R")

args <- commandArgs(trailingOnly = TRUE)
data_model   <- args[1]  # "M1", "M2", "M3", "B", "IM1", "IM2"
algorithm <- args[2]  # "L1", "L2", "NN", "CEL", "WCEL" 
task_id    <- as.numeric(args[3])

cat(sprintf("Running Task %d | data_model: %s | Model: %s\n", task_id, data_model, algorithm))

set.seed(task_id)
c <- 4
se <- c(0.95, 0.98)
sp <- c(0.98, 0.99)

if (algorithm %in% c("L1", "L2")) {
  Ntr  <- 25000
  train <- generate_group_data(Ntr, c, se, sp, data_model, algorithm)
  X_train  <- train$X
  Y_train  <- train$Y
  Z_train  <- train$Z
  C_train  <- train$C
  B_train  <- train$B
  theta0 <- rep(0, ncol(X_train)) 
  b0 <- 0 
  
} else if (algorithm %in% c("NN","CEL","WCEL")) {
  Ntr <- 20000
  Nva <- 5000
  
  train <- generate_group_data(Ntr, c, se, sp, data_model, algorithm)
  val   <- generate_group_data(Nva, c, se, sp, data_model, algorithm)
  
  X_train  <- train$X
  Y_train  <- train$Y
  Z_train  <- train$Z
  C_train  <- train$C
  B_train  <- train$B
  
  X_val  <- val$X
  Z_val  <- val$Z
  C_val  <- val$C
  B_val  <- val$B
  
  if (algorithm == "WCEL"){
    dfX <- as.data.frame(X_train)
    
    w_df <- dfX |>
      dplyr::group_by(dplyr::across(dplyr::everything())) |>
      dplyr::mutate(count = dplyr::n(),
                    w = 1 / count) |>
      dplyr::ungroup()
    
    # extract w and normalize so that mean(w) = 1
    w <- w_df$w
    w <- w / mean(w)
    weight <- as.matrix(w)
  }
  
  nodes = c(50, 50, 50, 1)
  layers = 4
  activations = c('relu', 'relu', 'relu', 'sigmoid')
  learning_rate = 0.01
  
  if (algorithm %in% c("CEL", "WCEL")) {
    epochs = 20000
  } else if (algorithm == "NN") {
    epochs = 2000 
  }
  
  initial_model <- NULL
}

start_time <- Sys.time()

train_row <- Ntr
p0 <- rep(0.05, train_row)
Ymat <- matrix(-99, nrow = train_row, ncol = 5000)
Y0 <- Y_train[, 1]

best_loglik1 <- -Inf
loglik1 <- 1500 
loglik0 <- 2000
criteria <- 0.5
iteration_count <- 0

while (criteria > 0.00001) {
  iteration_count <- iteration_count + 1
  Y_train[, 1] <- Y0
  loglik0 <- loglik1
  
  for (i in 1:5000) {
    u <- runif(train_row)
    Ymat[, i] <- SampLatent(N = train_row, p = p0, Y = Y_train, Z = Z_train, U = u, se = se, sp = sp, na = 2)
    Y_train[, 1] <- Ymat[, i]
  }
  Ei <- apply(Ymat, 1, mean)
  
  if (algorithm %in% c("L1","L2")) {
    opt_result <- logistic_regression_opt(X = X_train, Ei = Ei, theta0 = theta0, b0 = b0)
    theta0 <- opt_result$theta
    b0 <- opt_result$b
    
    p0 <- exp(X_train %*% theta0 + b0) / (1 + exp(X_train %*% theta0 + b0))
    loglik1 <- loglik_cpp(C = C_train, B = B_train, Z = Z_train, p = p0, Se = se, Sp = sp, k_test = 1:nrow(B_train))
    
    if (loglik1 > best_loglik1) {
      best_loglik1 <- loglik1
      best_iter <- iteration_count
      best_b <- b0
      best_theta <- theta0
    }
    
  } else if (algorithm %in% c("NN","CEL","WCEL")) {
    w_param <- if (algorithm == "WCEL") weight else NULL
    
    loglik_model <- train_neural_network_loglik(
      X_train = X_train, Y_t = matrix(Ei),
      X_val = X_val, C_val = C_val, B_val = B_val, Z_val = Z_val,
      se = se, sp = sp, layers = layers, nodes = nodes, power = w_param,
      activations = activations, learning_rate = learning_rate, 
      epochs = epochs, initial_model = initial_model
    )
    p0 <- forward_propagation(input_data = X_train, model = loglik_model)$output
    loglik1 <- loglik_cpp(C = C_train, B = B_train, Z = Z_train, p = p0, Se = se, Sp = sp, k_test = 1:nrow(B_train))
    initial_model <- loglik_model
    
    if (loglik1 > best_loglik1) {
      best_loglik1 <- loglik1
      best_iter <- iteration_count
    }
  }
  
  criteria <- abs(loglik1 - loglik0) / abs(loglik0)
  
  cat("Iter:", iteration_count, ", loglik:", loglik1, ", criteria:", criteria, "\n")
  cat("Best_iter:", best_iter, ", best_loglik:", best_loglik1, "\n")
}

N_test <- 500

if (data_model %in% c("M1","M2","M3")){
  X_test_1 <- seq(from = -pi/2, to = pi/2, length.out = N_test)
  conditions <- expand.grid(X_test_2 = c(0, 1), X_test_3 = c(0, 1), X_test_4 = c(0, 1))
  conditions <- conditions[rep(seq_len(nrow(conditions)), length.out = N_test), ]
  X_test <- as.matrix(cbind(X_test_1, conditions))
  
  if (data_model == "M1") {
    p.t1 <- -5 + X_test[,1] + 0.5 * X_test[,2] + 1.5 * X_test[,3] + 2 * X_test[,4]
  } else if (data_model == "M2") {
    p.t1 <- -4.5 + 3 * X_test[,1] * (-1.5 + 0.5 * X_test[,2] + 1 * X_test[,3] + 2 * X_test[,4])
  } else if (data_model == "M3") {
    p.t1 <- -4 + 3 * sin(2 * X_test[,1]) * (-1.5 + 0.5 * X_test[,2] + 1 * X_test[,3] + 2 * X_test[,4])
  }
  
} else if (data_model %in% c("B","IM1","IM2")){
  X_test <- seq(from = 0, to = 100, length.out = N_test)
  X_test <- matrix(X_test, ncol = 1)
  p.t1 <- -4 + 3*sin(X_test*pi/100+pi/4)
}


if (algorithm %in% c("L1", "NN", "CEL", "WCEL")) {
  X_new <- X_test
} else if (algorithm == "L2") {
  X_new <- matrix(cbind(
    X_test, 
    X_test[, 1] * X_test[, 2], 
    X_test[, 1] * X_test[, 3], 
    X_test[, 1] * X_test[, 4], 
    X_test[, 2] * X_test[, 3], 
    X_test[, 2] * X_test[, 4], 
    X_test[, 3] * X_test[, 4],
    X_test[, 1] * X_test[, 2] * X_test[, 3], 
    X_test[, 1] * X_test[, 2] * X_test[, 4], 
    X_test[, 1] * X_test[, 3] * X_test[, 4], 
    X_test[, 2] * X_test[, 3] * X_test[, 4],
    X_test[, 1] * X_test[, 2] * X_test[, 3] * X_test[, 4]
  ), ncol = 15)
}

Real_p <- exp(p.t1) / (1 + exp(p.t1))
Real_p <- as.matrix(Real_p)

if (algorithm %in% c("L1", "L2")) {
  predY <- exp(X_new %*% best_theta + best_b) / (1 + exp(X_new %*% best_theta + best_b))
} else if (algorithm %in% c("NN","CEL","WCEL")) {
  predY <- forward_propagation(input_data = X_test, model = initial_model)$output
}

output_EM <- data.frame(X_test, Real_p, predY)

save_directory <- sprintf("%s_%s/results", data_model, algorithm)
dir.create(save_directory, recursive = TRUE, showWarnings = FALSE)

write.csv(output_EM, file = sprintf("%s/task_%d.csv", save_directory, task_id), row.names = FALSE)

end_time <- Sys.time()
running_time <- end_time - start_time

running_time_seconds <- as.numeric(running_time, units = "secs")
hours <- floor(running_time_seconds / 3600)
minutes <- floor((running_time_seconds %% 3600) / 60)
seconds <- round(running_time_seconds %% 60)
print(paste("Running time:", hours, "hours", minutes, "minutes", seconds, "seconds"))

time_log_file <- sprintf("%s/running_time.txt", save_directory)

# Convert the running time to a formatted string
running_time_str <- paste("Task id:", task_id,"- Running time:", hours, "hours", minutes, "minutes", seconds, "seconds\n")
loglik_str <- paste("Best_iter:", best_iter, ", best_loglik:", best_loglik1, "\n")

cat(running_time_str, file = time_log_file, append = TRUE)
cat(loglik_str, file = time_log_file, append = TRUE)