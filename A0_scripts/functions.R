## initialization
theta.init <- function(L=50, sd=1, sdx=1){
  return(list(a=rnorm(n=L, mean=0, sd=sd),
              b=rnorm(n=L, mean=0, sd=sd/sdx),
              c=rnorm(n=L, mean=0, sd=sd),
              d=0))
}

## grad_level-th derivative of activation function
activation <- function(z=0, grad_level=0){
  return(switch(as.character(grad_level),
                              "0" = tanh(z),
                              "1" = 1-tanh(z)^2,
                              "2" = 2 * tanh(z) * (tanh(z)^2-1),
                              "3" = -2 * (tanh(z)^2-1) * (3*tanh(z)^2-1),
                              "4" = 8 * tanh(z) * (tanh(z)^2-1) * (3*tanh(z)^2-2),
                              stop("invalid grad_level")))
}

## k-th derivative (w.r.t. x) of f
f <- function(x, theta, k=0){
  x = as.vector(x); n = length(x)
  U1 = rep(1, n) %*% t(theta$a * theta$b^k) 
  U2 = apply(x %*% t(theta$b) + rep(1,n) %*% t(theta$c), 
             c(1,2), function(z) activation(z, grad_level=k))
  return(apply(U1 * U2, 1, sum) + theta$d * (k==0))
}

## k-th derivative (w.r.t. x) of the gradient df/d\theta
f.grad <- function(x, theta, k=0){
  x = as.vector(x); n = length(x)
  V = x %*% t(theta$b) + rep(1,n) %*% t(theta$c)
  V1 = apply(V, c(1,2), function(z) activation(z, grad_level=k))
  V2 = apply(V, c(1,2), function(z) activation(z, grad_level=k+1))
  Ua = rep(1, n) %*% t(theta$b^k) 
   a = Ua * V1
  Ub1 = k * rep(1,n) %*% t(theta$a * theta$b^(k-1))
  Ub2 = x %*% t(theta$a * theta$b^k)
   b = Ub1 * V1 + Ub2 * V2
  Uc = rep(1, n) %*% t(theta$a * theta$b^k)
   c = Uc * V2
   d = rep(if(k==0) 1 else 0, n)
  return(list(a=a, b=b, c=c, d=d, k=k))
}

## loss function (Tukey's biweight)
nu <- function(z, delta=1, grad_level=0){
  z = as.vector(z)
  .v0 <- function(z) z^2
  .v1 <- function(z) 2*z
  return(switch(as.character(grad_level),
         "0" = sapply(z, .v0),
         "1" = sapply(z, .v1),
         stop("invalid grad_level")))
}

alpha <- function(x, y, theta, delta=1){
  L = length(theta$a)
  x = as.vector(x); n = length(x)
  y = as.vector(y)
  nu1 = nu(y - f(x, theta), delta=delta, grad_level=1)
  fg = f.grad(x, theta, k=0)
  a = -apply((nu1 %*% t(rep(1,L))) * fg$a, 2, mean)
  b = -apply((nu1 %*% t(rep(1,L))) * fg$b, 2, mean)
  c = -apply((nu1 %*% t(rep(1,L))) * fg$c, 2, mean)
  d = -mean(nu1 * fg$d)
  return(list(a=a, b=b, c=c, d=d))
}

beta <- function(theta){
  return(lapply(theta, function(z) z/sqrt(length(z))))
}

gamma <- function(z, theta, k=1){
  L = length(theta$a)
  z = as.vector(z)
  v = f(z, theta, k=k)  
  V = v %*% t(rep(1,L))
  U = f.grad(z, theta, k=k)
  a = apply(U$a * V, 2, mean)
  b = apply(U$b * V, 2, mean)
  c = apply(U$c * V, 2, mean)
  d = mean(U$d * v)
  return(list(a=a, b=b, c=c, d=d))
}

theta.mult <- function(theta, mult=1){
  return(lapply(theta, function(z) mult*z))
}

theta.sum <- function(theta1, theta2){
  return(list(a = theta1$a + theta2$a,
              b = theta1$b + theta2$b,
              c = theta1$c + theta2$c,
              d = theta1$d + theta2$d))
}

VRSGD <- function(theta0=NULL, x=NULL, y=NULL, constants, monitor_loss=TRUE){
  theta = theta0; N = length(x); lr = constants$lr0
  if(monitor_loss) monitor=list(loss=NULL, itr=NULL)
  
  pb <- progress_bar$new(total = constants$n_itr)
  for(itr in 1:constants$n_itr){
    pb$tick()
    
    ## adaptively specifying delta
    smp_i = sample(1:N, size=min(N,50), replace=F)

    ## gradient computation  
    smp_i = sample(1:N, size=constants$n, replace=F)
    z = runif(n=constants$m, min=-1/2, max=1/2)
    grad = alpha(x=x[smp_i], y=y[smp_i], theta, delta)
    grad = theta.sum(grad, theta.mult(beta(theta), constants$lambda))
    grad = theta.sum(grad, theta.mult(gamma(z, theta, k=1), constants$eta[1]))
    grad = theta.sum(grad, theta.mult(gamma(z, theta, k=2), constants$eta[2]))
    grad = theta.sum(grad, theta.mult(gamma(z, theta, k=3), constants$eta[3]))
    
    ## current learning rate
    if(itr %% constants$lr_it == 0) lr = lr * constants$lr_dr
    if(itr %% constants$lr_period == 0) lr = constants$lr0
    
    if(monitor_loss && (itr %% 10 == 0)){
      loss_tmp = mean(nu(y - f(x, theta), grad_level=0))
      monitor$loss = append(monitor$loss, loss_tmp)
      monitor$itr = append(monitor$itr, itr)
    }
    
    ## parameter update
    theta = theta.sum(theta, theta.mult(grad, -lr))
  }
  
  return(if(monitor_loss) list(theta=theta, monitor=monitor)
         else list(theta=theta, monitor=NULL))
}
