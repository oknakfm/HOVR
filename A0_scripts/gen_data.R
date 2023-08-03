N = 1000; N_test = 10000

## linear
for(seed in 1:n_seeds){
  set.seed(seed)
  
  sd = 0.1
  x = runif(n=N, min=-1/2, max=1/2)
  y = x
  e = rnorm(n=N, mean=0, sd=sd)
  write.csv(x=data.frame(x=x, y=y, e=e),
            file=paste0(dir$data,"/1_seed",seed,".csv"), row.names=FALSE)
  
  x_test = seq(-1/2,1/2,length.out = N_test)
  y_test = x_test
  write.csv(x=data.frame(x=x_test, y=y_test), 
            file=paste0(dir$data,"/1_test.csv"), row.names=FALSE)
}

## quadratic
for(seed in 1:n_seeds){
  set.seed(seed)
  
  sd = 0.1
  x = runif(n=N, min=-1/2, max=1/2)
  y = 4 * x^2
  e = rnorm(n=N, mean=0, sd=sd)
  write.csv(x=data.frame(x=x, y=y, e=e),
            file=paste0(dir$data,"/2_seed",seed,".csv"), row.names=FALSE)
  
  x_test = seq(-1/2,1/2,length.out = N_test)
  y_test = 4 * x_test^2
  write.csv(x=data.frame(x=x_test, y=y_test), 
            file=paste0(dir$data,"/2_test.csv"), row.names=FALSE)
}

## cubic
for(seed in 1:n_seeds){
  set.seed(seed)
  
  sd = 0.1
  x = runif(n=N, min=-1/2, max=1/2)
  y = (64/7) * (x+3/8) * x * (x-3/8)
  e = rnorm(n=N, mean=0, sd=sd)
  write.csv(x=data.frame(x=x, y=y, e=e),
            file=paste0(dir$data,"/3_seed",seed,".csv"), row.names=FALSE)
  
  x_test = seq(-1/2,1/2,length.out = N_test)
  y_test = (64/7) * (x_test+3/8) * x_test * (x_test-3/8)
  write.csv(x=data.frame(x=x_test, y=y_test), 
            file=paste0(dir$data,"/3_test.csv"), row.names=FALSE)
}