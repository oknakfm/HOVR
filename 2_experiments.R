## Higher-Order Total Variation (HOTV)
## A. Okuno (ISM, okuno@ism.ac.jp)
## Augst 3rd, 2023

## ----------------------------------------------------
working_dir = getwd()
## ----------------------------------------------------

## dependencies
require("progress")
require("parallel"); require("foreach"); require("doParallel"); require("doRNG")

## directories
dir = list(
  main = (tmp <- working_dir),
  scripts = paste0(tmp,"/A0_scripts"),
  data = paste0(tmp,"/A1_data"), 
  computed = paste0(tmp,"/A2_computed")
)
lapply(dir, function(z) dir.create(z, showWarnings=FALSE))

## CPU cores for parallel computation
N.Cores = detectCores() 

## number of experiments (random seeds)
n_seeds = 5

## number of training samples
N=100

## loading scripts
source(paste0(dir$scripts,"/gen_data.R"))
source(paste0(dir$scripts,"/functions.R"))

constants = vector(mode='list', length=8)
n_itr = 2*10**4

## Baseline (no regularization)
constants[[1]] = list(L = 200, ## number of hidden units
                 lambda = 0, ## reg. coef. for beta
                 eta = c(0,0,0), ## reg. coef. for gamma 
                 n = 5, ## num. of subsampling for alpha
                 m = 5, ## num. of subsampling for gamma
                 lr0 = 10**(-3), ## initial learning rate
                 lr_dr = 0.9, ## decay rate
                 lr_it = 25, ## decay interval
                 lr_period = 10**3, ## period of cyclic decay
                 n_itr = n_itr ## num. of SGD iteration
)

## Baseline (ridge regularization)
constants[[2]] = list(L = 200, ## number of hidden units
                  lambda = 10**(-1), ## reg. coef. for beta
                  eta = c(0,0,0), ## reg. coef. for gamma 
                  n = 5, ## num. of subsampling for alpha
                  m = 5, ## num. of subsampling for gamma
                  lr0 = 10**(-3), ## initial learning rate
                  lr_dr = 0.9, ## decay rate
                  lr_it = 25, ## decay interval
                  lr_period = 10**3, ## period of cyclic decay
                  n_itr = n_itr ## num. of SGD iteration
)

constants[[3]] = list(L = 200, ## number of hidden units
                      lambda = 10**(-2), ## reg. coef. for beta
                      eta = c(0,0,0), ## reg. coef. for gamma 
                      n = 5, ## num. of subsampling for alpha
                      m = 5, ## num. of subsampling for gamma
                      lr0 = 10**(-3), ## initial learning rate
                      lr_dr = 0.9, ## decay rate
                      lr_it = 25, ## decay interval
                      lr_period = 10**3, ## period of cyclic decay
                      n_itr = n_itr ## num. of SGD iteration
)

constants[[4]] = list(L = 200, ## number of hidden units
                      lambda = 10**(-3), ## reg. coef. for beta
                      eta = c(0,0,0), ## reg. coef. for gamma 
                      n = 5, ## num. of subsampling for alpha
                      m = 5, ## num. of subsampling for gamma
                      lr0 = 10**(-3), ## initial learning rate
                      lr_dr = 0.9, ## decay rate
                      lr_it = 25, ## decay interval
                      lr_period = 10**3, ## period of cyclic decay
                      n_itr = n_itr ## num. of SGD iteration
)

## Proposal (variation regularization)
constants[[5]] = list(L = 200, ## number of hidden units
                  lambda = 0, ## reg. coef. for beta
                  eta = c(10**(-1),0,0), ## reg. coef. for gamma 
                  n = 5, ## num. of subsampling for alpha
                  m = 5, ## num. of subsampling for gamma
                  lr0 = 10**(-3), ## initial learning rate
                  lr_dr = 0.9, ## decay rate
                  lr_it = 25, ## decay interval
                  lr_period = 10**3, ## period of cyclic decay
                  n_itr = n_itr ## num. of SGD iteration
)

constants[[6]] = list(L = 200, ## number of hidden units
                      lambda = 0, ## reg. coef. for beta
                      eta = c(0,10**(-3),0), ## reg. coef. for gamma 
                      n = 5, ## num. of subsampling for alpha
                      m = 5, ## num. of subsampling for gamma
                      lr0 = 10**(-3), ## initial learning rate
                      lr_dr = 0.9, ## decay rate
                      lr_it = 25, ## decay interval
                      lr_period = 10**3, ## period of cyclic decay
                      n_itr = n_itr ## num. of SGD iteration
)

constants[[7]] = list(L = 200, ## number of hidden units
                      lambda = 0, ## reg. coef. for beta
                      eta = c(0,0,10**(-5)), ## reg. coef. for gamma 
                      n = 5, ## num. of subsampling for alpha
                      m = 5, ## num. of subsampling for gamma
                      lr0 = 10**(-3), ## initial learning rate
                      lr_dr = 0.9, ## decay rate
                      lr_it = 25, ## decay interval
                      lr_period = 10**3, ## period of cyclic decay
                      n_itr = n_itr ## num. of SGD iteration
)

## smaller NN
constants[[8]] = list(L = 20, ## number of hidden units
                      lambda = 0, ## reg. coef. for beta
                      eta = c(0,0,0), ## reg. coef. for gamma 
                      n = 5, ## num. of subsampling for alpha
                      m = 5, ## num. of subsampling for gamma
                      lr0 = 10**(-3), ## initial learning rate
                      lr_dr = 0.9, ## decay rate
                      lr_it = 25, ## decay interval
                      lr_period = 10**3, ## period of cyclic decay
                      n_itr = n_itr ## num. of SGD iteration
)


## ==============
##  experiments
## ==============
for(f_id in 1:3){
  dir.create(paste0(dir$computed,"/f",f_id))
  for(seed_id in 1:n_seeds){
    
    ## dataset
    df = read.csv(paste0(dir$data,"/",f_id,"_seed",seed_id,".csv"), header=T)
    x=df$x; mu=df$y; y=mu+df$e; 
    x=x[1:N]; y=y[1:N]; mu=mu[1:N]
    
    ## experiments
    registerDoParallel(N.Cores)
    registerDoRNG(123)
    foreach(id = 1:8, .export="progress_bar") %dopar% {
      ## initialization: parameter
      theta0 = theta.init(L=constants[[id]]$L, sd=1, sdx=sd(x))
      
      ## Variance-reduced SGD
      .sgd = VRSGD(theta0=theta0, x=x, y=y, constants=constants[[id]])
      save(file=paste0(dir$computed,"/f",f_id,"/seed=",seed_id,"_id=",id,".RData"), 
           theta0, x, y, constants, .sgd)
    }
    stopImplicitCluster()
  }
}


## ==============================
##  Predictive correlation (PC)
## ==============================
PC = vector(mode='list', length=3)
ex_grid = expand.grid(1:8, 1:n_seeds)

for(f_id in 1:3){
  test = read.csv(paste0(dir$data,"/",f_id,"_test.csv"))
  registerDoParallel(N.Cores)
  .tmp = foreach(j = 1:nrow(ex_grid), .combine=cbind) %dopar% {
    xx =test$x[1001:9000]; yy = test$y[1001:9000]
    id = ex_grid[j,1]
    seed = ex_grid[j,2]
    load(paste0(dir$computed,"/f",f_id,"/seed=",seed,"_id=",id,".RData"))
    y_pred = f(x=xx, theta=.sgd$theta, k=0)
    cor(y_pred, yy)
  }
  PC[[f_id]] = .tmp
  stopImplicitCluster()
}

PC_mu = PC_sd = matrix(0, 8, 3)
colnames(PC_mu) = colnames(PC_sd) = paste0("f",1:3)
rownames(PC_mu) = rownames(PC_sd) = paste0("setting[",1:8,"]")

for(f_id in 1:3){
  .tmp = matrix(PC[[f_id]],8,10)
  PC_mu[,f_id] = apply(.tmp,1,mean)
  PC_sd[,f_id] = apply(.tmp,1,sd)
}
PC_mu = signif(PC_mu,digits=3)
PC_sd = signif(PC_sd,digits=3)

## print the results
matrix(paste0(PC_mu,"\\pm",PC_sd),8,3)