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
n_seeds = 1

## loading scripts
source(paste0(dir$scripts,"/gen_data.R"))
source(paste0(dir$scripts,"/functions.R"))

## dataset
df = read.csv(paste0(dir$data,"/2_seed1.csv"), header=T)
x=df$x; mu=df$y; y=mu+df$e; 
N=100
x=x[1:N]; y=y[1:N]; mu=mu[1:N]

constants = vector(mode='list', length=3)

## Baseline (no regularization)
constants[[1]] = list(L = 200, ## number of hidden units
                 lambda = 10**(-1), ## reg. coef. for beta
                 eta = c(0,0,0), ## reg. coef. for gamma 
                 n = 5, ## num. of subsampling for alpha
                 m = 5, ## num. of subsampling for gamma
                 lr0 = 10**(-3), ## initial learning rate
                 lr_dr = 0.9, ## decay rate
                 lr_it = 25, ## decay interval
                 lr_period = 10**3, ## period of cyclic decay
                 n_itr = 2*10**4 ## num. of SGD iteration
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
                  n_itr = 2*10**4 ## num. of SGD iteration
)

## Baseline (ridge regularization)
constants[[3]] = list(L = 200, ## number of hidden units
                      lambda = 10**(-3), ## reg. coef. for beta
                      eta = c(0,0,0), ## reg. coef. for gamma 
                      n = 5, ## num. of subsampling for alpha
                      m = 5, ## num. of subsampling for gamma
                      lr0 = 10**(-3), ## initial learning rate
                      lr_dr = 0.9, ## decay rate
                      lr_it = 25, ## decay interval
                      lr_period = 10**3, ## period of cyclic decay
                      n_itr = 2*10**4 ## num. of SGD iteration
)

## Proposal (variation regularization)
constants[[4]] = list(L = 200, ## number of hidden units
                  lambda = 10**(-5), ## reg. coef. for beta
                  eta = c(0,0,0), ## reg. coef. for gamma 
                  n = 5, ## num. of subsampling for alpha
                  m = 5, ## num. of subsampling for gamma
                  lr0 = 10**(-3), ## initial learning rate
                  lr_dr = 0.9, ## decay rate
                  lr_it = 25, ## decay interval
                  lr_period = 10**3, ## period of cyclic decay
                  n_itr = 2*10**4 ## num. of SGD iteration
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
                      n_itr = 2*10**4 ## num. of SGD iteration
)

## Proposal (variation regularization)
constants[[6]] = list(L = 200, ## number of hidden units
                      lambda = 0, ## reg. coef. for beta
                      eta = c(0,10**(-3),0), ## reg. coef. for gamma 
                      n = 5, ## num. of subsampling for alpha
                      m = 5, ## num. of subsampling for gamma
                      lr0 = 10**(-3), ## initial learning rate
                      lr_dr = 0.9, ## decay rate
                      lr_it = 25, ## decay interval
                      lr_period = 10**3, ## period of cyclic decay
                      n_itr = 2*10**4 ## num. of SGD iteration
)

## Proposal (variation regularization)
constants[[7]] = list(L = 200, ## number of hidden units
                      lambda = 0, ## reg. coef. for beta
                      eta = c(0,0,10**(-5)), ## reg. coef. for gamma 
                      n = 5, ## num. of subsampling for alpha
                      m = 5, ## num. of subsampling for gamma
                      lr0 = 10**(-3), ## initial learning rate
                      lr_dr = 0.9, ## decay rate
                      lr_it = 25, ## decay interval
                      lr_period = 10**3, ## period of cyclic decay
                      n_itr = 2*10**4 ## num. of SGD iteration
)

## Proposal (ridge + variation regularization)
constants[[8]] = list(L = 20, ## number of hidden units
                      lambda = 10**(-3), ## reg. coef. for beta
                      eta = c(0,0,0), ## reg. coef. for gamma 
                      n = 5, ## num. of subsampling for alpha
                      m = 5, ## num. of subsampling for gamma
                      lr0 = 10**(-3), ## initial learning rate
                      lr_dr = 0.9, ## decay rate
                      lr_it = 25, ## decay interval
                      lr_period = 10**3, ## period of cyclic decay
                      n_itr = 2*10**4 ## num. of SGD iteration
)

## experiments
registerDoParallel(N.Cores)
registerDoRNG(123)
foreach(id = 1:8, .export="progress_bar") %dopar% {
  ## initialization: parameter
  theta0 = theta.init(L=constants[[id]]$L, sd=1, sdx=sd(x))
  
  ## Variance-reduced SGD
  .sgd = VRSGD(theta0=theta0, x=x, y=y, constants=constants[[id]])
  save(file=paste0(dir$computed,"/",id,".RData"), 
       theta0, x, y, constants, .sgd)
}
stopImplicitCluster()

## plot
for(id in 1:8){
  
  name = if(f_id==1) "linear" else if(f_id==2) "quadratic" else "cubic"
  
  load(file=paste0(dir$computed,"/",name,"/",id,".RData"))
  
  pdf(file=paste0(dir$computed,"/",name,"/",id,"_regression.pdf"), 
      width=4, height=4)
  par(mar = c(4, 4, 1, 1), oma = c(0,0,0,0))  
  xl = c(-1/2, 1/2)
  xx = seq(-1/2, 1/2, length.out=100)
  yp = f(xx, .sgd$theta)
  yl = range(y)
  plot(x,y, xlim=xl, ylim=yl, xlab="x", ylab="y")
  color = if(5 <= id && id <= 7) "blue" else "black"
  par(new=T)
  plot(xx, yp, col=color, type="l", xlim=xl, ylim=yl, xlab=" ", ylab=" ", 
       xaxt="n", yaxt="n", lwd=2)
  dev.off()

  pdf(file=paste0(dir$computed,"/",name,"/",id,"_variation.pdf"), 
      width=4, height=4)
  par(mar = c(4, 4, 1, 1), oma = c(0,0,0,0))  
  xx = seq(-0.5, 0.5, 0.01)
  y0 = f(xx, .sgd$theta, k=0)
  y1 = f(xx, .sgd$theta, k=1)
  y2 = f(xx, .sgd$theta, k=2)
  y3 = f(xx, .sgd$theta, k=3)
  
  yl = range(y0, y1, y2, y3)
  plot(xx, y0, type="l", ylim=yl, lty=1, xlab="x", ylab="k-th variation")
  par(new=T)
  plot(xx, y1, type="l", ylim=yl, lty=2, xlab=" ", ylab=" ", xaxt="n", yaxt="n")
  par(new=T)
  plot(xx, y2, type="l", ylim=yl, lty=3, xlab=" ", ylab=" ", xaxt="n", yaxt="n")
  par(new=T)
  plot(xx, y3, type="l", ylim=yl, lty=4, xlab=" ", ylab=" ", xaxt="n", yaxt="n")
  legend("topright",legend=c("k=0","k=1","k=2","k=3"),
         lty=1:4)
  
  dev.off()
}
