## Higher-Order Total Variation (HOTV)
## A. Okuno (ISM, okuno@ism.ac.jp)
## Augst 3rd, 2023

## ----------------------------------------------------
working_dir = getwd()
## ----------------------------------------------------

## dependencies
require("progress")

## directories
dir = list(
  main = (tmp <- working_dir),
  scripts = paste0(tmp,"/A0_scripts"),
  data = paste0(tmp,"/A1_data"), 
  computed = paste0(tmp,"/A2_computed")
)
lapply(dir, function(z) dir.create(z, showWarnings=FALSE))

## number of datasets generated
n_seeds = 1

## loading scripts
source(paste0(dir$scripts,"/gen_data.R"))
source(paste0(dir$scripts,"/functions.R"))

## ------------------------
##  dataset specification
## ------------------------
## loading quadratic datasets
df = read.csv(paste0(dir$data,"/2_seed1.csv"), header=T)
N=100
x=df$x[1:N]; y=(df$y+df$e)[1:N]; 


## --------------
##  NN training
## --------------

## settings
constants = list(L = 50, ## number of hidden units
                 lambda = 0, ## reg. coef. for beta (ridge/weight decay)
                 eta = c(0,0,10**(-5)), ## reg. coef. for gamma (k-TV)
                 n = 5, ## num. of subsampling for alpha
                 m = 5, ## num. of subsampling for gamma
                 lr0 = 10**(-3), ## initial learning rate
                 lr_dr = 0.9, ## decay rate
                 lr_it = 25, ## decay interval
                 lr_period = 10**3, ## period of cyclic decay
                 n_itr = 10**4 ## num. of SGD iteration
)

## initialization of the parameter
theta0 = theta.init(L=constants$L, sd=1, sdx=sd(x))
  
## Variation-regularized SGD
.sgd = VRSGD(theta0=theta0, x=x, y=y, constants=constants,
             monitor_loss=TRUE)

## --------
##   plot
## --------
xl = c(-1/2, 1/2); yl = range(y); 
xx = seq(xl[1], xl[2], length.out=100)
yp = f(xx, .sgd$theta) 

par(mfrow=c(1,2))

## [plot 1] monitored loss functions (of SGD)
plot(.sgd$monitor$itr, .sgd$monitor$loss, type="l", xlab="t", ylab="loss", log="y")

## [plot 2] training data and predictions
plot(x, y, xlim=xl, ylim=yl, xlab="x", ylab="y")
par(new=T)
plot(xx, yp, col="blue", type="l", xlim=xl, ylim=yl, 
     xlab=" ", ylab=" ", xaxt="n", yaxt="n", lwd=2)
