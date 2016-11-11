#' Cissy Chan
#' Statistical Pattern Recognition
#' M3S7
#' January 2014 

# all the packages used in this script, uncomment if they aren't already installed
#install.packages(c("mvnmle", "matrixcalc", "nnet", "MASS", "stats", "class"))

library(mvnmle)
library(matrixcalc)
library(nnet)
library(MASS)
library(stats)
library(class)

# read data
data6 <- utils::read.table(
  "http://www2.imperial.ac.uk/~eakc07/S7/data6.dat",
  header=FALSE,
  sep=" ",
  na.strings="NA",
  dec=".",
  strip.white=TRUE
)
#View(data6)

# split into training and test samples
# we will not touch this test sample until the very end
set.seed(1)
v1 <- sample(1:1102, 100, FALSE)
te <- data6[v1, ]
tr <- data6[-v1, ]

# plot all the feature vectors to get a rough idea of what we're dealing with
par(mfrow=c(4, 4),mar=c(1, 2, 2, 1))
for (i in 2:17) {
  plot(tr[, i], xaxt="n", frame.plot=TRUE, main=as.character(i))
}

par(mfrow=c(4,4), mar=c(1, 2, 2, 1))
for (i in 18:29) {
  plot(tr[, i], xaxt="n", frame.plot=TRUE, main=as.character(i))
}

# pick out all the NA values
isna <- is.na(tr)
# see which features have the NA values
colSums(isna)
# see how many NA values each observation has
max(rowSums(isna))
# so each has max. of 1 NA. How many obs contain NA?
sum(isna)
# 85 out of 1002. There are ways to deal with this, like substituting the data with the mean, 
# or conditional mean. However I think we can afford to get rid of that
trfull <- tr[complete.cases(tr), ]
#oh and check that it's not mostly biased towards one class
dim(trfull[trfull[, 1]==0, ])

dev.off()

#---------------------------------------------------------------------------------------------------
# Linear discriminant analysis
#---------------------------------------------------------------------------------------------------

# split data into training (tr1) and test (tr2)
set.seed(1)
v <- sample(1:nrow(trfull), 100, FALSE)
tr1 <- data.frame(trfull[-v, ])
tr2 <- data.frame(trfull[v, ])

# set up function to assign scores for each feature
filter2 <- function(a) {
  lda1 <- lda(as.matrix(trfull[,a]), trfull[, 1], CV=TRUE)$class
  e <- sum(lda1 == trfull[, 1]) / nrow(trfull)
  
  return(e)
}

range <- matrix(2:29, ncol=1)

# run features through filter and plot scores
a <- apply(range, 1, filter2)
plot(range, a, main="Score for each feature", xlab="Features", ylab="Score")

b <- cbind(a, 2:29)
o <- order(b[, 1], decreasing=T)

# set up function to test LDA against increasing number of features
a <- function(d) {
  c <- b[o, ][1:d, 2]  
  lda1 <- lda(as.matrix(trfull[,c]), trfull[, 1], CV=TRUE)$class
  e <- sum(lda1 != trfull[, 1]) / nrow(trfull)
  
  return(e)
}

# run features through function and plot scores
f1 <- apply(as.matrix(2:28), 1, a)
plot(
  x=2:28,
  y=f1,
  main="CV error against number of features",
  xlab="Number of features",
  ylab="CV error"
)

# get the number of features that gives minimum score
which.min(f1)

# the optimal features
c <- b[o, ][1:(which.min(f1)+1), 2]
c

# apparent error
lda1 <- lda(trfull[, c], trfull[, 1])
lda2 <- predict(lda1, trfull[, c])$class
sum(lda2 != trfull[, 1]) / nrow(trfull)

# hold-out error
lda1 <- lda(as.matrix(tr1[, c]), tr1[, 1])
lda2 <- predict(lda1, as.matrix(tr2[, c]))$class
sum(lda2 != tr2[, 1]) / nrow(tr2)

# CV error
lda1 <- lda(as.matrix(trfull[, c]), trfull[, 1], CV=TRUE)$class
sum(lda1 != trfull[, 1]) / nrow(trfull)

# CV all features
lda1 <- lda(as.matrix(trfull[, 2:29]), trfull[, 1], CV=TRUE)$class
sum(lda1 != trfull[, 1])/nrow(trfull)

# 10-fold cv
a <- matrix(0, ncol=10, nrow=91)
b <- 1
count <- 1
while(count < 11) {
  set.seed(1)
  a[, count] <- sample((1:917)[-b],91,FALSE)
  b <- matrix(a[, 1:count], ncol=1)
  count <- count + 1
}

tenfoldlda <- function(d) {
  k <- matrix(a[, -d], ncol=1)
  lda1 <- lda(as.matrix(trfull[k, c]),trfull[k, 1])
  lda2 <- predict(lda1, as.matrix(trfull[a[, d], c]))$class
  error<- sum(lda2 != trfull[a[, d], 1])/nrow(trfull[a[, d], ])
  list(error = error)
}

e <- apply(as.matrix(1:10), 1, tenfoldlda)
errorrates <- numeric(10)
for (i in 1:10) {errorrates[i] <- e[[i]]$error}
sum(errorrates)/10

#---------------------------------------------------------------------------------------------------
# Quadratic discriminant analysis
#---------------------------------------------------------------------------------------------------

filter2 <- function(a) {
  qda1 <- qda(as.matrix(trfull[, a]), trfull[, 1], CV=TRUE)$class
  e <- sum(qda1 == trfull[, 1])/nrow(trfull)
  
  return(e)
}

range <- matrix(2:29, ncol=1)

a <- apply(range, 1, filter2)
plot(range, a, main="Score for each feature", xlab="Features", ylab="Score")

b <- cbind(a, 2:29)
o <- order(b[, 1], decreasing=T)

a <- function(d) {
  c <- b[o, ][1:d, 2]  
  qda1 <- qda(as.matrix(trfull[, c]), trfull[, 1], CV=TRUE)$class
  e <- sum(qda1 != trfull[, 1])/nrow(trfull)
  return(e)
}
f1 <- apply(as.matrix(2:28), 1, a)
plot(
  x=2:28,
  y=f1,
  main="CV error against number of features",
  xlab="Number of features",
  ylab="CV error"
)
which.min(f1)

# the optimal features
c <- b[o, ][1:(which.min(f1)+1), 2]
c

# apparent error
qda1 <- qda(trfull[, c], trfull[, 1])
qda2 <- predict(qda1, trfull[, c])$class
sum(qda2 != trfull[, 1])/nrow(trfull) 

# hold-out error
qda1 <- qda(as.matrix(tr1[, c]),tr1[, 1])
qda2 <- predict(qda1, as.matrix(tr2[, c]))$class
sum(qda2 != tr2[, 1])/nrow(tr2) 

# CV error
qda1 <- qda(as.matrix(trfull[, c]),trfull[, 1], CV=TRUE)$class
sum(qda1 != trfull[, 1])/nrow(trfull)

# CV all features
qda1 <- qda(as.matrix(trfull[, 2:29]), trfull[, 1], CV=TRUE)$class
sum(qda1 != trfull[, 1])/nrow(trfull)

# 10-fold cv
a <- matrix(0, ncol=10, nrow=91)
b <- 1
count <- 1
while(count < 11) {
  set.seed(1)
  a[, count] <- sample((1:917)[-b], 91, FALSE)
  b <- matrix(a[, 1:count], ncol=1)
  count <- count + 1
}

tenfoldqda <- function(d) {
  k <- matrix(a[, -d], ncol=1)
  qda1 <- qda(as.matrix(trfull[k, c]), trfull[k, 1])
  qda2 <- predict(qda1, as.matrix(trfull[a[, d], c]))$class
  error<- sum(qda2 != trfull[a[, d], 1])/nrow(trfull[a[, d], ])
  list(error = error)
}

e <- apply(as.matrix(1:10), 1, tenfoldqda)
errorrates <- numeric(10)
for (i in 1:10) {errorrates[i] <- e[[i]]$error}
sum(errorrates)/10

qdafeatures <- c

#---------------------------------------------------------------------------------------------------
# K nearest neighbours
#---------------------------------------------------------------------------------------------------

plot(
  x=tr[tr[, 1]==0, 2],
  y=tr[tr[, 1]==0, 12],
  xlab="Feature 2",
  ylab="Feature 12",
  main="Features 2 and 12 unscaled",
  xlim=c(-300, 300),
  ylim=c(-300, 300)
)
points(tr[tr[, 1]==1, 2], tr[tr[, 1]==1, 12], col=2)

# rescaling the data
trknn <- trfull
rescale <- function(x) (x - min(x))/(max(x) - min(x)) * 600 - 300
trknn[, -1] <- apply(trfull[, -1], 2, rescale)

plot(
  x=trknn[trknn[, 1]==0, 2],
  y=trknn[trknn[, 1]==0, 12],
  xlab="Feature 2",
  ylab="Feature 12",
  main="Features 2 and 12 scaled",
  xlim=c(-300, 300),
  ylim=c(-300, 300)
)
points(trknn[trknn[, 1]==1, 2], trknn[trknn[, 1]==1, 12], col=2)

set.seed(1)
v <- sample(1:nrow(trknn), 100, FALSE)
trk1 <- data.frame(trknn[-v, ])
trk2 <- data.frame(trknn[v, ])

# knn for wrapper
knnwrap <- function(r) {
  set.seed(1)
  x <- knn(as.matrix(trk1[, r]), as.matrix(trk2[, r]), trk1[, 1], 6)
  J <- sum(x == trk2[, 1])/nrow(trk2)
  return(J)
}

# WRAPPER
range <- matrix(2:29, ncol=1)

top <- 13

a <- apply(range, 1, knnwrap)
feature <- range[which.max(a)]
jscore <- numeric(top)
range <- range[-which.max(a)] #remove the row with lowest J score
newrange <- cbind(rep(feature, length(range)), range)


while(length(feature) < top) {
  a <- apply(newrange, 1, knnwrap)
  feature <- newrange[which.max(a), ]
  jscore[length(feature)] <- max(a)
  range <- range[-which.max(a)]
  newrange <- cbind(t(matrix(rep(feature, length(range)),
                             nrow=length(feature))), range)
}

plot(
  x=2:top,
  y=jscore[-1],
  type="l",
  main="Score against number of features",
  xlab="Number of features",
  ylab="Score"
)

feature <- feature[1:top-1]
feature1 <- feature

jscore2<- numeric(top-2)

while(length(feature) > 9) {
  a <- t(matrix(feature, nrow=length(feature), ncol=length(feature)))
  diag(a) <- NA
  a <- t(a)
  range <- t(matrix(a[!is.na(a)], nrow=length(feature)-1, ncol=length(feature)))
  b <- apply(range, 1, knnwrap)
  feature <- feature[-which.max(b)]
  jscore2[length(feature)] <- max(b)
}

feature2 <- feature
feature2

while(length(feature) > 5) {
  a <- t(matrix(feature, nrow=length(feature), ncol=length(feature)))
  diag(a) <- NA
  a <- t(a)
  range <- t(matrix(a[!is.na(a)], nrow=length(feature)-1, ncol=length(feature)))
  b <- apply(range, 1, knnwrap)
  feature <- feature[-which.max(b)]
  jscore2[length(feature)] <- max(b)
}

feature1

plot(
  x=5:(top-2),
  y=jscore2[-(1:4)],
  type="l",
  main="Score against number of features",
  xlab="Number of features",
  ylab="Score"
)


knncv <- function(k) {
  set.seed(1)
  w <- knn.cv(trknn[, feature2], trknn[, 1], k)
  error <- sum(w != trknn[, 1])/nrow(trknn)
  return(error)
}

k <- matrix(2:50)
error <- apply(k, 1, knncv)
plot(k, error, main="CV errors against k")
which.min(error)
min(error)

#apparent error
set.seed(1)
w <- knn(trknn[, feature2], trknn[, feature2], trknn[, 1], 8)
sum(w!=trknn[, 1])/nrow(trknn)

#hold-out error
set.seed(1)
w <- knn(trk1[, feature2], trk2[, feature2], trk1[, 1], 8)
sum(w!=trk2[, 1])/nrow(trk2)

#cv error
set.seed(1)
w <- knn.cv(trknn[, feature2], trknn[, 1], 8)
sum(w!=trknn[, 1])/nrow(trknn)

#10-fold cv
a <- matrix(0, ncol=10, nrow=91)
b <- 1
count <- 1
while(count < 11) {
  set.seed(1)
  a[,count] <- sample((1:917)[-b], 91, FALSE)
  b <- matrix(a[, 1:count],ncol=1)
  count <- count + 1
}

tenfoldknn <- function(d) {
  j <- matrix(a[,-d], ncol=1)
  set.seed(1)
  w <- knn(trknn[j, feature2], trknn[a[, d], feature2], trknn[j, 1], 8)
  error <- sum(w!=trknn[a[, d], 1])/nrow(trknn[a[, d] , ])
  list(error = error)
}

e <- apply(as.matrix(1:10), 1, tenfoldknn)
errorrates <- numeric(10)
for (i in 1:10) {errorrates[i] <- e[[i]]$error}
sum(errorrates)/10

#---------------------------------------------------------------------------------------------------
# Distance-weighted k nearest neighbours
#---------------------------------------------------------------------------------------------------

train <- trk1[, 2:19]
test <- trk2[, 2:19]
class <- trk1[, 1]
classte <- trk2[, 1]

p <- ncol(train)
ntr <- nrow(train)
nte <- nrow(test)
ngroups <- 2

knn.dist <- function(k) {
  
  k <- k + 1
  
  class1 <- array(0, nte)
  
  for (i in 1:nte) {
    a <- array(0,dim=c(2, p, ntr))
    a[1, , ] <- rep(test[i, ], ntr)
    a <- array(a,dim=c(2, p, ntr))
    a[2, , ] <- t(train)
    
    d1 <- apply(a, 3, function(x) abs(dist(x, method="euclidean")))
    d <- cbind(d1, class)
    o <- order(d[, 1], decreasing=F)
    kmin1 <- d[o, ][(1:k), ]
    
    if (kmin1[k, 1]>0) {
      kmin <- cbind(kmin1[, 1]/kmin1[k, 1],kmin1[, 2])
    }
    
    w <- kmin
    w[, 1] <- apply(t(kmin[, 1]), 1,function(x) (1/x))
    w <- kmin1
    
    b <- array(0, dim=c(ngroups, 1))
    
    b[1] <- sum(w[, 1]*(w[, 2]==0))
    b[2] <- sum(w[, 1]*(w[, 2]==1))
    
    class1[i] <- c(0, 1)[which.max(b)]
  }
  error <- sum(class1 != classte)/nte
  return(error)
  
}

knnall <- function(k) {
  set.seed(1)
  a <- knn(train, test, class, k)
  return(sum(a != classte)/nte)
}
#distance-weighted knn
a <- apply(as.matrix(seq(3, 19, 2)), 1, knn.dist)
#knn
b <-apply(as.matrix(seq(3, 19, 2)), 1, knnall)

#---------------------------------------------------------------------------------------------------
# Multi-layer perceptron
#---------------------------------------------------------------------------------------------------

#filter
filter1 <- function(a) {
  s0 <- var(trfull[trfull[, 1]==0, a])
  s1 <- var(trfull[trfull[, 1]==1, a])
  SW <- s0 + s1
  m0 <- mean(trfull[trfull[, 1]==0, a])
  m1 <- mean(trfull[trfull[, 1]==1, a])
  m <- mean(trfull[,a])
  SB <- nrow(trfull[trfull[, 1]==0, ]) / nrow(trfull) * (m0-m) %*% t(m0-m) + 
    nrow(trfull[trfull[, 1]==1, ])/nrow(trfull) * (m1-m) %*% t(m1-m)
  
  J <- Matrix::solve(SW) %*% SB
  
  return(J)  
}

filter <- function(a) {
  s0 <- mlest(trfull[trfull[, 1]==0, a])$sigmahat
  s1 <- mlest(trfull[trfull[, 1]==1, a])$sigmahat
  SW <- s0 + s1
  m0 <- mlest(trfull[trfull[, 1]==0, a])$muhat
  m1 <- mlest(trfull[trfull[, 1]==1, a])$muhat
  m <- mlest(trfull[,a])$muhat
  SB <- nrow(trfull[trfull[, 1]==0, ])/nrow(trfull) * (m0-m) %*% t(m0-m) + 
    nrow(trfull[trfull[, 1]==1, ])/nrow(trfull) * (m1-m) %*% t(m1-m)
  
  J <- matrix.trace(Matrix::solve(SW) %*% SB)
  
  return(J)
}

range <- matrix(2:28)
a <- apply(range, 1, filter1)
feature <- range[which.max(a)]
jscore <- numeric(28)
range <- range[-which.max(a)] #remove the row with lowest J score
newrange <- cbind(rep(feature, length(range)), range)

while(length(feature) < 13) {
  a <- apply(newrange, 1, filter)
  feature <- newrange[which.max(a), ]
  jscore[length(feature)] <- max(a)
  range <- range[-which.max(a)]
  newrange <- cbind(t(matrix(rep(feature, length(range)), nrow=length(feature))), range)
}

feature


"which.is.min" <-
  function(x)
  {
    y <- seq(along = x)[x == min(x)]
    if(length(y) > 1)
      sample(y, 1)
    else y
  }

trmlp <- trfull
rescale <- function(x) (x - min(x))/(max(x) - min(x))
trmlp[, -1] <- apply(trfull[, -1], 2, rescale)

set.seed(1)
v <- sample(1:nrow(trmlp), 100, FALSE)
trm1 <- data.frame(trmlp[-v, ])
trm2 <- data.frame(trmlp[v, ])

H <- c(6, 9, 12)
wd <- c(0.001, 0.0001, 0.00001)
gr <- as.matrix(expand.grid(wd,H))

# MLP with one hidden layer
"nnet.f" <-
  function(x, ntry)
  {
    set.seed(1)
    H <- x[2]
    decay <- x[1]
    preds <- matrix(0,ncol=ntry,nrow=nrow(trm2))
    score <- numeric(ntry)
    count <- 0
    while(count < ntry) {
      mlp <- nnet(trm1[, 2:19], trm1[, 1], size=H, decay=decay, maxit=10000)
      evs <- eigen(nnetHess(mlp, trm1[, 2:19],trm1[, 1]), TRUE)$values
      #ensures that this is a minima; evalues of Hessian all > 0
      if(min(evs) > 0) {
        count <- count + 1
        preds[, count] <- as.numeric(predict(mlp, trm2[, 2:19]))
        score[count] <- sum((preds[, count] > 0.5) != trm2[, 1])/nrow(trm2)
      }
    }
    avgscore <- sum((rowSums(preds) > 0.5*ntry) != trm2[, 1])/nrow(trm2)
    indx <- which.is.min(score)
    list(score = score[indx], avgscore = avgscore)
  }

a <- apply(gr, 1, nnet.f,ntry=3) #3 different starting points
errorrates <- avgerror <- numeric(9)
for (i in 1:9) {errorrates[i] <- a[[i]]$score
avgerror[i] <- a[[i]]$avgscore}
min(errorrates)
gr[which.min(errorrates), ]
grid <- cbind(gr, errorrates, avgerror)

# holdout error
ntry <- 5
preds <- matrix(0, ncol=ntry, nrow=nrow(trm2))
score <- numeric(ntry)
count <- 0
set.seed(1)
while(count < ntry) {
  mlp <- nnet(trm1[, (2:19)], trm1[, 1], size=6, decay=0.0001, maxit=10000)
  evs <- eigen(nnetHess(mlp, trm1[, (2:19)], trm1[, 1]), TRUE)$values
  # ensures that this is a minima; evalues of Hessian all > 0
  if(min(evs) > 0) {
    count <- count + 1
    preds[, count] <- as.numeric(predict(mlp, trm2[, (2:19)]))
  }
}
ho <- sum((rowSums(preds) > 0.5*ntry) != trm2[, 1])/nrow(trm2)
ho

# 10-fold cv
a <- matrix(0, ncol=10, nrow=91)
b <- 1
count <- 1
while(count < 11) {
  set.seed(1)
  a[,count] <- sample((1:917)[-b], 91, FALSE)
  b <- matrix(a[, 1:count], ncol=1)
  count <- count + 1
}

tenfold <- function(c) {
  set.seed(1)
  preds <- matrix(0, ncol=ntry, nrow=nrow(trmlp))
  score <- numeric(1)
  count <- 0
  while(count < ntry) {
    k <- matrix(a[, -c], ncol=1)
    mlp <- nnet(trmlp[k, 2:19], trmlp[k, 1], size=6, decay=0.0001, maxit=10000)
    evs <- eigen(nnetHess(mlp, trmlp[k, 2:19], trmlp[k, 1]),TRUE)$values
    #ensures that this is a minima; evalues of Hessian all > 0
    if(min(evs) > 0) {
      count <- count + 1
      preds <- as.numeric(predict(mlp, trmlp[a[, c], 2:19]))
      score[count] <- sum((preds > 0.5) != trmlp[a[, c], 1])/nrow(trmlp[a[, c], ])
    }
  }
  indx <- which.is.min(score)
  list(score = score[indx])
}

a <- apply(as.matrix(1:10), 1, tenfold)
errorrates <- numeric(10)
for (i in 1:10) { errorrates[i] <- a[[i]]$score }
tencv <- sum(errorrates)/10
tencv

# mcnemar
te <- te[complete.cases(te), ]

# qda
qda1 <- qda(trfull[, qdafeatures], trfull[, 1])
qdapredict <- predict(qda1, te[, qdafeatures])$class

# rescaling data for mlp the same way we did it for training set
tem <- te
rescale <- function(x, min, max) ((x - min)/(max - min))
for (i in 2:29) {
  tem[, i] <- rescale(te[, i], min(trfull[, i]), max(trfull[, i]))
}

# mlp
ntry <- 5
preds <- matrix(0, ncol=ntry, nrow=nrow(tem))
score <- numeric(ntry)
count <- 0
set.seed(1)
while(count < ntry) {
  mlp <- nnet(trmlp[, (2:19)], trmlp[, 1], size=6,decay=0.0001, maxit=10000)
  evs <- eigen(nnetHess(mlp, trmlp[, (2:19)], trmlp[, 1]), TRUE)$values
  # ensures that this is a minima; evalues of Hessian all > 0
  if(min(evs) > 0) {
    count <- count + 1
    preds[, count] <- predict(mlp, tem[, (2:19)])
  }
}

mlppredict <- as.numeric((rowSums(preds) > 0.5*5))

mcnemar <- function(s) {
  p <- s[1]
  q <- s[2]
  r <- s[3]
  if (p != q & q == r) {
    return(0)
  }
  if (p != q & p == r) {
    return(1)
  }
  if (p != r & p == q) {
    return(2)
  }
  if (p == q & p == r) {
    return(3)
  }
}

qdapredict <- as.numeric(qdapredict)
if(min(qdapredict) == 1) {
  qdapredict <- qdapredict -1
}
s <- cbind(te[, 1], qdapredict, mlppredict)
h <- apply(s, 1, mcnemar)
n00 <- sum(h==0)
n01 <- sum(h==1)
n10 <- sum(h==2)
n11 <- sum(h==3)

z <- (abs(n01 - n10) - 1)/(sqrt(n10 + n01))
z

