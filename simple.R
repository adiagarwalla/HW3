library(data.table)
trips = read.table("~/Desktop/COS/COS424/P3/HW3/txTripletsCounts.txt", nrows=3348026)

#options(max.print=100000000) 
#print(trips)

DT.trips = data.table(trips[seq(1,length(trips), 3)], trips[seq(2,length(trips))])
setnames(DT.trips, c("giver", "receiver", "N"))
setkey(DT.trips, "giver", "receiver")

#print(DT.trips)

DT.binary = DT.trips[,1, by=c("giver", "receiver")]
setnames(DT.binary, c("giver", "receiver", "indicator"))
setkey(DT.binary, "giver", "receiver")

#print(DT.binary)

UNIQUE = sort(unique(c(DT.binary[,giver], DT.binary[,receiver])))
NUM.ADDR = length(UNIQUE)
library(Matrix)
MAT.bin = spMatrix(NUM.ADDR, NUM.ADDR, 
  i = DT.binary[,giver]+1,
  j = DT.binary[,receiver]+1,
  x = DT.binary[,indicator])

test.data = read.table("~/Desktop/COS/COS424/P3/HW3/testTriplets.txt")
test.data[,1] = test.data[,1] + 1
test.data[,2] = test.data[,2] + 1
test.data = data.table(test.data)
setnames(test.data, c("giver", "receiver", "bool"))
#print(test.data)

set.seed(1234)
library(irlba)
#svd.out = irlba(MAT.bin, 25, 25, tol=1e-10)
svd.out = irlba(MAT.bin, 11, 11, tol=1e-10)
plot(svd.out$d)

u = svd.out$u
print(dim(u))
print(u[1,])
print(u[2,])
print(u[3,])


v = svd.out$v
print(dim(v))
print(v[1,])
print(v[2,])
print(v[3,])
print(v[444075,])
print(v[17,])

d = svd.out$d
print(dim(d))

prob = rep(0, nrow(test.data))
for (i in 1:nrow(test.data)) {
  p = sum(u[test.data[i,giver],]*d*v[test.data[i,receiver],])
  if (i < 10) {
    #print(v[test.data[i,receiver]])
    print("**")
    #print(test.data[i,giver])
    print(u[test.data[i,giver],])
    print("**")
    print(d)
    print("**")
    print(u[test.data[i,giver],]*d)
    #print("**")
    #print(u[test.data[i,giver],]*d*v[test.data[i,receiver]])
    print("**")
    #print(test.data[i,receiver])
    #print(v[test.data[i,receiver],])
    #print(i)
    print(u[test.data[i,giver],]*d*v[test.data[i,receiver],])
    print(p)
    print("------")
  }
  if (p < 0) p = 0
  if (p > 1) p = 1
  prob[i] = p
}

library(ggplot2)
dat = data.frame(prob, as.factor(test.data[,bool]))
colnames(dat) = c("prob", "value")
ggplot(dat, aes(value, prob)) + geom_violin() + geom_jitter(alpha=0.1) + 
  scale_y_log10() + labs(x="test value", y="prob")
