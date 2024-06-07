
# Data:
nobs = 2000
X, y = make_circles(nobs, noise = 0.1, factor = 0.5)
Xmat = Float32.(permutedims(matrix(X)))
X = table(permutedims(Xmat))
batch_size = Int(round(nobs / 10))
