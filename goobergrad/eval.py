def RMSE(ypred, ytrue):
    return sum((yp-yt)**2 for yp, yt in zip(ypred, ytrue))