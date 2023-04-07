import math
from goobergrad.nn import Value

def RMSE(ypred, ytrue):
    return sum((yp-yt)**2 for yp, yt in zip(ypred, ytrue))

def CrossEntropy(ypred, ytrue):

	x = -sum((yt * math.log(yp.data) for yp, yt in zip(ypred, ytrue)))
	result = Value(x, (ypred), _op='CrossEntropy')

	def _backward():
		for node in ypred:
			node.grad += -(1 / node.data) * result.grad
	result._backward = _backward
            
	return result
		


    