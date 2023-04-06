import math
from goobergrad.nn import Value

def RMSE(ypred, ytrue):

    return sum((yp-yt)**2 for yp, yt in zip(ypred, ytrue))

def CrossEntropy(ypred, ytrue):

	result = Value(-sum((yt * math.log(yp.data)) for yp, yt in zip(ypred, ytrue)), (ypred), _op='CrossEntropy')

	def _backward():
		for node, label in zip(ypred, ytrue):
			node.grad += -(label / node.data) * result.grad

	result._backward = _backward
            
	return result
		


    