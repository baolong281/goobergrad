import math
from goobergrad.nn import Value

def RMSE(ypred, ytrue):

    return sum((yp-yt)**2 for yp, yt in zip(ypred, ytrue))

def CrossEntropy(ypred, ytrue):

	result = Value(-sum((yt * math.log10(yp.data)) for yp, yt in zip(ypred, ytrue)), (ypred), _op='CrossEntropy')
	ln10 = 2.30258509299

	def _backward():
		for node in ypred:
			node.grad += -(-1 / (node.data * ln10)) * result.grad

	result._backward = _backward
            
	return result
		


    