import math
import random

class Value:
	
	def __init__ (self, data, _children=(), _op="", label=""):
		self._prev = set(_children)
		self.grad = 0.0
		self._backward = lambda: None
		self.data = data
		self._op = _op
		self.label=label
	
	def __repr__(self):
		return f"Value({self.data})"

	def __ge__(self, other):
		return self.data >= other.data
	
	def __gt__(self, other):
		return self.data > other.data

	def __rmul__(self, other):
		return self * other
	
	def __radd__(self, other):
		return self + other
	
	def __rsub__(self, other):
		return self-other
	
	def __add__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		result = Value(self.data + other.data, (self, other), _op="+")
		
		def _backward():
			self.grad += 1.0 * result.grad
			other.grad += 1.0 * result.grad
		result._backward = _backward
		
		return result
	
	def __mul__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		result = Value(self.data * other.data, (self, other), _op="*")

		def _backward():
			self.grad += other.data * result.grad
			other.grad += self.data * result.grad
			
		result._backward = _backward
			
		return result
	
	def __truediv__(self, other):
		return self * other**-1
	
	def __neg__(self):
		return self * -1
	
	def __sub__(self, other):
		return self + (-other)
	
	def __pow__(self, other):
		assert isinstance(other, (int, float))
		result = Value(self.data**other, (self, ), f'**{other}')
					   
		def _backward():
			self.grad += (other * (self.data**(other-1))) * result.grad
						
		result._backward = _backward
		return result

	def log(self):
		result = Value(math.log10(self.data), (self, ), 'log')
		ln10 = 2.30258509299

		def _backward():
				self.grad += (1 / (self.data * ln10)) * result.grad
		result._backward = _backward

		return result 
	
	def tanh(self):
		x = self.data
		t = (math.exp(2*x)-1) / (math.exp(2*x)+1)
		result = Value(t, (self, ), 'tanh')
		
		def _backward():
			self.grad += (1-t**2) * result.grad
		result._backward = _backward
			
		return result
	
	def relu(self):
		
		x = max(0, self.data)
		result = Value(x, (self, ), 'relu')
		
		def _backward():
			self.grad += (result.grad > 0) * result.grad
		result._backward = _backward
		
		return result

	def leaky_relu(self):

		c = .01
		x = c * self.data if self. data < 0 else self.data
		result = Value(x, (self, ), 'leaky_relu')

		def _backward():
			self.grad += 1 * result.grad if self.data>0 else c * result.grad
		result._backward = _backward

		return result
	
	def exp(self):
		x = self.data
		result = Value(math.exp(x), (self, ), 'exp')
		
		def _backward():
			self.grad += result.data * result.grad
		result._backward = _backward
						 
		return result
	
	def _softmax(self, denom):



		result = Value(math.exp(self.data) / denom, (self, ), 'softmax')
		
		def _backward():
			self.grad += (result.data * (1-result.data)) * result.grad
		result._backward = _backward
		
		return result
	
	def backward(self, clipping):
		
		topo = []
		visited = set()
		def build_topo(v):
			if v not in visited:
				visited.add(v)
				for child in v._prev:
					build_topo(child)
				topo.append(v)
				
		build_topo(self)
				
		self.grad = 1.0
		for node in reversed(topo):
			node._backward()
			if node.grad > clipping:
				node.grad = clipping
			elif node.grad < -clipping:
				node.grad = -clipping
			
class Neuron:
	
	def __init__(self, nin, _activation ): #num inputs
		std = math.sqrt(2/nin)
		self.w = [Value(random.uniform(-1, 1) * std) for _ in range(nin)]
		self.b = Value(math.sqrt(2 / nin) * random.uniform(-1, 1))
		self._activation = _activation
		
	def __call__(self, x):
		act = sum((wi * wx for wi, wx in zip(self.w, x)), self.b)
		return act.leaky_relu() if self._activation else act 
	
	def parameters(self):
		return [self.b] + self.w
	
class Layer:
	
	def __init__(self, nin, nout, _activation):
		self._activation = _activation 
		self.neurons = [Neuron(nin, True) if self._activation else Neuron(nin, False) for _ in range(nout)]
	
	def __call__(self, x):
		outputs = [n(x) for n in self.neurons]
		return outputs[0] if len(outputs) ==1 else outputs
	
	def parameters(self):
		return [p for neurons in self.neurons for p in neurons.parameters()]

	
class MLP:
	
	def __init__(self, _layers):
		assert all(isinstance(item, int) for item in _layers)
		self._nin = _layers[0]
		self.layers = [Layer(_layers[i], _layers[i+1], True) if i+1<len(_layers)-1 else Layer(_layers[i], _layers[i+1], False ) for i in range(len(_layers)-1)]
	
	def __call__(self, x):
		assert len(x) == self._nin
		output = x
		
		for layer in self.layers:
			output = layer(output)
		
		return output
	
	def parameters(self):
		
		return [p for layer in self.layers for p in layer.parameters()]
	
	def zero_grad(self):
		for p in self.parameters():
			p.grad = 0
			
	def step(self, learning_rate, minimize): # minimize boolean 
		
		for p in self.parameters():

			"""
			if p.grad >= self.grad_max:
				p.grad = self.grad_max
			elif p.grad <= -self.grad_max:
				p.grad = -self.grad_max
			"""

			p.data += p.grad * learning_rate if minimize else -p.grad * learning_rate
			
def softmax(values):
	
	denom = sum((math.exp(n.data)) for n in values)
	return [n._softmax(denom) for n in values]