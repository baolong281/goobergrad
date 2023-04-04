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
            self.grad += result.grad if x > 0 else 0
        result._backward = _backward
        
        return result
    
    def exp(self):
        x = self.data
        result = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += result.data * result.grad
        result._backward = _backward
                         
        return result
    
    def backward(self):
        
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
class Neuron:
    
    def __init__(self, nin): #num inputs
        self.w = [Value(random.uniform(-1, 1), label='w') for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1), label='b')
        
    def __call__(self, x):
        act = sum((wi * wx for wi, wx in zip(self.w, x)), self.b)
        result = act.tanh()
        return result
    
    def parameters(self):
        return [self.b] + self.w
    
class Layer:
    
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs[0] if len(outputs) ==1 else outputs
    
    def parameters(self):
        return [p for neurons in self.neurons for p in neurons.parameters()]
    
class MLP:
    
    def __init__(self, _layers):
        assert all(isinstance(item, int) for item in _layers)
        self.layers = [Layer(_layers[i], _layers[i+1]) for i in range(len(_layers)-1)]
    
    def __call__(self, x):
        assert len(x) == len(self.layers)
        output = x
        
        for layer in self.layers:
            output = layer(output)
        
        return output
    
    def parameters(self):
        
        return [p for layer in self.layers for p in layer.parameters()]
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
            
    def step(self, learning_rate):
        
        for p in self.parameters():
            p.data += -p.grad * learning_rate
            