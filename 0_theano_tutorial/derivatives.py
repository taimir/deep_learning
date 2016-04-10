import numpy
import theano
import theano.tensor as T
from theano import pp

"""
Simple example
"""
x = T.dscalar('x')
y = x ** 2
# gradient of y with respect to x
gy = T.grad(y, x)
pp(gy)
# dy\dx, evaluated at x. Theano simplifies the above pp(gy) to only one
# APPLY node with multiplication by 2
f = theano.function([x], gy)
print(f(4)) # 8
print(numpy.allclose(f(94.2), 188.4)) # True

"""
Jacobian matrix
"""
# the first way is to directly use:
# theano.gradient.jacobian()

# second way is to do it manually
x1 = T.dvector('x')
y1 = x1 ** 2
# Here we use a trick: we create a range for i, from 0 to size of y, to iterate across
# reason is we cannot simply use y as sequences directly, because then the single extracted
# elements won't be part of our function graph, i.e. won't be functions of x
J, updates = theano.scan(lambda i, y1, x1: T.grad(y1[i], x1), sequences=T.arange(y1.shape[0]), non_sequences=[y1,x1])
f = theano.function([x1], J, updates = updates)
print(f([4, 4])) # [[8, 0], [0, 8]]

"""
Hessian matrix
"""
#Absolutely analog to the jacobian, but this time using gradient of y instead of y

# In some cases, we can skip the computation of the jacobian on its own
"""
Jacobian times a vector: optimization
"""
# Two operations are supported: multiply vector from right or from left

# R-OP
W = T.dmatrix('W')
V = T.dmatrix('V')
x2 = T.dvector('x')
y2 = T.dot(x2, W)
# here we use the R-OP to speed up the computation
JV = T.Rop(y2, W, V)
f = theano.function([W, V, x2], JV)
print(f([[1, 1], [1, 1]], [[2, 2], [2, 2]], [0,1]))

# L-OP
W1 = T.dmatrix('W')
v = T.dvector('v')
x3 = T.dvector('x')
y3 = T.dot(x3, W1)
VJ = T.Lop(y3, W1, v)
f = theano.function([v, x3], VJ)
print(f([2,2], [0,1]))
"""
Hessian times a vector: optimization
"""
# Absolutely analog., we use the gradient of y instead of y again
