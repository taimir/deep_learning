import numpy
import theano.tensor as T
from theano import function
from theano import pp

"""
Introduction to expressions with tensors
"""
# create two double-scalar variables (symbols) for the formula
# NOTE: the specified name is not required, but is useful during debugging
# NOTE: here, x and y are of type TensorVariable, but have a type field set to
# dscalar
x = T.dscalar('x')
y = T.dscalar('y')

# z is another TensorVariable. It's type field is float64 (double), because
# it is a sum of two TensorVariables with type field double
z = x + y
print(pp(z))
f = function([x, y], z)

# call the function
print(f(2,3))
# NOTE: another option is using z.eval({x : 2.3, y : 1.3}), but it is less
# expressive
print(numpy.allclose(f(16.3, 12.1), 28.4))

"""
Adding matrices
"""
X = T.dmatrix('X')
Y = T.dmatrix('Y')
Z = X + Y
f_mat = function([X, Y], Z)
# NOTE: here we can use both python arrays and numpy arrays
print(f_mat([[1, 2], [3, 4]], [[10, 20], [30, 40]]))

"""
The tensorflow tensor types are (regex):
(b|w|i|l|f|d|c)(scalar|vector|matrix|row|col|tensor3|tensor4)

where:
b = byte
w = word
i = int
l = long
f = float
d = double
c = complex
"""

"""
Exercise: solution
"""
a = T.vector()
b = T.vector()
out = a ** 2 + b ** 2 + 2 * a * b
f_vec = function([a, b], out)
print(f_vec([0, 1, 2], [3, 4, 5]))
