import numpy as np
import theano
import theano.tensor as T
X = T.dmatrix('X')
# NOTE: this function will be applied component-wise to the matrix, because
# all operators in it are elementwise
s = 1 / (1 + T.exp(-X))
logistic = theano.function([X], s)
# call the function
print(logistic([[0, 1], [2, 3]]))
# now define the same through tanh and assert they are equal
# NOTE: we can actually reuse the variable X
s2 = (1 + T.tanh(X / 2)) / 2
logistic2 = theano.function([X], s2)
print(np.allclose(logistic([[0, 1], [2, 3]]), logistic2([[0, 1], [2, 3]]))) # True

"""
Computing multiple things
"""
# shorthand definition of vars
a, b = T.dmatrices('a', 'b')
diff = a - b
# NOTE: we do not need to use a, b again - diff is already a formed expression
abs_diff = abs(diff)
diff_squared = diff ** 2
f = theano.function([a, b], [diff, abs_diff, diff_squared])
print(f([[1, 1], [1, 1],], [[0, 1], [2, 3]])) # has 3 outputs

"""
Default values
"""
from theano import In
x, y = T.dscalars('x', 'y')
z = x + y
# NOTE: you can do even more complex stuff with In(), not only default values
# NOTE: default values always come AFTER the other inputs with no defaults
f = theano.function([x, In(y, value=1)], z)
print(f(33)) # 34
print(f(33, 2)) # 35

w = T.dscalar('w')
z2 = (x + y)*w
f2 = theano.function([x, In(y, value=1), In(w, value=2, name='w_by_name')], z2)
print(f2(33)) # 68
print(f2(33, 2)) # 70
print(f2(33, 0, 1)) # 33
# NOTE: the above w_by_name allows us to specify w's value at another position
# than that of the original w (just like in regular languages). w_by_name overrides
# the original name
print(f2(33, w_by_name = 1)) # 34
print(f2(33, w_by_name = 1, y = 0)) # 33

"""
Using shared variables
"""
# in order to achieve a system with an internal state that changes with each call
from theano import shared
state = shared(0)
inc = T.iscalar('inc')
# now we can call the function with any argument of type int, and it will
# increate state by that argument on each call (because of updates)
accumulator = function([inc], state, updates=[(state, state+inc)])
print(accumulator(20)) # 0
print(accumulator(3)) # 20
print(accumulator(0)) # 23

# NOTE: shared values can by updated by multiple functions
# NOTE: trying to reset state by state = shared(0) in between function calls won't work
# because it is now a different object, not the one used by the functions.
# Instead, use the .get_value() and .set_value() methods to update the current state.
print(state.get_value()) # 23
state.set_value(12)
print(accumulator(0)) # 12

# Now let's define another function to use state
decrementor = function([inc], state, updates[(state, state - inc)])
print(decrementor(2)) # 10

"""
Replacing parts in the expression graph
"""
# givens replaces a particular named node in the graph with any expression
fn_of_state = state * 2 + inc
# since we will replace the state, we will need a var of type state
foo = T.scalar(dtype = state.dtype)
# NOTE: here foo is an input, which we "plug" into the formula using state
# NOTE: we do not update state of course, as we "don't use it" at all
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
print(skip_shared(1, 3)) # 7

"""
Reusing function definitions with different states
"""
state.set_value(10)
# create an accumulator with another state by copying the previous one
new_state = theano.shared(0)
# NOTE: the optimized tree does not need to be recompiled
new_accumulator = accumulator.copy(swap={state: new_state})
print(new_accumulator(100)) # 0 and not 10
print(state.get_value()) # 10
print(new_state.get_value()) # 100

"""
Using random numbers
"""
# since in Theano, we always define expressions (using variables, i.e. placeholders)
# it is only logical to think of any random number in our formulas as random
# variables
# NOTE: we call the generator (sampler) for the random variable a random stream,
# and in a sense, it's state is a shared state across functions that call it

from theano.tensor.shared_randomstreams import RandomStreams as RS
srng = RS(seed=234)
# 2x2 matrix of uniformly distr. RVs
rv_unif = srng.uniform((2,2))
# 2x2 matrix of normally distr. RVs
rv_normal = srng.normal((2,2))

f = function([], rv_unif)
# NOTE: here the rng is not updated between function calls !
g = function([], rv_normal, no_default_updates=True)

print(f()) # different uniform numbers
print(f()) # different uniform numbers
print(g()) # different normal numbers
print(g()) # same normal numbers as the prev. call

# NOTE: a single RV is sampled only once in one function call, regardless of how many
# times it appears in the formula (which makes sense, in math it is the same)
nearly_zeros = function([], rv_unif + rv_unif - 2*rv_unif)
print(nearly_zeros()) # returns 0

# Using seeds: you can seed each RV separately or all at once (pretty much to the same effect)
rng_val = rv_unif.rng.get_value(borrow=True)
rng_val.seed(81232)
rv_unif.rng.set_value(rng_val, borrow=True)

# or all at once
srng.seed(123321)

# and to explicitly show that RandomStreams have a shared state:
state_after_v0 = rv_unif.rng.get_value().get_state()
nearly_zeros()
v1 = f()
# Go one step back
rng = rv_unif.rng.get_value(borrow=True)
rng.set_state(state_after_v0)
rv_unif.rng.set_value(rng, borrow=True)
print(v1 == f()) # False
print(v1 == f()) # True

"""
Copying random states from one function to another
"""
from theano.sandbox.rng_mrg import MRG_RandomStreams as MRG_RS
class Graph():
    def __init__(self, seed=123):
        self.rng = RS(seed)
        self.y = self.rng.uniform(size=(1,))

# define two functions with 2 different rng's
g1 = Graph(seed=123)
f1 = theano.function([], g1.y)

g2 = Graph(seed=987)
f2 = theano.function([], g2.y)

# by default functions are out of sync
print(f1())
print(f2())

# now let's copy the random states
def copy_random_state(g1, g2):
    if isinstance(g1.rng, MRG_RS):
        g2.rng.rstate = g1.rng.rstate
    # copy all the updates from g1 to g2
    for (su1, su2) in zip(g1.rng.state_updates, g2.rng.state_updates):
        su2[0].set_value(su1[0].get_value())

copy_random_state(g1, g2)
print(f1() == f2()) # true
