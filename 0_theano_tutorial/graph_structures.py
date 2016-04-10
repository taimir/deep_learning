"""
Understanding how theano works
"""

"""
OPS:

Theano GRAPHS: connected "VARIABLE", "OP", "APPLY", "CONSTANT" and "TYPE" nodes

OP: some expression of placeholders (TheanoVariables), operators that produces
certain outputs, it is the DEFINITION of the expression. It is always referenced
by an APPLY node that actually applies (calls) it.

VARIABLE: a theano variable (placeholder). If it results from a computation, its
"owner" field points to the APPLY node that is used to produce it. That way you can
traverse the graph. VARIABLE nodes are always interconnected through APPLY nodes,
building the computation graph. The "type" field points to their TYPE node,
the "index" field is the index for this variable at .owner.outputs[]. The "name"
field is the name this node can be identified by.

APPLY: the actual APPLICATION of OP on the VARIABLES that are specified as "inputs"
of the APPLY node. Apply nodes also point to their results (output, returned value)
through its "outputs" field. The "op" field points to the OP node that's used.
The result VARIABLE points back to the APPLY node that produces it through its
owner field. An APPLY node is the actual function call.

TYPE: defines the type of a variable, through the constraints that are applicable
to this type. Used by Theano to actually generate C code.

CONSTANT: subclass of VARIABLE, but the data in it is to be defined just once.

see drawing at: http://deeplearning.net/software/theano/extending/graphstructures.html
"""

import theano
x = theano.tensor.dmatrix('x')
y = x * 2

print(type(y.owner)) # owner is an apply node that produces y
print(y.owner.op.name) # elementwise multiplication
print(len(y.owner.inputs)) # the apply node has 2 inputs
print(y.owner.inputs[0]) # x
print(y.owner.inputs[1]) # a broadcasted 2, using DimShuffle on X to form its shape
print(type(y.owner.inputs[1].owner)) # note: the input 2 actually has an apply node
print(y.owner.inputs[1].owner.inputs) # the broadcasted 2 is formed through an apply
                                     # which is a DimShuffle with input the constant 2.0

"""
Automatic differentiation

Theano just traverses the graph back from the root, following the "owner" fields. Each
apply node points to an OP node, which is used to compute the derivative of this particular
part. Then the "chain rule" is applied to compose all of the nested APPLY nodes's functions.
"""

"""
Optimization

Theano automatically simplifies the graph for a function (see lat picture here:
http://deeplearning.net/software/theano/extending/graphstructures.html)
"""
