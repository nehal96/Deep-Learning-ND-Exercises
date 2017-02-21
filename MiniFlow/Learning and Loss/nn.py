"""
NOTE: Here we're using an Input node for more than a scalar.
In the case of weights and inputs the value of the Input node is
actually a python list!

In general, there's no restriction on the values that can be passed to an Input node.
"""
from miniflow import *

inputs, weights, bias = Input(), Input(), Input()

f = Linear(inputs, weights, bias)

feed_dict = {
    inputs: [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

print(output) # should be 12.7 with this example
