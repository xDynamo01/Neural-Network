'''solving for x
e ** x = b
'''
import numpy as np
import math

#b = 5.2

#print(np.log(b))  # natural log, base e
#print(math.e ** 1.6486586255873816)

softmax_outputs = [0.7, 0.1, 0.2]
targer_outputs = [1, 0, 0]

loss = -(math.log(softmax_outputs[0])*targer_outputs[0] +
         math.log(softmax_outputs[1])*targer_outputs[1] +
         math.log(softmax_outputs[2])*targer_outputs[2])

print(loss)
loss = -math.log(softmax_outputs[0])
print(loss)

print(math.log(0.7))
print(-math.log(0.5))
