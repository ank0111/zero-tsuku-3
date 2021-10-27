import numpy as np
from dezero import Variable
from dezero import as_array
import dezero.functions as F
import matplotlib.pyplot as plt
from dezero.models import MLP
from dezero import optimizers


import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 800
hidden_size = (10, 1)

model = MLP(hidden_size)
optimizer = optimizers.MomentumSGD(lr).setup(model)
# Plot

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    if i % 1000 == 0:
        print(loss)


# Plot
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = model(t)
plt.plot(t, y_pred.data, color='r')
plt.show()
