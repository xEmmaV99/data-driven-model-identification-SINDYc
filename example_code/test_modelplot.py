import matplotlib.pyplot as plt

from source import *

model = load_model("currents_model.pkl")
plot_coefs2(model, log = True)
plt.show()