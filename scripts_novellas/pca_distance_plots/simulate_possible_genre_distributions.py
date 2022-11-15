import numpy as np
import matplotlib.pyplot as plt

period1_N_feat1 = np.random.normal(loc= -5, scale=2.0, size= 100)
period1_N_feat2 = np.random.normal(loc= -3, scale=2.0, size= 100)

period1_E_feat1 = np.random.normal(loc= 5, scale=2.0, size= 100)
period1_E_feat2 = np.random.normal(loc= 3, scale=2.0, size= 100)

period2_E_feat1 = np.random.normal(loc= -5, scale=2.0, size= 100)
period2_E_feat2 = np.random.normal(loc= -3, scale=2.0, size= 100)

period2_N_feat1 = np.random.normal(loc= 5, scale=2.0, size= 100)
period2_N_feat2 = np.random.normal(loc= 3, scale=2.0, size= 100)


#plt.scatter(period1_N_feat1, period1_N_feat2, color="green", label = "Novellen before 1850")
plt.scatter(period2_N_feat1, period2_N_feat2, color="orange", label = "Novellen after 1850")
#plt.scatter(period1_E_feat1, period1_E_feat2, color="blue", label = "Erzählungen before 1850")
plt.scatter(period2_E_feat1, period2_E_feat2, color="red", label = "Erzählungen after 1850")

plt.title("Simulation: Change of rules of Label Use with unchanged writing style")
plt.legend()


plt.show()