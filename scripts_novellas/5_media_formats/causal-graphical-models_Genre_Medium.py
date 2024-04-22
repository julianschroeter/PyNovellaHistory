from causalgraphicalmodels import CausalGraphicalModel
import daft
import matplotlib.pyplot as plt



system = "my_xps"

# The following code is copied from the code for Richard McElreath’s text book Statistical Rethinking. The code has been provided by: https://github.com/aloctavodia/Statistical-Rethinking-with-Python-and-PyMC3


# Note - There is no explicit code section for drawing the second DAG
# but the figure appears in the book and hence drawing it as well
fig, ax = plt.subplots()
dag5_2 = CausalGraphicalModel(nodes=["G", "M", "TM", "Z"], edges=[("G", "TM"), ("M", "TM"), ("Z", "TM")])
pgm = daft.PGM()
pgm.add_text(0, 0.5, "Unabhängige Faktoren", fontsize=10)
pgm.add_text(0, 0, "", fontsize=6)
coordinates = {"G": (0, 1), "TM": (1, 2), "M": (1, 1), "Z": (2,1)}
for node in dag5_2.dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag5_2.dag.edges:
    pgm.add_edge(*edge,  ax=ax)
pgm.render()
#ax = pgm._rendering_context.ax()
#plt.title("Gattung, Medium und Zeit als unabhängige Kausalfaktoren")
plt.gca().invert_yaxis()

#ax.set_title("Gattung, Medium und Zeit als unabhängige Kausalfaktoren")
plt.tight_layout()
plt.savefig("/home/julian/Documents/CLS_temp/figures/DAG-genre-medium-time-indep-causal_model1.svg")
plt.show()



# Note - There is no explicit code section for drawing the second DAG
# but the figure appears in the book and hence drawing it as well
dag5_2 = CausalGraphicalModel(nodes=["G", "M", "TM"], edges=[("G", "TM"), ("M", "TM")])
pgm = daft.PGM()
coordinates = {"G": (0, 0), "TM": (1, 1), "M": (2, 0)}
for node in dag5_2.dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag5_2.dag.edges:
    pgm.add_edge(*edge)
pgm.render()
plt.gca().invert_yaxis()
plt.title("Gattung und Medium als unabhängige Kausalfaktoren")
plt.show()


# In[33]:


dag5_2 = CausalGraphicalModel(nodes=["G", "M", "TM"], edges=[("M", "G"), ("G", "TM"), ("M", "TM")])
pgm = daft.PGM()
pgm.add_text(0, 0.5, "  Gattung als medien-\n   abhängiger Faktor", fontsize=10)
pgm.add_text(0, 0, "", fontsize=6)
coordinates = {"G": (0, 1), "TM": (1, 2), "M": (2, 1)}
for node in dag5_2.dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag5_2.dag.edges:
    pgm.add_edge(*edge)
pgm.render()
plt.gca().invert_yaxis()
plt.title("Gattung als medienabhängiger Kausalfaktor")
plt.savefig("/home/julian/Documents/CLS_temp/figures/DAG-model2.svg")
plt.show()


# In[66]:


dag5_2 = CausalGraphicalModel(nodes=["M", "TM", "G"], edges=[("M", "TM"), ("TM", "G")])
pgm = daft.PGM()
pgm.add_text(0, 0.5, "  Medium als Faktor \nmit Gattung als Effekt", fontsize=10)
pgm.add_text(0, 0, "", fontsize=6)
coordinates = {"M": (0, 1), "TM": (1, 1), "G": (2, 1)}
for node in dag5_2.dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag5_2.dag.edges:
    pgm.add_edge(*edge)
pgm.render()

plt.gca().invert_yaxis()
#plt.title("Medium als textprägender Kausalfaktor, mit Gattung als Epiphänomen")
plt.savefig("/home/julian/Documents/CLS_temp/figures/DAG-Model3.svg")
plt.show()


# but the figure appears in the book and hence drawing it as well
dag5_2 = CausalGraphicalModel(nodes=["G", "M", "TM"], edges=[("M", "TM")])

pgm = daft.PGM()
coordinates = {"G": (0, 0), "TM": (1, 1), "M": (2, 0)}
for node in dag5_2.dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag5_2.dag.edges:
    pgm.add_edge(*edge)
pgm.render()
plt.gca().invert_yaxis()
plt.title("Medium als textprägender Kausalfaktor")
plt.savefig("/home/julian/Documents/CLS_temp/figures/DAG-Medium-causal.svg")
plt.show()



# the old view: only genre causes textual features
dag = CausalGraphicalModel(nodes=["G", "M", "TM"], edges=[("G", "TM")])
pgm = daft.PGM()
coordinates = {"G": (0, 0), "TM": (1, 1), "M": (2, 0)}
for node in dag.dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag.dag.edges:
    pgm.add_edge(*edge)
pgm.render()
plt.gca().invert_yaxis()
plt.title("Gattung als textprägender Kausalfaktor")
plt.show()


# but the figure appears in the book and hence drawing it as well
dag5_2 = CausalGraphicalModel(nodes=["G", "M", "TM"], edges=[("M", "TM")])

pgm = daft.PGM()
coordinates = {"G": (0, 0), "TM": (1, 1), "M": (2, 0)}
for node in dag5_2.dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag5_2.dag.edges:
    pgm.add_edge(*edge)
pgm.render()
plt.gca().invert_yaxis()
plt.title("Medium als textprägender Kausalfaktor")
plt.savefig("/home/julian/Documents/CLS_temp/figures/DAG-medium-but-no-genre_model1.svg")
plt.show()


fig, ax = plt.subplots()
pgm = daft.PGM()
dag = CausalGraphicalModel(nodes=["G", "M", "TM"], edges=[("G", "TM")])
pgm.add_text(0, 0, "", fontsize=6)
pgm.add_text(0, 0.5, "Monofaktorielles Modell", fontsize=10)


coordinates = {"G": (0, 1), "TM": (1, 2), "M": (2, 1)}
for node in dag.dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag.dag.edges:
    pgm.add_edge(*edge,  ax=ax)

pgm.render()
plt.gca().invert_yaxis()
#plt.title("Gattung als textprägender Kausalfaktor")
plt.tight_layout()
plt.savefig("/home/julian/Documents/CLS_temp/figures/DAG-genre-but-no-medium_model1.svg")
plt.show()




# but the figure appears in the book and hence drawing it as well
dag5_2 = CausalGraphicalModel(nodes=["G", "M", "TM"], edges=[("M", "TM")])

pgm = daft.PGM()
coordinates = {"G": (0, 0), "TM": (1, 1), "M": (2, 0)}
for node in dag5_2.dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag5_2.dag.edges:
    pgm.add_edge(*edge)
pgm.render()
plt.gca().invert_yaxis()
plt.title("Medium als textprägender Kausalfaktor")
plt.savefig("/home/julian/Documents/CLS_temp/figures/DAG-Monofaktoriell_medium-but-no-genre_model1b.svg")
plt.show()

fig, ax = plt.subplots()
pgm = daft.PGM()
dag = CausalGraphicalModel(nodes=["G", "M", "TM"], edges=[("M", "TM")])
pgm.add_text(0, 0, "", fontsize=6)
pgm.add_text(0, 0.5, "Monofaktorielles Modell", fontsize=10)


coordinates = {"G": (0, 1), "TM": (1, 2), "M": (2, 1)}
for node in dag.dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in dag.dag.edges:
    pgm.add_edge(*edge,  ax=ax)

pgm.render()
plt.gca().invert_yaxis()
#plt.title("Gattung als textprägender Kausalfaktor")
plt.tight_layout()
plt.savefig("/home/julian/Documents/CLS_temp/figures/Monofaktorielles_Modell_1b_Medium.svg")
plt.show()
