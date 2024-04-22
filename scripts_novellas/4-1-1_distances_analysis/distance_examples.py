import matplotlib.pyplot as plt
import numpy as np
from preprocessing.presetting import local_temp_directory
import os

system = "my_xps"

# Define vertices of the triangle
triangle_vertices = np.array([[0, 10], [15, 12], [12, 20]])

# Define points for the blue line
blue_line_points = np.array([[30, 0], [35, 7]])

# Calculate midpoint between each vertex and corresponding point on the blue line
midpoints = [(v + p) / 2 for v, p in zip(triangle_vertices, blue_line_points)]
print(triangle_vertices[:, 0])
# Plot triangle with labels for endpoints
plt.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], 'r-')
plt.plot([12, 0], [20, 10], 'r-')
for i, (x, y) in enumerate(triangle_vertices):
    plt.text(x, y, f'Roman-{i+1}', ha='right')

# Plot blue line with labels for endpoints
plt.plot(blue_line_points[:, 0], blue_line_points[:, 1], 'b-')
for i, (x, y) in enumerate(blue_line_points):
    plt.text(x, y, f'Märchen-{i+1}', ha='right')

# Plot green dotted lines connecting vertices to points on blue line
print(list(zip(triangle_vertices, midpoints)))

from itertools import combinations

# import itertools package
import itertools
from itertools import permutations

# initialize lists
list_1 = list(triangle_vertices)
list_2 = list(blue_line_points)

unique_combinations = []

for i in range(len(list_1)):
    for j in range(len(list_2)):
        unique_combinations.append((list_1[i], list_2[j]))

print(unique_combinations)


for v, m in unique_combinations:
    plt.plot([v[0], m[0]], [v[1], m[1]], 'g--')

plt.xlabel('Häufigkeit des Worts Wald')
plt.ylabel('Häufigkeit des Worts Menschheit \n')
plt.title('Beispiel: Distanz zwischen Gruppen')
plt.grid(False)
plt.xlim(-5,40)
#plt.axis('equal')
#plt.tight_layout()
plt.savefig(os.path.join(local_temp_directory(system), "figures","distance_examples.svg"))
plt.show()


triangle_vertices = np.array([[1, 1], [10, 2], [4, 10]])

# Define points for the blue line
blue_line_points = np.array([[3, 4], [4, 3]])

# Calculate midpoint between each vertex and corresponding point on the blue line
midpoints = [(v + p) / 2 for v, p in zip(triangle_vertices, blue_line_points)]
print(triangle_vertices[:, 0])
# Plot triangle with labels for endpoints
plt.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], 'r-')
plt.plot([4, 1], [10, 1], 'r-')
for i, (x, y) in enumerate(triangle_vertices):
    plt.text(x, y, f'Roman-{i+1}', ha='right')

# Plot blue line with labels for endpoints
plt.plot(blue_line_points[:, 0], blue_line_points[:, 1], 'b-')
for i, (x, y) in enumerate(blue_line_points):
    plt.text(x, y, f'Märchen-{i+1}', ha='right')

# Plot green dotted lines connecting vertices to points on blue line
print(list(zip(triangle_vertices, midpoints)))

from itertools import combinations

# import itertools package
import itertools
from itertools import permutations

# initialize lists
list_1 = list(triangle_vertices)
list_2 = list(blue_line_points)

unique_combinations = []

for i in range(len(list_1)):
    for j in range(len(list_2)):
        unique_combinations.append((list_1[i], list_2[j]))

print(unique_combinations)


for v, m in unique_combinations:
    plt.plot([v[0], m[0]], [v[1], m[1]], 'g--')

plt.xlabel('Häufigkeit des Worts Wald')
plt.ylabel('Häufigkeit des Worts Menschheit \n')
plt.title('Beispiel: Distanz zwischen Gruppen')
plt.grid(False)
plt.xlim(-1, 12)
#plt.axis('equal')
#plt.tight_layout()
plt.savefig(os.path.join(local_temp_directory(system), "figures","distance_examples2.svg"))
plt.show()