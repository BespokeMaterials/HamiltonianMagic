import numpy as np
from random import randint
from ase.build import graphene_nanoribbon
from ase.io import write
from ase.visualize import view


import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pristine_graphene():
    graphene_sheet = graphene_nanoribbon(5, 10, type='armchair', saturated=False,
                               vacuum=3.5)
    return graphene_sheet

def introduce_vacancies(atoms, num_vacancies):
    """
    Introduce vacancies randomly into the graphene nanoribbon.
    """
    n_atoms = len(atoms)
    for _ in range(num_vacancies):
        index = randint(0, n_atoms - 1)
        del atoms[index]

def introduce_substitutions(atoms, substitution_element, num_substitutions):
    """
    Introduce substitutions randomly into the graphene nanoribbon.
    """
    n_atoms = len(atoms)
    for _ in range(num_substitutions):
        index = randint(0, n_atoms - 1)
        atoms[index].symbol = substitution_element


graphene_sheet=pristine_graphene()
# Extract coordinates of atoms
positions = graphene_sheet.positions

# Plot 3D graphene structure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of atoms
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1, c='black')

# Set plot labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Graphene Sheet')


plt.show()

view(graphene_sheet)
introduce_vacancies(graphene_sheet, 5)
view(graphene_sheet)

# Introduce 5 substitutions with nitrogen ('N') atoms
introduce_substitutions(graphene_sheet, 'N', 30)
view(graphene_sheet)
write('../artificial_graph_database/Graphene/ag.xyz', graphene_sheet)