import numpy as np
from random import randint
from ase.build import graphene_nanoribbon
from ase.io import write
from ase.visualize import view
from ase import units
from ase.optimize import BFGS
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.calculators.emt import EMT

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy


def pristine_graphene(dim=[10, 10]):
    """
    piece of grapheene of the target dim.
    :param dim: dimsnions.[x,z] how many tiles on x and ho many on x
    :return:
    """
    graphene_sheet = graphene_nanoribbon(dim[0], dim[1], type='zigzag', saturated=False,
                                         vacuum=3.5, )
    return graphene_sheet


def introduce_vacancies(atoms, num_vacancies):
    """
    Introduce vacancies randomly into the graphene nanoribbon.
    """

    for _ in range(int(num_vacancies)):
        n_atoms = len(atoms)
        index = randint(0, n_atoms - 2)
        del atoms[index]


def introduce_substitutions(atoms, substitution_element, num_substitutions):
    """
    Introduce substitutions randomly into the graphene nanoribbon.
    """
    n_atoms = len(atoms)
    for _ in range(num_substitutions):
        index = randint(0, n_atoms - 1)
        atoms[index].symbol = substitution_element


def relax_structure(structure, steps=1000, fmax=0.05, temperature=300, timestep=1.0):
    """
    Relax the structure a bit to not be completely SF.
    :param structure: graphene structure as ase object
    :param steps: nr of steps for VelocityVerlet
    :param fmax: max force for BFGS
    :param temperature: temperature  for VelocityVerlet in units.kB units
    :param timestep: time step size in * units.fs units for VelocityVerlet
    :return: relaxed structure.
    """

    st = deepcopy(structure)
    st.calc = EMT()

    # Relax the structure using BFGS optimization
    opt = BFGS(st)
    opt.run(fmax=fmax)

    # Perform Langevin dynamics simulation to further relax the structure
    # dyn = Langevin(st, timestep=timestep * units.fs, temperature_K=temperature, friction=0.02)
    # dyn.run(steps)  # Run for the specified number of steps

    # Visualize the relaxed structure
    st.center()  # Center the structure for visualization

    return st


def main():
    graphene_sheet = pristine_graphene()
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
    ax.set_title('DummyGraphene Sheet')

    plt.show()

    view(graphene_sheet)
    introduce_vacancies(graphene_sheet, 5)
    view(graphene_sheet)

    graphene_sheet_relax = relax_structure(graphene_sheet, steps=100, fmax=0.05, temperature=300, timestep=1.0)
    view(graphene_sheet_relax)

    # Introduce 5 substitutions with nitrogen ('N') atoms
    introduce_substitutions(graphene_sheet, 'N', 30)
    view(graphene_sheet)
    write('../artificial_graph_database/DummyGraphene/ag.xyz', graphene_sheet)


if __name__ == "__main__":
    main()
