"""
The code to generate pieces of graphene.
"""

import random
from graphene_db_generator.dummy_graphene import (pristine_graphene,
                                                  introduce_vacancies,
                                                  introduce_substitutions,
                                                  relax_structure)
from copy import deepcopy
from ase.io import write
from ase.visualize import view


def generate_graphene_vac(dim, nr_of_samples, place_to_save, vacancy_interval):
    slices = []

    pristine = pristine_graphene(dim=dim)
    file_name = "graphene_{}".format(0)
    write(f'{place_to_save}/{file_name}.xyz', pristine)
    slices.append(pristine)
    for i in range(nr_of_samples):
        nv = random.uniform(vacancy_interval[0], vacancy_interval[1])
        file_name = "graphene_{}".format(i)
        new_sl = deepcopy(pristine)
        # add defect vacancy
        introduce_vacancies(new_sl, nv)
        # save the file
        write(f'{place_to_save}/{file_name}.xyz', new_sl)
        slices.append(new_sl)

    return slices


def put_substitution(slices, place_to_save, subst_atoms=["N"], nr_subst=[1, 1]):
    new_slices = []
    for i, slice in enumerate(slices):
        ns = int(random.uniform(nr_subst[0], nr_subst[1]))
        el = random.choice(subst_atoms)
        new_sl = deepcopy(slice)
        introduce_substitutions(new_sl, el, ns)
        file_name = f"vacancy_{i}"
        write(f'{place_to_save}/{file_name}.xyz', new_sl)
        new_slices.append(new_sl)

    return new_slices


def main():
    dim = [10, 10]
    nr_of_samples = 300
    place_to_save_vac = "artificial_graph_database/DummyGraphene/vacancy_2"
    place_to_save_vac_relax = "artificial_graph_database/DummyGraphene/vacancyRelax"
    vacancy_interval = [60, 100]

    place_to_save_subst = "artificial_graph_database/DummyGraphene/substitution_2"
    place_to_save_vac_subst_relax = "artificial_graph_database/DummyGraphene/vacancySubRelax"
    subst_atoms = ["N", "B"]
    nr_subst = [20, 100]

    vac_slice = generate_graphene_vac(dim, nr_of_samples, place_to_save_vac, vacancy_interval)
    view(vac_slice[3])

    subst_slices = put_substitution(vac_slice, place_to_save_subst, subst_atoms=subst_atoms, nr_subst=nr_subst)
    view(subst_slices[3])

    relax_slices_v = []
    for i, slice in enumerate(vac_slice):
        r_slice = relax_structure(slice, steps=100, fmax=0.05, temperature=300, timestep=1.0)
        file_name = f"vacancy_relax_{i}"
        write(f'{place_to_save_vac_relax}/{file_name}.xyz', r_slice)
        relax_slices_v.append(r_slice)
    view(relax_slices_v[3])

    relax_slices_vs = []
    for i, slice in enumerate(subst_slices):
        r_slice = relax_structure(slice, steps=100, fmax=0.07, temperature=300, timestep=1.0)
        file_name = f"vacancy_relax_{i}"
        write(f'{place_to_save_vac_subst_relax}/{file_name}.xyz', r_slice)
        relax_slices_vs.append(r_slice)
    view(relax_slices_vs[3])


if __name__ == "__main__":
    main()
