import copy
from scipy.spatial import cKDTree
from .site_list import SiteList

#THE ONLY INPUT OF GET_NIEHGBORS SHOULD BE SYSTEM

def __create_supercell(system, pbc = [True,True,True]):
    scsites= SiteList()
    lat = system.lattice_vectors.T
    N0 = (-1,0,1) if pbc[0] else (0,)
    N1 = (-1,0,1) if pbc[1] else (0,)
    N2 = (-1,0,1) if pbc[2] else (0,)

    for _site in system.get_sites():
        for n0 in N0:
            for n1 in N1:
                for n2 in N2:
                    site = copy.copy(_site)
                    site.set_coord( site.get_coord() + n0*lat[0]+ n1*lat[1]+ n2*lat[2]  )
                    scsites.site_list.append(site)       
    return scsites


def get_neighbor_indexes(syst, cutoff, pbc):
    scsites = __create_supercell(syst, pbc=pbc)
    tree = cKDTree(scsites.coordinates().T)

    indexes =[]
    for site in syst.get_sites():
        indices = tree.query_ball_point(site.coord, r=cutoff)
        indexes.append( [site.uid, scsites.get_uids(indices)] )
    return indexes