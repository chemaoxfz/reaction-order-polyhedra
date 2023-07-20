# reaction-order-polyhedra
 code to compute reaction order polyhedra of a binding network


# Utility functions for binding networks
The code in binding_network.py defines the binding network object class and its other classes and methods relevant for computation of binding networks.

The hierarchy of objects is the following.

binding network > vertex regimes > dom regimes > ld regimes "A>B" means B always has an A as a parent object that it belongs to.

A binding network has multiple vertex regimes.

If a (catalytic) activity is specified, then we can define which regimes are dominant for this activity, corresponding to dominant species (or rows for the A or H matrices of a vertex) of a dominant vertex. So adding activity on top of a binding network yields dom_regimes.

But multiple dom_regimes can have the same ld (log derivative or reaction order). Since ld is what we often care about, we define ld_regimes, specified by ld, so it can have several dom_regimes if there are indeed multiple dom_regimes with this ld.

Then we can add constraints on top to ask, for a given activity and binding network, whether a dom_regime is feasibile, defined by whether the region reaching this dom_regime has nonzero projective measure in x space (i.e. a ray from origin to infinity is an infinitesimal of measure, so a band to infinity is zero measure, while a wedge to infinity has nonzero measure). This labels each dom_regime to be feasible or not. This can also label ld_regimes for whether they are feasible, where one ld_regime is feasible if it has at least one dom_regime that is feasible.

Notation: x is vector of species, of length n. Number of binding reactions is m, and the rank of the binding network (i.e. the number of linearly independent binding reactions) is r. The number of conserved species is d=n-r.
