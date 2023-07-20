class rop_ld_regime:
  """
  The ld_regime object, for a given ld (log derivative or reaction order),
    there could be multiple dominance regimes (defined by dominance relations in x
    for a given catalytic activity on top of a binding network).
  Each ld_regime has possibly several dominance regimes with the same ld.

  Parameters
  ----------
  ld : a tuple of integers
    the log derivative, or reaction order, of this ld_regime.
  b_vec : a numpy array vector
    the b vector defining the activity, b^T x.
  bn : a binding_network object
    the binding network that this ld_regime belongs to.
  is_ray : boolean
    whether this ld_regime is a ray or not.
  dom_regime_keys : list of tuples
    The list of tuples, each tuple is the key for a dominance regime that has
    reaction order (ld) the same as this ld_regime.
  dom_regime_dict : dictionary
    The dictionary with key as the tuple (perm,dom_idx) representing a dominance regime
     and value of the dominance regime object.
  neighbors_dict : dictionary
    The dictionary for ld_regime neighbors of this ld_regime.
    Has four keys, three of them are "finite", "infinite", and "all",
      for finite, infinite (ray) ld_regime neighbors, and all of ld_regime
      neighbors, respectively.
    For each of these keys, we get a dictionary as well, with ld tuple as key and
      the ld_regime object as value.
    The last key is "zero", which maps to neighbors connected via ld_regimes
      that are zero rays. The value is itself a dictionary with "finite", "infinite",
      and "all" mapping to dictionaries with (ld:ld_regime) entries.
  is_feasible : boolean
    Whether this ld_regime is feasible. It is feasible if it has at least one dom_regime
    that is feasible.
  neighbors_constrained_dict : dictionary
    The dictionaory for ld_regime neighbors of this ld_regime that are feasible
    under constraints applied to each dom_regime.
  """
  # b_vec defines an activity, yielding possibly multiple regimes for each vertex,
  # with logder corresponding to different rows of the vertex logder matrix.
  # ld_regime focus on logder, collapsing the same logder to be the same ld_regime,
  # which may come from different vertices.
  # Each ld_regime has multiple dom_regimes and regions of feasibility.
  def __init__(self,ld,is_ray,b_vec,dom_regime_keys,bn):
    '''
    Parameters
    ----------
    ld : a tuple of integers
      the log derivative, or reaction order, of this ld_regime.
    is_ray : boolean
      whether this ld_regime is a ray or not.
    b_vec : a numpy array vector
      the b vector defining the activity, b^T x.
    dom_regime_keys : list of tuples
      The list of tuples, each tuple is the key for a dominance regime that has
      reaction order (ld) the same as this ld_regime.
    bn : a binding_network object
      the binding network that this ld_regime belongs to.
    '''
    self.ld=ld
    self.b_vec=tuple(b_vec)
    self.bn=bn
    self.is_ray=is_ray
    self.dom_regime_keys=dom_regime_keys
    self.dom_regime_dict={key:bn.activity_regime_dict[self.b_vec]['all'][key] for key in dom_regime_keys}
    self.is_feasible=True

  def find_neighbors(self):
    # Construct self.neighbors_dict based on dominance regimes' neighbors,
    # then construct neighbors_constrained_dict.
    neighbors_fin={};neighbors_inf={};neighbors_fin_zero={};neighbors_inf_zero={}
    # Instead of iterating through all ld_regimes, we just look at this ld_regime's
    # dom_regimes and their corresponding ld_regimes as neighbors.
    for key,dom_regime in self.dom_regime_dict.items():
      for nb_key,nb_regime in dom_regime.neighbors_dict['finite'].items():
        neighbors_fin[nb_regime.ld]=self.bn.activity_ld_regime_dict[self.b_vec]['finite'][nb_regime.ld]
      for nb_key,nb_regime in dom_regime.neighbors_dict['infinite'].items():
        neighbors_inf[nb_regime.ld]=self.bn.activity_ld_regime_dict[self.b_vec]['infinite'][nb_regime.ld]
      for nb_key,nb_regime in dom_regime.neighbors_dict['zero']['finite'].items():
        neighbors_fin_zero[nb_regime.ld]=self.bn.activity_ld_regime_dict[self.b_vec]['finite'][nb_regime.ld]
      for nb_key,nb_regime in dom_regime.neighbors_dict['zero']['infinite'].items():
        neighbors_inf_zero[nb_regime.ld]=self.bn.activity_ld_regime_dict[self.b_vec]['infinite'][nb_regime.ld]
    neighbors_all={**neighbors_fin,**neighbors_inf}
    neighbors_all_zero={**neighbors_fin_zero,**neighbors_inf_zero}
    self.neighbors_dict={'all':neighbors_all,'finite':neighbors_fin,'infinite':neighbors_inf,
                         'zero':{'all':neighbors_all_zero,'finite':neighbors_fin_zero,'infinite':neighbors_inf_zero}}

  def update_feasibility(self):
    # Construct self.dom_regime_constrained_dict using dom_regime's is_feasible tag.
    # Then update self.is_feasible tag, True if self.dom_regime_constrained_dict is not empty,
    # since this means there are some dom_regimes satisfying the constraints.
    self.dom_regime_constrained_dict={key:regime for key,regime in self.dom_regime_dict.items() if regime.is_feasible}
    if self.dom_regime_constrained_dict: self.is_feasible=True
    else: self.is_feasible=False

  def update_constrained_neighbors(self):
    # Construct self.neighbors_constrained_dict based on dominance regimes' neighbors_constrained_dict.
    # This does not distinguish neighbors via zero or not.

    neighbors_constrained_fin={}
    neighbors_constrained_inf={}
    # Instead of iterating through all ld_regimes, we just look at this ld_regime's
    # dom_regimes and their corresponding ld_regimes as neighbors.
    for key,dom_regime in self.dom_regime_constrained_dict.items():
      for nb_key,nb_regime in dom_regime.neighbors_constrained_dict['finite'].items():
        neighbors_constrained_fin[nb_regime.ld]=self.bn.activity_ld_regime_dict[self.b_vec]['finite'][nb_regime.ld]
      for nb_key,nb_regime in dom_regime.neighbors_constrained_dict['infinite'].items():
        neighbors_constrained_inf[nb_regime.ld]=self.bn.activity_ld_regime_dict[self.b_vec]['infinite'][nb_regime.ld]
    neighbors_constrained_all={**neighbors_constrained_fin,**neighbors_constrained_inf}
    self.neighbors_constrained_dict={'all':neighbors_constrained_all,
                                     'finite':neighbors_constrained_fin,
                                     'infinite':neighbors_constrained_inf}

class rop_dom_regime:
  """
  A dominance regime (dom_regime) object for a catalytic activity on top of
    a given binding network defined by b^T x, for some b vector.
  A dominance regime is labeled by a tuple ((perm),j) where perm is the permutation
    (a length-n tuple) defining the vertex, and j is the dominant species index
    in activity b^T x.

  Parameters
  ----------
  row_idx : integer
    the integer j indicating x_j is the dominant species in b^T x at this regime.
  b_vec : numpy array vector
    The b vector defining the activity b^T x.
  vertex_perm : tuple of integers
    The tuple of length d (number of totals of binding network) refering to the
      dominance vertex that this dom_regime belongs to.
  vertex : a ROP_vertex object
    The vertex that this dom_regime belongs to. This dom_regime is at this vertex
      with an additional dominance condition for the activity.
  bn : a binding_network object
    the binding network that this dom_regime belongs to.
  ld : tuple
    The reaction order or log derivative of this dom_regime.
    The row_idx row of the vertex's h_mat.
  neighbors_dict : dictionary
    The dictionary for dom_regime neighbors of this dom_regime.
    Has four keys, three of them are "finite", "infinite", and "all",
      for finite, infinite (ray) dom_regime neighbors, and all of dom_regime
      neighbors, respectively.
    For each of these keys, we get a dictionary as well, with a (perm,row_idx)
      tuple as key and the dom_regime object as value.
    The last key is "zero", which maps to neighbors connected via dom_regimes
      that are zero rays in reaction orders (or log derivative).
      The value is itself a dictionary with "finite", "infinite",
      and "all" mapping to dictionaries with (ld:ld_regime) entries.
  is_feasible : bool
    Whether this dom_regime is feasible under the last given constraints.
    If a new set of constraints is given and tested, then this is overwritten.
  neighbors_constrained_dict : dictionary
    The dictionaory for ld_regime neighbors of this ld_regime that are feasible
    under constraints applied to each dom_regime.
  c_mat_add_tk
  c_mat_add_x
  c_mat_add_
  """
  def __init__(self,row_idx,b_vec,vertex_perm,bn):
    """Initiates a ROP vertex.

    Parameters
    ----------
    row_idx : integer
      the integer j indicating x_j is the dominant species in b^T x at this regime.
    b_vec : numpy array vector
      The b vector defining the activity b^T x.
    vertex_perm : tuple of integers
      The tuple of length d (number of totals of binding network) refering to the
      dominance vertex that this dom_regime belongs to.
    bn : a binding_network object
      the binding network that this dom_regime belongs to.
    """
    self.row_idx=row_idx
    self.b_vec=tuple(b_vec)
    self.vertex_perm=vertex_perm
    self.vertex=bn.vertex_dict['all'][vertex_perm]
    self.bn=bn
    self.ld=tuple(self.vertex.h_mat[row_idx,:])
    self.is_feasible=True

  def feasibility_test(self,chart='x',opt_constraints=[],positive_threshold=1e-5,is_asymptotic=True):
    if chart=='x':
      try: c_mat_add=self.c_mat_add_x
      except AttributeError: # c_mat_add is not yet calculated
        self.calc_c_mat_add_x()
        c_mat_add=self.c_mat_add_x
    elif chart=='xak':
      try: c_mat_add=self.c_mat_add_xak
      except AttributeError: # c_mat_add_xak is not yet calculated
        self.calc_c_mat_add_xak()
        c_mat_add=self.c_mat_add_xak
    elif chart=='tk':
      try: c_mat_add=self.c_mat_add_tk
      except AttributeError: # c_mat_add_tk is not yet calculated
        self.calc_c_mat_add_tk()
        c_mat_add=self.c_mat_add_tk
    else:
      raise Exception('chart that is not one of "x,xak,tk" is not implemented yet')

    vv=self.vertex
    opt_var=self.bn.opt_var
    opt_constraints_test=[]
    opt_constraints_test+=opt_constraints
    if np.any(c_mat_add): #c_mat_add could be empty if b_vec has just one nonzero entry.
      if is_asymptotic: # if the test condition is asymptotic, c0_vec_add is considered 0.
        opt_constraints_test+=[c_mat_add @ opt_var >= positive_threshold]
      else: # if not asymptotic, it is exact, so c0_vec_add needs to be considered.
        opt_constraints_test+=[c_mat_add @ opt_var + self.c0_vec_add >= positive_threshold]
    is_feasible=vv.vertex_feasibility_test(chart=chart,opt_constraints=opt_constraints_test,is_asymptotic=is_asymptotic)
    return is_feasible

  def calc_c_mat_add_x(self):
    # compute feasibility conditions of this dominance regime in addition to vertex feasibility conditions
    # in the form of c_mat_add_x * x > 0, i.e. in chart x.
    j=self.row_idx
    b_vec=np.array(self.b_vec)
    nnz_b=np.where(b_vec > 0)[0]
    n_ineq=(len(nnz_b)-1,self.bn.dim_n)
    c_mat_add_x=np.zeros(n_ineq)
    c0_vec_add=np.zeros(n_ineq)
    counter=0
    for jp in nnz_b:
      if jp!=j:
        c_mat_add_x[counter,j]=1
        c_mat_add_x[counter,jp]=-1
        c0_vec_add[counter]=np.log10(b_vec[j])-np.log10(b_vec[jp])
        counter+=1
    self.c_mat_add_x=c_mat_add_x
    self.c0_vec_add=c0_vec_add

  def calc_c_mat_add_xak(self):
    # compute feasibility conditions of this dominance regime in addition to vertex feasibility conditions
    # in the form of c_mat_add_xak * log(xa,k) > 0, i.e. in chart log(xa,k)
    try: c_mat_add_x=self.c_mat_add_x
    except AttributeError: # c_mat_add is not yet calculated
      self.calc_c_mat_add_x()
      c_mat_add_x=self.c_mat_add_x
    try: xak2x_map=self.bn.xak2x_map
    except AttributeError:
      self.bn.calc_xak2x_map()
      xak2x_map=self.bn.xak2x_map
    c_mat_add_xak=c_mat_add_x.dot(xak2x_map)
    self.c_mat_add_xak=c_mat_add_xak

  def calc_c_mat_add_tk(self):
    # compute feasibility conditions of this dominance regime in addition to vertex feasibility conditions
    # in the form of c_mat_add_tk * log(t,k) > 0, i.e. in chart log(t,k)
    try: c_mat_add_x=self.c_mat_add_x
    except AttributeError: # c_mat_add is not yet calculated
      self.calc_c_mat_add_x()
      c_mat_add_x=self.c_mat_add_x
    try: tk2x_map=self.vertex.tk2x_map #tk2x_map is vertex-specific
    except AttributeError:
      self.vertex.vertex_calc_tk2x_map()
      tk2x_map=self.vertex.xak2x_map
    c_mat_add_tk=c_mat_add_x.dot(tk2x_map)
    self.c_mat_add_tk=c_mat_add_tk

  def find_neighbors(self):
    vv=self.vertex
    neighbors_fin={}
    neighbors_inf={}
    regime_fin_dict=self.bn.activity_regime_dict[tuple(self.b_vec)]['finite']
    regime_inf_dict=self.bn.activity_regime_dict[tuple(self.b_vec)]['infinite']
    for key,regime in regime_fin_dict.items():
      perm,row_idx=key
      # if same vertex, but different dominance, then it's neighbor.
      if perm==self.vertex_perm and row_idx!=self.row_idx:
        neighbors_fin[key] = regime
      # if different vertex, but it's a neighboring vertex, then it's neighbor.
      else:
        try: # if perm is in neighbors
          vv.neighbors_dict['all'][perm]
          # If the neighboring vertex's transition index (the row_idx that changed compared to self)
          # is the same as its dominance index, then it's a neighbor.
          # Or, if the transition index is not the same, then its dominance index needs to be the same as self.
          if row_idx == self.row_idx or perm[np.where(np.array(perm)!=self.vertex_perm)[0][0]]==row_idx:
            neighbors_fin[key] = regime
        except KeyError: # not in neighbors
          continue


    for key,regime in regime_inf_dict.items():
      perm,row_idx=key
      # if same vertex, but different dominance, then it's neighbor.
      if perm==self.vertex_perm and row_idx!=self.row_idx:
        neighbors_inf[key] = regime
      # if different vertex, but it's a neighboring vertex, then it's neighbor.
      else:
        try: # if perm is in neighbors
          vv.neighbors_dict['all'][perm]
          if row_idx == self.row_idx or perm[np.where(np.array(perm)!=self.vertex_perm)[0][0]]==row_idx:
            neighbors_inf[key] = regime
        except KeyError: # not in neighbors
          continue
    neighbors_all = {**neighbors_fin,**neighbors_inf}
    self.neighbors_dict={'finite':neighbors_fin,'infinite':neighbors_inf,'all':neighbors_all}

  def find_neighbors_zero(self):
    # Find neighbors that connect through dom_regimes with zero ld.
    # neighbors via zero are not considered neighbors of dom_regimes,
    # but we would like to keep them tracked, since they will become neighbors
    # in ld space.
    # This should be done after all neighbors of dom_regimes are constructed
    # since it relies on knowing the neighbors of dom_regime with ld=0.
    # A zero neighbor could have a zero neighbor, so recursion is needed to exhaust this.
    def extract_zero_neighbor(infinite_neighbors, finite_zero_neighbors, infinite_zero_neighbors, ld_zero, visited):
      zero_neighbors = {
          key: dom_regime for key, dom_regime in infinite_neighbors.items() if tuple(dom_regime.ld) == ld_zero
      }
      if zero_neighbors:  # the dictionary is not empty
          for key, dom_regime in zero_neighbors.items():
              if key not in visited:
                  visited.add(key)
                  finite_zero_neighbors.update(dom_regime.neighbors_dict['finite'])
                  infinite_zero_neighbors.update(dom_regime.neighbors_dict['infinite'])
                  finite_zero_neighbors, infinite_zero_neighbors, visited = extract_zero_neighbor(
                      dom_regime.neighbors_dict['infinite'],
                      finite_zero_neighbors,
                      infinite_zero_neighbors,
                      ld_zero,
                      visited
                  )
      return finite_zero_neighbors, infinite_zero_neighbors, visited

    n = self.bn.dim_n
    ld_zero = tuple(np.zeros(n))
    finite_zero_neighbors = {}
    infinite_zero_neighbors = {}
    visited = set()

    infinite_neighbors = self.neighbors_dict['infinite']
    finite_zero_neighbors, infinite_zero_neighbors, visited = extract_zero_neighbor(
        infinite_neighbors, finite_zero_neighbors, infinite_zero_neighbors, ld_zero, visited
    )

    # Remove self from the neighbors' dictionaries if present
    try:
        if self.vertex.orientation == 0:
            del infinite_zero_neighbors[(self.vertex_perm, self.row_idx)]
        else:
            del finite_zero_neighbors[(self.vertex_perm, self.row_idx)]
    except KeyError:
        pass

    all_zero_neighbors = {**finite_zero_neighbors, **infinite_zero_neighbors}
    self.neighbors_dict['zero'] = {
        'finite': finite_zero_neighbors,
        'infinite': infinite_zero_neighbors,
        'all': all_zero_neighbors
    }
    self.neighbors_dict['allnzero'] = {
        'finite': {**finite_zero_neighbors,**self.neighbors_dict['finite']},
        'infinite': {**infinite_zero_neighbors,**self.neighbors_dict['infinite']},
        'all': {**all_zero_neighbors,**self.neighbors_dict['all']}
    }

  def update_constrained_neighbors(self):
    # Using dom_regimes' is_feasible tag to have updated neighbors.
    # The feasible neighbors of an infeasible neighbor becomes this vertex's neighbors.
    # We search the next level for the infeasible neighbors of an infeasible neighbor.
    # So this uses recursion.

    def extract_infeasible_neighbor(infeasible_neighbors,neighbors_feasible_all,visited):
      for key,dom_regime in infeasible_neighbors.items():
          if key not in visited:
              visited.add(key)
              # get the feasible and infeasible neighbors of this dom_regime
              infeasible_neighbors = {}
              feasible_neighbors={}
              for key,dom_regime in dom_regime.neighbors_dict['allnzero']['all'].items():
                if dom_regime.is_feasible:
                  feasible_neighbors[key]=dom_regime
                else:
                  infeasible_neighbors[key]=dom_regime
              # the feasible ones are neighbors under constraint.
              neighbors_feasible_all.update(feasible_neighbors)
              # the infeasible ones we need to look further
              if infeasible_neighbors:
                neighbors_feasible_all, visited = extract_infeasible_neighbor(
                  infeasible_neighbors,
                  neighbors_feasible_all,
                  visited
                )
      return neighbors_feasible_all, visited

    neighbors_feasible_all = {key:dom_regime for key,dom_regime in self.neighbors_dict['allnzero']['all'].items() if dom_regime.is_feasible}
    infeasible_neighbors = {key:dom_regime for key,dom_regime in self.neighbors_dict['allnzero']['all'].items() if not dom_regime.is_feasible}
    visited = set()
    if infeasible_neighbors:
      neighbors_feasible_all,visited = extract_infeasible_neighbor(
        infeasible_neighbors,neighbors_feasible_all,visited
      )
    # Remove self from the neighbors' dictionaries if present
    try:
        del neighbors_feasible_all[(self.vertex_perm,self.row_idx)]
    except KeyError:
        pass
    neighbors_feasible_inf = {
        key:dom_regime for key,dom_regime in neighbors_feasible_all.items() if dom_regime.vertex.orientation==0
        }
    neighbors_feasible_fin = {
        key:dom_regime for key,dom_regime in neighbors_feasible_all.items() if dom_regime.vertex.orientation!=0
        }
    self.neighbors_constrained_dict={'finite':neighbors_feasible_fin,
                                  'infinite':neighbors_feasible_inf,
                                  'all':neighbors_feasible_all}

  # def print_validity_condition(self,is_asymptotic=False):
  #   # print the expression for t=x, x(t,k) and inequalities for the
  #   # region of validity for this dominance regime,
  #   # using the labels of x,t,k

class rop_vertex:
  """
  A vertex object for reaction order polyhedra.
  Each binding network has multiple vertices.

  Parameters
  ----------
  perm : an integer tuple
    an integer tuple of length dim_d, indicating for each conserved quantity, which species is dominant.
  p_mat : numpy array
    d x n matrix with exactly one nonzero entry in each row, of value 1. One-hot representation of perm.
  orientation : an integer
    take value in +1,0,-1. Sign of determinant of [A' N'] matrix.
  neighbors : list of perm
    a list of perm that correspond to vertices neighboring this vertex,
    i.e. they can be reached by changing one dominance condition in this binding network
  bn : a binding network object
    The binding network that this vertex belongs to.
  h_mat : numpy array
    n x n matrix corresponding to the log derivative of this vertex
    if finite, then this is the log derivative
    if infinite, then this is the direction that log derivative goes into.
  m_mat : numpy array
    n x n matrix, concatenation of p_mat with stoichiometry matrix of the binding network.
  c_mat_x : numpy array
    Matrix encoding feasibility condition of this vertex in 'x' chart, c_mat_x * logx + c0_vec > 0.
    If the feasibility condition is considered "asymptotic", i.e. in positive
      projective measure rather than Lebesgue measure (so a ray is an infinitesimal
      of volume, not a point), then c0_vec is dropped.
    Not always defined, computed and stored when used in feasibility tests.
  c_mat_xak : numpy array
    Matrix encoding feasibility condition of this vertex in 'xak' chart, c_mat_xak * (logxa, logk) + c0_vec > 0.
    If the feasibility condition is considered "asymptotic", i.e. in positive
      projective measure rather than Lebesgue measure (so a ray is an infinitesimal
      of volume, not a point), then c0_vec is dropped.
    Not always defined, computed and stored when used in feasibility tests.
  c0_vec : numpy vector
    Numpy vector encoding a part of the feasibility condition of this vertex, same
      for 'x' chart and 'xak' chart.
    Not always defined, computed and stored when used in feasibility tests.
  c_mat_tk : numpy array
    Matrix encoding feasibility condition of this vertex in 'tk' chart, c_mat_tk * (logt, logk) + c0_vec_tk > 0.
    If the feasibility condition is considered "asymptotic", i.e. in positive
      projective measure rather than Lebesgue measure (so a ray is an infinitesimal
      of volume, not a point), then c0_vec_tk is dropped.
    Not always defined, computed and stored when used in feasibility tests.
  c0_vec_tk : numpy vector
    Numpy vector encoding a part of the feasibility condition of this vertex in the
      'tk' chart.
    Not always defined, computed and stored when used in feasibility tests.
  """
  def __init__(self,perm,bn):
    """Initiates a ROP vertex

    Parameters
    ----------
    perm : an integer tuple
      an integer tuple of length dim_d, indicating for each conserved quantity, which species is dominant.

    bn : a binding network object
      the binding network that this vertex is a part of.
      used to get dimensions and stoichiometry matrix.
    """
    self.perm=perm
    self.bn=bn
    p_mat=np.zeros((bn.dim_d,bn.dim_n)) # p_mat is all 0 and 1
    p0_vec=np.zeros(bn.dim_d) # the value of nonzero entries of l_mat chosen by p_mat in each row for this vertex
    for i in range(bn.dim_d):
      p_mat[i,perm[i]]=1
      p0_vec[i]=np.log10(bn.l_mat[i,perm[i]])
    self.p_mat=p_mat
    self.p0_vec=p0_vec
    self.m_mat=np.concatenate((self.p_mat,self.bn.n_mat),axis=0)
    self.m0_vec=np.concatenate((self.p0_vec,np.zeros(self.bn.dim_r)),axis=0)
    self.orientation=np.sign(np.linalg.det(self.m_mat))
    self.is_feasible=True

  def vertex_c_mat_x_calc(self):
    # Get feasibility condition for this vertex expressed in (x) chart.
    # c_mat*log(x) + c0_vec > 0 is the validity condition, log is log10.
    l_mat=self.bn.l_mat
    j_list=self.j_func(self.p_mat,l_mat)
    dim=np.sum([len(j) for j in j_list])
    c_mat_x=np.zeros((dim,self.bn.dim_n))
    c0_vec = np.zeros(dim)
    dim_n=self.bn.dim_n
    counter=0
    for i in range(self.bn.dim_d):
      j_i=self.perm[i]
      for j in j_list[i]:
        c_mat_x[counter,j_i]=1
        c_mat_x[counter,j]=-1
        c0_vec[counter]=np.log10(l_mat[i,j_i])-np.log10(l_mat[i,j])
        counter+=1
    self.c_mat_x=c_mat_x
    self.c0_vec = c0_vec

  def j_func(self,p_mat,l_mat):
    """
    For p_mat and l_mat, find (column) indices that are nonzero in each row of
      l_mat but not nonzero in p_mat.

    Parameters
    ----------
    p_mat : numpy array
      d-by-n matrix, each row has only one nonzero entry that is 1.
    l_mat : numpy array
      d-by-n matrix, binding network's conservation law matrix.

    Returns
    -------
    j_list: list of int
      List of (column) indices that are nonzero in l_mat but zero in p_mat.
    """
    j_list=[]
    l_mat_masked=np.ma.array(l_mat,mask=p_mat).filled(fill_value=0)
    for i in range(p_mat.shape[0]):
      j_list+=[np.nonzero(l_mat_masked[i,:])[0]]
    return j_list

  def vertex_feasibility_test(self,chart='x',opt_constraints=[],positive_threshold=1e-5,is_asymptotic=True):
    """
    For given constraints, compute whether this vertex is feasible.

    Parameters
    ----------
    chart : str, optional
      A string indicating the chart that the opt_constraints are specified in.
      Choices are 'x','xak', and 'tk'.
    opt_constraints : list of cvxpy inequalities, optional
      A list of optimization constraints specified in terms of inequalities
        relating cvxpy variables of the binding network.
    positive_threshold : float, optional
      The vertex itself has inequality conditions, of the form c_mat*x + c0_vec > th,
        where th is the positive threshold used here. Default to 1e-5.
    is_asymptotic : boolean, optional
      Whether the inequalities of the vertex itself should be considered asymptotically
        or exactly. If asymptotically, then the inequality tested ommits c0_vec,
        so it is c_mat*x > th. is_asymptotic=True corresponds to inequality satisfied
        for the positive projective measure (where a ray is an infinitesimal of volume),
        and is_asymptotic=False is Lebesgue measure (a point is an infinitesimal of volume).

    Returns
    -------
    is_feasible: boolean
       whether this vertex is feasible under the constraints.
    """
    # first prepare the c_mat and c0_vec for the desired chart.
    if chart=='x':
      try:
        c_mat=self.c_mat_x
        c0_vec=self.c0_vec
      except AttributeError: # if c_mat_x is not yet calculated
        self.vertex_c_mat_x_calc()
        c_mat=self.c_mat_x
        c0_vec=self.c0_vec
    elif chart=='xak':
      try:
        c_mat=self.c_mat_xak
        c0_vec=self.c0_vec #c0_vec is the same for chart x and xak.
      except AttributeError: # if c_mat_xak is not yet calculated
        self.vertex_c_mat_xak_calc()
        c_mat=self.c_mat_xak
        c0_vec=self.c0_vec
    elif chart=='tk':
      try:
        c_mat=self.c_mat_tk
        c0_vec=self.c0_vec_tk
      except AttributeError: # if c_mat_xak is not yet calculated
        self.vertex_c_mat_tk_calc()
        c_mat=self.c_mat_tk
        c0_vec=self.c0_vec_tk
    else:
      raise Exception('chart that is not one of x,xak or tk is not implemented yet')
    is_feasible=True
    opt_var=self.bn.opt_var #opt_var came from the binding network.
    if is_asymptotic: #if asymptotic, c0_vec is zero.
      prob=cp.Problem(cp.Minimize(1),opt_constraints+[c_mat @ opt_var >= positive_threshold])
    else: # if not asymptotic, it is exact, so c0_vec is taken into account.
      prob=cp.Problem(cp.Minimize(1),opt_constraints+[c_mat @ opt_var + c0_vec >= positive_threshold])
    prob.solve()
    if prob.status=='infeasible':
      is_feasible=False
    return is_feasible

  def vertex_find_neighbors(self):
    # given a binding network with (already-computed) vertex dictionaries,
    # compute this vertex's neighbors and store in self.neighbors,
    # depending on whether it is finite or infinite.
    neighbors_fin={}
    neighbors_inf={}
    for perm,vv in self.bn.vertex_dict['finite'].items():
      # temp=[not np.all(self.p_mat[i,:]==vv.p_mat[i,:]) for i in range(self.bn.dim_d)]
      # if np.sum(temp)==1: #the difference is just one row
      if np.sum(np.abs(self.p_mat-vv.p_mat))==2: #difference is in just one row and it's -1, 1.
        neighbors_fin[perm]=vv
    for perm,vv in self.bn.vertex_dict['infinite'].items():
      # temp=[not np.all(self.p_mat[i,:]==vv.p_mat[i,:]) for i in range(self.bn.dim_d)]
      # if np.sum(temp)==1: #the difference is just one row
      if np.sum(np.abs(self.p_mat-vv.p_mat))==2: #difference is in just one row and it's -1, 1.
        neighbors_inf[perm]=vv
    neighbors_all={**neighbors_fin,**neighbors_inf}
    self.neighbors_dict={'finite':neighbors_fin,'infinite':neighbors_inf,'all':neighbors_all}


  def vertex_update_constrained_neighbors(self):
    # Using vertices' is_feasible tag to have updated neighbors under constraints.
    # The feasible neighbors of an infeasible neighbor becomes this vertex's neighbors
    # under constraints.
    # If an infeasible neighbor has infeasible neighbors as well, we need to look at
    # their feasible neighbors as neighbors under constraint as well.
    # So this uses recursion.
    # The resulting constrained neighbors is stored in self.neighbors_constrained_dict.

    def extract_infeasible_neighbor(neighbors_all,neighbors_feasible_all,visited):
      infeasible_neighbors = {
        perm: vv for perm,vv in neighbors_all.items() if not vv.is_feasible
      }
      if infeasible_neighbors:  # the dictionary is not empty
        for perm,vv in infeasible_neighbors.items():
            if perm not in visited:
                visited.add(perm)
                neighbors_feasible_all.update({perm:vv for perm,vv in vv.neighbors_dict['all'].items() if vv.is_feasible})
                neighbors_feasible_all, visited = extract_infeasible_neighbor(
                  vv.neighbors_dict['all'],
                  neighbors_feasible_all,
                  visited
                )
      return neighbors_feasible_all, visited

    # first, each neighbor that is feasible should be a neighbor under constraint.
    neighbors_feasible_all = {perm:vv for perm,vv in self.neighbors_dict['all'].items() if vv.is_feasible}
    visited = set()

    neighbors_feasible_all,visited = extract_infeasible_neighbor(
      self.neighbors_dict['all'],neighbors_feasible_all,visited
    )
    # Remove self from the neighbors' dictionaries if present
    try:
        del neighbors_feasible_all[self.perm]
    except KeyError:
        pass
    neighbors_feasible_inf = {
        perm:vv for perm,vv in neighbors_feasible_all.items() if vv.orientation==0
        }
    neighbors_feasible_fin = {
        perm:vv for perm,vv in neighbors_feasible_all.items() if vv.orientation!=0
        }
    self.neighbors_constrained_dict={'finite':neighbors_feasible_fin,
                                  'infinite':neighbors_feasible_inf,
                                  'all':neighbors_feasible_all}


  def vertex_ld_calc(self):
    # calculate log derivative matrix,
    # gives ray direction matrix if infinite vertex
    dim_n = self.bn.dim_n
    m_mat = self.m_mat
    if self.orientation==0: # if singular, get the ray direction
      m_mult_left=np.kron(np.eye(dim_n),m_mat)
      m_mult_right=np.kron(m_mat.T,np.eye(dim_n))
      temp=np.concatenate( (m_mult_left,m_mult_right),axis=0)
      # the following assertion test is not necessary if guaranteed to be rank 1 singularity.
      # assert temp.shape[1]-np.linalg.matrix_rank(temp)==1, "the vertex is singular in more than one direction."
      # we assert that an infinite vertex cannot be singular in more than one direction
      # so the ray is always rank 1. We assume higher order rays are convex combinations of first order rays.
      rslt=null_space(temp)
      # the following normalizes the entries to largest entry = 1.
      h_mat_coarse=np.reshape(rslt,(dim_n,dim_n),order='F') #order='F' is earlier index changes first. here fortran
      temp=np.abs(h_mat_coarse)
      minfactor=np.min(temp[temp>1e-7])
      h_mat_coarse_int=h_mat_coarse/minfactor #all non-zero entries are now integers
      h_mat_int=np.rint(h_mat_coarse_int)
      assert np.max(np.abs(h_mat_int-h_mat_coarse_int)) <1e-3, 'rounding of h_mat for vertices caused large error.'
      h_mat=h_mat_int/np.max(h_mat_int) # normalize largest entry to 1.

      # now we need to check the directionality. This utilizes neighbors
      vv_nb=list(self.neighbors_dict['finite'].values())[0]
      if np.sign(np.linalg.det(vv_nb.h_mat+1e5*h_mat))!=self.bn.orientation:
        h_mat=-1*h_mat
    else: # if not singular, just invert the matrix.
      h_mat = np.linalg.inv(m_mat)
      self.h0_vec = -h_mat.dot(self.m0_vec) # only non-singular vertices have h0_vec.
    self.h_mat=h_mat

  def vertex_c_mat_xak_calc(self):
    # get the C matrix but for log(xa,k) coordinate
    # store as self.c_mat_xak
    try: c_mat_x=self.c_mat_x
    except AttributeError:
      self.vertex_c_mat_x_calc()
      c_mat_x=self.c_mat_x
    try: xak2x_map=self.bn.xak2x_map
    except AttributeError:
      self.bn.calc_xak2x_map()
      xak2x_map=self.bn.xak2x_map
    c_mat_xak=c_mat_x.dot(xak2x_map)
    self.c_mat_xak=c_mat_xak

  def vertex_c_mat_tk_calc(self):
    # get the C matrix but for log(t,k) coordinate
    # store as self.c_mat_tk
    assert self.orientation!=0, "only finite vertices can have non-singular (t,k) chart"
    try:
      c_mat_x=self.c_mat_x
    except AttributeError:
      self.vertex_c_mat_x_calc()
      c_mat_x=self.c_mat_x
    c_mat_tk=c_mat_x.dot(self.h_mat) #because h_mat*log(t,k)=log(x)
    c0_vec_tk=self.c0_vec - c_mat_x.dot(self.h_mat.dot(self.m0_vec))
    self.c_mat_tk=c_mat_tk
    self.c0_vec_tk=c0_vec_tk

  def vertex_print_validity_condition(self,is_asymptotic=False):
    # print the expression for t=x, x(t,k) and inequalities for the
    # region of validity given a regime,
    # using the labels of x,t,k
    try:
      c_mat=self.c_mat_tk
      c0_vec=self.c0_vec_tk
    except AttributeError: # if c_mat_xak is not yet calculated
      self.vertex_c_mat_tk_calc()
      c_mat=self.c_mat_tk
      c0_vec=self.c0_vec_tk
    h_mat=self.h_mat
    h0_vec=self.h0_vec
    p0_vec=self.p0_vec
    perm=self.perm
    x_sym=self.bn.x_sym
    tk_sym=self.bn.tk_sym

    print('======This is vertex perm '+str(perm)+'======\n')
    t2x_list=[]
    for i in range(self.bn.dim_d):
      p0=p0_vec[i]
      if p0==1 or is_asymptotic:
        t2x_list+=[tk_sym[i].name + '=' + x_sym[perm[i]].name]
      else:
        t2x_list+=[tk_sym[i].name + '=' + str(p0) + x_sym[perm[i]].name]
    print('(1) t in x: t is dominated by... \n'+ ','.join(t2x_list) + '\n')

    # BELOW we obtain x's expression in terms of t,k by
    # log(x) = h_mat * log(t,k) + h0_vec

    if self.orientation==0:
      print('(2) x in tk: Vertex is singular, cannot express (x) in (t,k).')
    else:
      x2tk_list=[]
      for i in range(self.bn.dim_n):
        h_row=h_mat[i,:]
        pos_idx=np.where(h_row>0)[0]
        neg_idx=np.where(h_row<0)[0]
        expr_pos=' '.join([tk_sym[j].name + '^' + str(h_row[j]) if h_row[j]!=1.0 else tk_sym[j].name for j in pos_idx])
        expr_neg=' '.join([tk_sym[j].name + '^' + str(-h_row[j]) if -h_row[j]!=1.0 else tk_sym[j].name for j in neg_idx])
        if neg_idx.size==0:
          expr=expr_pos
        else:
          expr=expr_pos + ' / ' + expr_neg
        h0=h0_vec[i]
        if h0==0:
          x2tk_list+=[x_sym[i].name + ' = '+ expr]
        else:
          x2tk_list+=[x_sym[i].name + ' = ' + expr + ' / '+str(10**h0)]
      print('(2) x in tk: \n'+',\n'.join(x2tk_list) + '\n')


    ineq_list=[]
    for i in range(c_mat.shape[0]):
      c_row=c_mat[i,:]
      pos_idx=np.where(c_row>0)[0]
      neg_idx=np.where(c_row<0)[0]
      expr_pos=' '.join([tk_sym[j].name + '^'+str(c_row[j]) if c_row[j]!=1.0 else tk_sym[j].name for j in pos_idx])
      expr_neg=' '.join([tk_sym[j].name + '^'+str(-c_row[j]) if -c_row[j]!=1.0 else tk_sym[j].name for j in neg_idx])
      c0=c0_vec[i]
      if c0==0:
        ineq_list+=[expr_pos + ' > ' + expr_neg]
      else:
        ineq_list+=[expr_pos + ' > ' + str(10**(-c0)) + ' ' + expr_neg]

    print('(3) BELOW is constraints for regions of validity \n'+',\n'.join(ineq_list) + '\n')


class binding_network:
  """A binding network object.

  Parameters
  ----------
  n_mat : numpy array
    The stoichiometry matrix defining the binding network.
  is_atomic: 'bool'
    Is the binding network atomic or not.
  l_mat : numpy array
    The conservation law matrix defining the conserved total quantities.
    The default will be calculated if the network is atomic.
  dim_n : 'int'
    The number of chemical species, same as number of columns of n_mat and l_mat.
  dim_r :
    The number of (linearly independent) binding reactions, same as number of rows of n_mat.
  dim_d :
    The number of conserved quantities or totals, same as the number of rows of l_mat.
  x_sym :
    The list of symbols for the chemical species.
  t_sym :
    The list of symbols for the totals.
  k_sym :
    The list of symbols for the binding constants.
  """

  def __init__(self,
               n_mat,
               l_mat=np.array([]),
               is_atomic=False,
               x_sym=np.array([]),
               t_sym=np.array([]),
               k_sym=np.array([])):
    """Initiate a binding network.

    Parameters
    ----------
    n_mat : numpy array
        The stoichiometry matrix defining the binding network.
    is_atomic: 'bool', optional
        Is the binding network atomic or not.
    l_mat : numpy array, optional
        The conservation law matrix defining the conserved total quantities.
        The default will be calculated if the network is atomic.
    x_sym : list of symbols, optional
        An ordered list of symbols for the chemical species, denoting columns of the n_mat and l_mat.
    t_sym : list of symbols, optional
        An ordered list of symbols for the total or conserved quantities, denoting rows of the l_mat.
    k_sym : list of symbols, optional
        An ordered list of symbols for the binding constants in the dissociation direction, denoting rows of the n_mat.
    """
    self.is_atomic=is_atomic
    self.n_mat=n_mat
    self.n_mat_sym=sp.Matrix(n_mat)
    self.dim_r,self.dim_n=self.n_mat.shape
    self.dim_d = self.dim_n - self.dim_r
    if not np.any(l_mat): #no input l_mat
      l_mat=self.l_from_n(self.n_mat)
      assert np.all(l_mat>=0)
    # if no input symbols for the x,t,k variables then give them default numerically ordered ones.
    if not np.any(x_sym):
      x_sym = sp.symbols("x:"+str(self.dim_n))
    if not np.any(t_sym):
      t_sym = sp.symbols("t:"+str(self.dim_d))
    if not np.any(k_sym):
      k_sym = sp.symbols("k:"+str(self.dim_r))

    self.l_mat=l_mat
    self.l_mat_sym=sp.Matrix(l_mat)
    self.x_sym=x_sym
    self.t_sym=t_sym
    self.k_sym=k_sym
    self.tk_sym=t_sym+k_sym
    self.opt_var=cp.Variable(self.dim_n)

    self.m_mat=np.concatenate((self.l_mat,self.n_mat),axis=0)
    self.orientation=np.sign(np.linalg.det(self.m_mat))

    self.ld_sym_dict={}
    self.activity_regime_dict={}
    self.activity_ld_regime_dict={}
    self.activity_regime_constrained_dict={}
    self.activity_ld_regime_constrained_dict={}

  def l_from_n(self,n_mat):
    """if the network is atomic, compute the L matrix from the N matrix

    Parameters
    ----------
    n_mat : numpy array
        The stoichiometry matrix defining the binding network.
        Assumes the columns are ordered so that the atomic species come first.

    Returns
    -------
    l_mat: numpy array
       The conservation law or totals matrix.
    """
    assert self.is_atomic
    r=n_mat.shape[0]
    d=n_mat.shape[1]-r
    n1_mat=n_mat[:,:d]
    n2_mat=n_mat[:,-r:]
    l2_mat=-(n1_mat.T).dot(np.linalg.inv(n2_mat).T)
    l_mat=np.concatenate((np.eye(d),l2_mat),axis=1)
    return l_mat

  def logder_num(self,x,a_mat=np.array([])):
    """compute the numerical log derivative of the binding network at point x.

    Parameters
    ----------
    x : numpy vector
      Vector of concentrations for all the species.
    a_mat : numpy array, optional
      Matrix defining the variables log derivative is taken in terms of.
      Assumes all entries are non-negative, and each row has at least one positive entry.
      Optional, defaults to l_mat of the binding network.

    Returns
    -------
    logder: numpy array
       n-by-n matrix of log derivative of x to (t,k), where t=l_mat_temp * x, n is number of species.
    """
    #assumes all x are positive
    assert np.all(x>0)
    n_mat=self.n_mat
    if not np.any(a_mat): # no a_mat argument is given
      a_mat=self.l_mat
    else:
      assert a_mat.shape==(self.dim_d,self.dim_n), f"the shape of L matrix should be {self.dim_d} by {self.dim_n}."
      assert np.all(a_mat>=0), "all entries of A matrix should be non-negative."
      assert np.all(a_mat.dot(np.ones(self.dim_n))>0), "each row of A matrix should have at least one positive entry."

    t_inv = 1/(a_mat.dot(x))
    temp=a_mat*x
    upper=(temp.T*t_inv).T
    logder_inv=np.concatenate((upper,n_mat),axis=0)
    return np.linalg.inv(logder_inv)

  def logder_num_atomic(self,xa,k,a_mat=np.array([])):
    """compute the numerical log derivative of dlog(x)/dlog(a_mat*x,k) at a point
    specified by (xa,k), where xa is concentration of atomic species,
    k is binding constants.
    Assumes the network is atomic, and n_mat,a_mat have atomic species coming first.

    Parameters
    ----------
    xa : numpy vector
      Vector of concentrations for the atomic species.
      All entries are positive.
    k : numpy vector
      Vector of binding constants.
      All entries are positive.
    a_mat : a_mat : numpy array, optional
      Matrix defining the variables log derivative is taken in terms of.
      Assumes all entries are non-negative, and each row has at least one positive entry.
      Optional, defaults to l_mat of the binding network.

    Returns
    -------
    logder: numpy array
       n-by-n matrix of log derivative of x to (t,k), where t=l_mat_temp * x, n is number of species.
    """
    assert self.is_atomic
    n_mat=self.n_mat
    if not np.any(a_mat): # no a_mat argument is given
      a_mat=self.l_mat
    else:
      assert a_mat.shape==(self.dim_d,self.dim_n), f"the shape of L matrix should be {self.dim_d} by {self.dim_n}."
      assert np.all(a_mat>=0), "all entries of A matrix should be non-negative."
      assert np.all(a_mat.dot(np.ones(self.dim_n))>0), "each row of A matrix should have at least one positive entry."

    d,n=a_mat.shape
    temp1=(a_mat.T).dot(np.log(xa))
    n2_mat=n_mat[:,d:]
    temp2=np.linalg.inv(n2_mat).dot(np.log(k))
    x=np.exp(temp1 + np.pad(temp2,(d,0),mode='constant',constant_values=(0,0)))
    return self.logder_num(x,a_mat)

  def logder_num_activity(self,b_vec,x,ld_mat):
    """given a logder matrix, compute the logder of b^T x.

    Parameters
    ----------
    b_vec : numpy vector
      Vector indicating which species are included in the catalytic activity.
      All entries are non-negative, with at least one nonzero.
    x : numpy vector
      x-vector indicating the point at which the logder is evaluated.
    ld_mat : numpy array
      n x n matrix for dlogx/dlog(t,k) at point x on the manifold.

    Returns
    -------
    ld_activity: numpy vector
       vector for dlog(b^T x)/dlog(t,k)
    """
    assert np.all(b_vec>=0), "all entries of b_vec should be non-negative."
    assert np.sum(b_vec)>0, "there should be at least one nonzero entry in b_vec."
    assert np.all(x>0), "all entries of x should be positive"
    bx=b_vec*x
    coeff=bx/np.sum(bx)
    ld_activity=coeff.dot(ld_mat)
    return ld_activity

  def x2tk_num(self,logx):
    """compute the (logt,logk) value given logx

    Parameters
    ----------
    logx : numpy vector
      Vector numerical value for total variables that define the x point.

    Returns
    -------
    logt,logk: numpy vector
      The numerical value of x at this point. log is base 10.
    """
    logt=np.log10(self.l_mat.dot(10**logx))
    logk=self.n_mat.dot(logx)
    return logt,logk

  def xak2x_num(self,logxa,logk):
    """compute the logx value given (logxa,logk)

    Parameters
    ----------
    logxa : numpy vector
      Vector numerical value for atomic species concentration. log is base 10
    logk : numpy vector
      Vector numerical value for binding reaction constants. log is base 10

    Returns
    -------
    logx: numpy vector
      The numerical value of logx at this point.
    """
    try: xak2x_map=self.xak2x_map
    except AttributeError:
      self.calc_xak2x_map() # if doesn't exist, calculate it
      xak2x_map=self.xak2x_map
    return xak2x_map.dot(np.concatenate((logxa,logk))) #because the chart is in log

  def calc_xak2x_map(self):
    # the matrix (linear map) that takes log(xa,k) to log(x)
    d=self.dim_d
    r=self.dim_r
    l2_mat=self.l_mat[:,d:]
    n2_mat=self.n_mat[:,d:]
    upper=np.concatenate((np.eye(d),np.zeros((d,r))),axis=1)
    lower=np.concatenate((l2_mat.T,np.linalg.inv(n2_mat)),axis=1)
    temp=np.concatenate((upper,lower),axis=0)
    self.xak2x_map=temp

  def tk2x_num(self,t,k,a_mat=np.array([])):
    """compute the x value by numerical integration along the equilibrium manifold using log derivatives.
    The point on the manifold defined by (t,k) is the same as that defined by x.

    Parameters
    ----------
    t : numpy vector
      Vector numerical value for total variables that define the x point. p_x = p_(t,k). log is base 10.
    k : numpy vector
      Vector numerical value for binding constants that define the x to be calculated. log is base 10.
    a_mat: numpy array, optional
      The matrix defining the total variables t=a_mat*x that the log derivatives are taken with respect to.
      Defaults to self.l_mat.

    Returns
    -------
    x: numpy vector
      The numerical value of x at this point. log is base 10.
    """
    if not np.any(a_mat): # no a_mat argument is given
      a_mat=self.l_mat

    # the initial point is always x=1, (t,k) = (A*1, 1)
    # or, in log, logx=0, (logt,logk) = (log(A*1),0)
    logt=np.log10(t)
    logk=np.log10(k)

    logt0=np.log10(np.sum(a_mat,axis=1))
    logk0=np.zeros(self.dim_r)
    y0=np.concatenate((logt0,logk0),axis=0)
    y1=np.concatenate((logt,logk),axis=0)
    logx0=np.zeros(self.dim_n)
    # The time is pseudo time, parameterizing trajectory from y0 to y1,
    # where y0=(logt0,logk0) = (log(A*1),0), and y1=(logt,logk) the input.
    # So a point on the trajectory is gamma(tau) = tau*(y1-y0)+y0, 0<=tau<=1.
    # The time derivative is therefore
    # dlogx/dtau (x0) = dlogx/dlog(t,k) (x0) * dlog(t,k)/dtau (x0)
    #                 = dlogx/dlog(t,k) (x0) * (y1-y0)
    # dlogx/dlog(t,k) (x0) is log derivative matrix evaluated at x0.
    time_derivative_func=lambda tau,logx: self.logder_num(10**logx,a_mat=a_mat).dot(y1-y0)
    sol=solve_ivp(time_derivative_func, [0, 1], logx0)
    logx=sol.y[:,-1]
    x=10**logx
    return x

# BELOW ARE VERTEX RELATED METHODS

  def vertex_construct(self):
    # Construct a dictionary of reachable vertices.
    # because l_mat tends to be sparse, we iterate through its rows to get nonzero indices,
    # then each vertex's dominance condition a_mat is choosen from the nonezro indices.

    print('Constructing vertex objects and test for feasibility...')
    d,n=self.l_mat.shape
    orientation=self.orientation
    nnz_list=[]
    for i in range(d):
      nnz_list=nnz_list+[np.nonzero(self.l_mat[i,:])[0]]
    vertex_inf_dict={}
    vertex_fin_dict={}
    for perm in itertools.product(*nnz_list):
      vertex=rop_vertex(perm,self)
      if vertex.orientation==0:
        # this is possibly an infinite vertex
        # check for rank =1, meaning perm has exactly one repeat
        perm_count=[perm.count(perm[i]) for i in range(d)]
        if max(perm_count)==2:
          if perm_count.count(2)==2: # because that same value shows up twice, each with count 2.
            # after all that check, there still can be infinite vertices that are not reachable
            # so we do feasibility test.
            is_feasible=vertex.vertex_feasibility_test(chart='x',opt_constraints=[])
            # and then add to vertex_inf_dict.
            if is_feasible:
              vertex_inf_dict[perm]=vertex
      elif vertex.orientation==orientation:
        # this is a finite vertex with the right orientation
        # we test for feasibility
        is_feasible=vertex.vertex_feasibility_test(chart='x',opt_constraints=[])
        # and then add to vertex_fin_dict.
        if is_feasible:
          vertex_fin_dict[perm]=vertex
    vertex_dict={**vertex_fin_dict,**vertex_inf_dict}
    self.vertex_dict={'all':vertex_dict,'finite':vertex_fin_dict,'infinite':vertex_inf_dict}

    print('Finished vertex construction, now computing neighbors of vertices...')
    for perm,vv in vertex_dict.items():
      vv.vertex_find_neighbors()

    print('Finished neighbors, now computing log derivatives...')
    # first compute log der for finite vertices, since infinite ones rely on
    # finite neighbors to find orientation.
    for perm,vv in vertex_dict.items():
      vv.vertex_ld_calc()

    print('Finished log derivatives, now computing c_mat_xak')
    # compute c_mat_xak for each vertex in preparation for feasibility tests.
    for perm,vv in vertex_dict.items():
      vv.vertex_c_mat_xak_calc()
    print('Done.')

  def vertex_constrained_construct(self,opt_constraints,chart='xak'):
    # for the given opt_constraints, test for each vertex whether it is feasible
    # under opt_constraints, and create a vertex dictionary for feasible vertices
    # under opt_constraints, stored in self.vertex_constrained_dict.
    # Also update each vertex's neighbors under constraints,
    # stored in each vertex.neighbors_constrained_dict.

    print('Compute feasible vertices...')
    for perm,vv in self.vertex_dict['all'].items():
      is_feasible=vv.vertex_feasibility_test(chart=chart,opt_constraints=opt_constraints)
      vv.is_feasible=is_feasible
    vertex_feasible_all={perm:vv for perm,vv in self.vertex_dict['all'].items() if vv.is_feasible}
    vertex_feasible_fin={perm:vv for perm,vv in self.vertex_dict['finite'].items() if vv.is_feasible}
    vertex_feasible_inf={perm:vv for perm,vv in self.vertex_dict['infinite'].items() if vv.is_feasible}
    self.vertex_constrained_dict={'all':vertex_feasible_all,'finite':vertex_feasible_fin,'infinite':vertex_feasible_inf}

    print('Compute neighbors under opt_constraints...')
    for perm,vv in self.vertex_dict['all'].items():
      if vv.is_feasible:
        # if it is feasible, we want to look at its neighbors.
        # if a neighbor is feasible, it is still a neighbor under constraint.
        # if a neighbor is infeasible, we look at its neighbors to see whether
        # they are feasible. This recurses.
        vv.vertex_update_constrained_neighbors()
    print('Done.')

  def vertex_list_feasibility_test(self,opt_constraints,chart='xak'):
    # for the given opt_constraints, test each of the vertex whether it is feasible
    is_feasible_fin={}
    is_feasible_inf={}
    for perm,vv in self.vertex_dict['finite'].items():
      is_feasible=vv.vertex_feasibility_test(chart=chart,opt_constraints=opt_constraints)
      is_feasible_fin[perm]=is_feasible
    for perm,vv in self.vertex_dict['infinite'].items():
      is_feasible=vv.vertex_feasibility_test(chart=chart,opt_constraints=opt_constraints)
      is_feasible_inf[perm]=is_feasible

    is_feasible_all={**is_feasible_fin,**is_feasible_inf}
    is_feasible_dict={'all':is_feasible_all,'finite':is_feasible_fin,'infinite':is_feasible_inf}
    return is_feasible_dict

  def activity_regime_construct(self,b_vec):
    # given b_vec, go through all vertices and their possible regimes
    # test for basic feasibility, and return a dictionary of regimes
    # {(perm,row):regime}; perm is vertex, row is dominant species in activity (b_vec)
    # regime is an rop_dom_regime object.
    b_vec=np.array(b_vec)
    print('Computing feasible regimes...')
    nnz_b=np.where(b_vec>0)[0]
    regime_fin_dict={}
    regime_inf_dict={}
    for perm,vv in self.vertex_dict['all'].items():
      for j in nnz_b:
        regime=rop_dom_regime(j,b_vec,perm,self)
        # if vv.orientation == 0 and np.all(regime.ld==0): #this is an infinite vertex in direction not relevant for this activity
        #   continue
        is_feasible=regime.feasibility_test(chart='x',opt_constraints=[])
        if is_feasible:
          if vv.orientation == 0:
            regime_inf_dict[(perm,j)]=regime
          else: regime_fin_dict[(perm,j)]=regime

    regime_all_dict={**regime_fin_dict,**regime_inf_dict}
    self.activity_regime_dict[tuple(b_vec)]={'finite':regime_fin_dict,
                                            'infinite':regime_inf_dict,
                                            'all':regime_all_dict}

    print('Feasible regime computed, computing their neighboring regimes...')
    for key,regime in regime_all_dict.items():
      regime.find_neighbors()
    for key,regime in regime_all_dict.items():
      regime.find_neighbors_zero()

    print('Computing activity logder regimes dictionary... ')
    # since several regimes will have the same log derivative, we use log derivative as key, rather than regimes.
    ld_regime_fin_key_dict={}
    ld_regime_inf_key_dict={}
    for key,regime in regime_fin_dict.items():
      try: ld_regime_fin_key_dict[tuple(regime.ld)]+=[key] #add regime to ld
      except KeyError: ld_regime_fin_key_dict[tuple(regime.ld)]=[key] #no regimes yet, initiate.
    for key,regime in regime_inf_dict.items():
      try: ld_regime_inf_key_dict[tuple(regime.ld)]+=[key] #add regime to ld
      except KeyError: ld_regime_inf_key_dict[tuple(regime.ld)]=[key] #no regimes yet, initiate.
    # convert into dictionary of ld_regime objects
    ld_regime_fin_dict={}
    ld_regime_inf_dict={}
    is_ray=False
    for ld,regime_keys in ld_regime_fin_key_dict.items():
      ld_regime=rop_ld_regime(ld,is_ray,b_vec,regime_keys,self)
      ld_regime_fin_dict[ld]=ld_regime
    is_ray=True
    for ld,regime_keys in ld_regime_inf_key_dict.items():
      ld_regime=rop_ld_regime(ld,is_ray,b_vec,regime_keys,self)
      ld_regime_inf_dict[ld]=ld_regime

    ld_regime_all_dict={**ld_regime_fin_dict,**ld_regime_inf_dict}
    self.activity_ld_regime_dict[tuple(b_vec)]={'finite':ld_regime_fin_dict,
                                               'infinite':ld_regime_inf_dict,
                                               'all':ld_regime_all_dict}

    print('Computing activity logder regimes neighbors... ')
    for key,ld_regime in ld_regime_all_dict.items():
      ld_regime.find_neighbors()

    print('Done')

  def activity_regime_constrained_construct(self,b_vec,opt_constraints,chart='xak'):
    # For the given opt_constraints, test for each dom_regime whether it is
    # feasible, update their neighbors under constraint,
    # and create a dictionary for feasible dom_regimes, stored in
    # self.activity_regime_contrained_dict[b_vec].
    # and create a dictionary for feasible ld_regimes, stored in
    # self.activity_ld_regime_constrained_dict[b_vec].
    print("Compute dominance regimes' feasibility...")
    regime_constrained_fin={}
    regime_constrained_inf={}
    for key,regime in self.activity_regime_dict[tuple(b_vec)]['all'].items():
      is_feasible=regime.feasibility_test(chart=chart,opt_constraints=opt_constraints)
      regime.is_feasible=is_feasible
      if is_feasible:
        if regime.vertex.orientation==0:
          regime_constrained_inf[key]=regime
        else:
          regime_constrained_fin[key]=regime
    regime_constrained_all={**regime_constrained_fin,**regime_constrained_inf}
    # regime_constrained_all={key:regime for key,regime in self.activity_regime_dict[tuple(b_vec)]['all'].items() if regime.is_feasible}
    # regime_constrained_fin={key:regime for key,regime in self.activity_regime_dict[tuple(b_vec)]['finite'].items() if regime.is_feasible}
    # regime_constrained_inf={key:regime for key,regime in self.activity_regime_dict[tuple(b_vec)]['infinite'].items() if regime.is_feasible}

    self.activity_regime_constrained_dict[tuple(b_vec)]={'all':regime_constrained_all,'finite':regime_constrained_fin,'infinite':regime_constrained_inf}

    print("Compute regimes' neighboring regimes under opt_constraints...")
    for key,regime in self.activity_regime_dict[tuple(b_vec)]['all'].items():
      if regime.is_feasible:
        # if it is feasible, we want to look at its neighbors.
        # if a neighbor is feasible, it is still a neighbor.
        # if a neighbor is infeasible, we look at its neighbors to see whether
        # they are feasible.
        regime.update_constrained_neighbors()

    print("Compute ld regimes' feasibility and compute neighbors under opt_constraints...")
    ld_regime_constrained_fin={}
    ld_regime_constrained_inf={}
    for ld,ld_regime in self.activity_ld_regime_dict[tuple(b_vec)]['all'].items():
      ld_regime.update_feasibility()
      if ld_regime.is_feasible:
        ld_regime.update_constrained_neighbors()
        if ld_regime.is_ray:
          ld_regime_constrained_inf[ld]=ld_regime
        else:
          ld_regime_constrained_fin[ld]=ld_regime
    ld_regime_constrained_all={**ld_regime_constrained_fin,**ld_regime_constrained_inf}
    self.activity_ld_regime_constrained_dict[tuple(b_vec)]={
        'all':ld_regime_constrained_all,
        'finite':ld_regime_constrained_fin,
        'infinite':ld_regime_constrained_inf
        }

    print('Done')

  def activity_list_feasibility_test(self,b_vec,opt_constraints,chart='xak'):
    # for the activity defined by b_vec (b_vec*x), and constraints given in
    # opt_constraints in chart (default xak), test each of the ld_regime
    # whether it is feasible.
    # A ld_regime is feasible if one of its dom_regime is feasible.
    is_feasible_fin={}
    is_feasible_inf={}
    for ld,ld_regime in self.activity_ld_regime_dict[tuple(b_vec)]['finite'].items():
      is_feasible_dom_regime_list=[dom_regime.feasibility_test(chart=chart,opt_constraints=opt_constraints) for dom_regime in ld_regime.dom_regime_dict.values()]
      is_feasible=np.any(is_feasible_dom_regime_list)
      is_feasible_fin[ld]=is_feasible
    for ld,ld_regime in self.activity_ld_regime_dict[tuple(b_vec)]['infinite'].items():
      is_feasible_dom_regime_list=[dom_regime.feasibility_test(chart=chart,opt_constraints=opt_constraints) for dom_regime in ld_regime.dom_regime_dict.values()]
      is_feasible=np.any(is_feasible_dom_regime_list)
      is_feasible_inf[ld]=is_feasible
    is_feasible_all={**is_feasible_fin,**is_feasible_inf}
    is_feasible_dict={'all':is_feasible_all,'finite':is_feasible_fin,'infinite':is_feasible_inf}
    return is_feasible_dict

# BELOW ARE SYMBOLIC METHODS for the binding network

  def logder_sym(self,a_mat=np.array([]),is_saved=True):
    """calculate the symbolic log derivative matrix of dlog(x)/dlog(a_mat*x,k).

    Parameters
    ----------
    a_mat : numpy array, optional
      Matrix defining the variables log derivative is taken in terms of.
      Assumes all entries are non-negative, and each row has at least one positive entry.
      Optional, defaults to l_mat of the binding network.
    is_saved: 'bool', optional
      To save the resulting logdermat as parameter ld_sym of the binding network or not.
      Defaults to True.
    Returns
    -------
    logdermat: sympy symbolic matrix
       Symbolic log derivative matrix for x to (t_sym_temp, k_sym) expressed in terms of x.

    Note that if the provided a_mat yields a noninvertible matrix, then the function will
    return logdermat as a matrix of zeros (with the appropriate size).
    """
    if not np.any(a_mat): # no a_mat argument is given
      a_mat=self.l_mat
    else:
      assert a_mat.shape==(self.dim_d,self.dim_n), f"the shape of L matrix should be {self.dim_d} by {self.dim_n}."
      assert np.all(a_mat>=0), "all entries of A matrix should be non-negative."
      assert np.all(a_mat.dot(np.ones(self.dim_n))>0), "each row of A matrix should have at least one positive entry."

    a_mat_sym=sp.Matrix(a_mat)
    n_mat_sym=self.n_mat_sym

    #it's much faster (10x) to use a dummy "total" variables to calculate matrix inverse,
    # and then substitute the expression for the totals in terms of x, rather than
    # inverting with the totals explicitly expressed in x variables.

    temp_sym = sp.symbols("t:"+str(self.dim_d))

    Lam_t = sp.Matrix.diag(*temp_sym)
    Lam_x = sp.Matrix.diag(*self.x_sym)

    topmat = Lam_t**-1 * a_mat_sym * Lam_x
    fullmat = topmat.col_join(n_mat_sym)

    logdermat_t = fullmat**-1

    temp_expr=a_mat_sym*sp.Matrix(self.x_sym)
    temp2x_subs_list=[(temp_sym[i],temp_expr[i]) for i in range(self.dim_d)]
    logdermat=sp.simplify(logdermat_t.subs(temp2x_subs_list))

    if is_saved: self.ld_sym_dict[a_mat.tobytes()]=logdermat
    return logdermat

  def subs_list_t2x(self):
    total_expr=self.l_mat_sym*sp.Matrix(self.x_sym)
    t2x_subs_list=[(self.t_sym[i],total_expr[i]) for i in range(self.dim_d)]
    return t2x_subs_list

  def subs_list_xc2xak(self):
    assert self.is_atomic, "this operation requires the binding network to be atomic"
    l2_mat_sym=self.l_mat_sym[:,self.dim_d:]
    n2_mat_sym=self.n_mat_sym[:,self.dim_d:]
    xa_sym_vec=sp.Matrix(self.x_sym[:self.dim_d])
    xa_sym_log=xa_sym_vec.applyfunc(sp.log)
    xc2xak_log_expr=l2_mat_sym.T*xa_sym_log+n2_mat_sym.inv()*sp.Matrix(self.k_sym).applyfunc(sp.log)
    func_exponentiate=lambda x:sp.E**x
    xc2xak_expr=xc2xak_log_expr.applyfunc(func_exponentiate)
    xc2xak_subs_list=[(self.x_sym[self.dim_d+i],xc2xak_expr[i]) for i in range(self.dim_r)]
    return xc2xak_subs_list

  def t2x_sym(self,expr):
    """input a symbolic expression containing totals t, map it to x

    Parameters
    ----------
    expr : sympy symbolic expression
       A symbolic expression to be converted

    Returns
    -------
    expr_x: sympy symbolic expression
       The symbolic expression after conversion.
    """
    # calculate the substitutions map for totals to species
    try: expr_x=expr.subs(self.t2x)
    except AttributeError:
      t2x_subs_list=self.subs_list_t2x()
      self.t2x=t2x_subs_list
      expr_x=expr.subs(self.t2x)
    return expr_x

  def xc2xak_sym(self,expr):
    """input a symbolic expression containing complex species x^c, map it to x^a and k.

    Parameters
    ----------
    expr : sympy symbolic expression
       A symbolic expression to be converted

    Returns
    -------
    expr_xak: sympy symbolic expression
       The symbolic expression after conversion.
    """
    assert self.is_atomic, "this operation requires the binding network to be atomic"
    # calculate the substitutions map for complex species to atomic species and k's, if it does not exist.
    try: expr_xak=expr.subs(self.xc2xak)
    except AttributeError:
      xc2xak_subs_list=self.subs_list_xc2xak()
      self.xc2xak=xc2xak_subs_list
      expr_xak=expr.subs(self.xc2xak)
    return expr_xak

  def logder_sym_activity(self,b_vec_sym,a_mat=np.array([])):
    """compute the log derivative of a linear sum of x, i.e. dlog(b_vec_sym*x)/dlog(tp,k), tp=a_mat*x

    Parameters
    ----------
    b_vec_sym : sympy symbolic vector, shape n-by-1
      A vector of symbolic expressions corresponding to coefficients to be summed
    a_mat: numpy array, optional
      The matrix denoting the variables a_mat*x that the log derivative is taken with respect to.
      Defaults to self.l_mat.

    Returns
    -------
    ld_sym_sum: sympy vector of symbolic expression
       The symbolic log derivative  dlog(b_vec_sym*x)/dlog(tp,k), tp=a_mat*x.
    """
    if not np.any(a_mat): # no a_mat argument is given
      a_mat=self.l_mat
    # check if logder is already calculated with respect to this a_mat.
    try: ld_sym=self.ld_sym_dict[a_mat.tobytes()]
    except KeyError: # not calculated yet
      print('Symbolic logder for this chart coordinate does not exist, calculating...')
      ld_sym=self.logder_sym(a_mat=a_mat)
    coeff_vec_sym=sp.Matrix(np.zeros(b_vec_sym.shape))
    bx = sp.matrix_multiply_elementwise(b_vec_sym,sp.Matrix(self.x_sym)) #b multiply x element-wise
    bx_sum = bx.T*sp.ones(*bx.shape)
    coeff_vec_sym=bx*bx_sum**(-1) # convex coefficients for the log derivative sum
    ld_sym_sum=(ld_sym.T*coeff_vec_sym).T # for logder matrix ld_sym, rows are species, columns are variables.
    return ld_sym_sum