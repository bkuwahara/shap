import networkx as nx
import numpy as np
from scipy.linalg import lu_factor, lu_solve

def nan_cov(mat):
    """
    Compute the covariance matrix of the given matrix, ignoring NaNs.
    """
    # Get the number of variables
    n_vars = mat.shape[1]
    
    # Initialize covariance matrix
    cov_matrix = np.empty((n_vars, n_vars))
    
    # Iterate over each pair of variables to calculate covariance
    for i in range(n_vars):
        for j in range(i):
            # Extract the valid data (non-NaN) for the variable pair
            valid_data_i = mat[:, i][~np.isnan(mat[:, i])]
            valid_data_j = mat[:, j][~np.isnan(mat[:, j])]
            
            # Find common valid indices
            valid_mask = ~np.isnan(mat[:, i]) & ~np.isnan(mat[:, j])
            valid_data_i = mat[:, i][valid_mask]
            valid_data_j = mat[:, j][valid_mask]
            
            # Compute covariance for the pair
            if len(valid_data_i) > 1 and len(valid_data_j) > 1:
                cov_matrix[i, j] = np.cov(valid_data_i, valid_data_j, ddof=1)[0, 1]
                cov_matrix[j, i] = np.cov(valid_data_i, valid_data_j, ddof=1)[0, 1]
            else:
                cov_matrix[i, j] = np.nan  # Not enough data to compute covariance
                cov_matrix[j, i] = np.nan
    for i in range(n_vars):
        cov_matrix[i, i] = np.nanvar(mat[:, i], ddof=1)

    return cov_matrix



class ChainComponent:
    """Class representing a component in a causal chain graph

    Parameters
    ----------
    features : np.ndarray[bool]
        Set of features in the component
    confounding : bool (default=False)
        Whether the features in the component are confounded by unobserved variables
    """
    def from_set(features, M, confounding=False, name=None, sample_method = "gaussian"):
        S = np.array([1 if f in features else 0 for f in range(M)]).astype(bool)
        return ChainComponent(S, confounding=confounding, name=name)


    def from_feature_name_list(names_to_include, names_list, confounding=False, group_name=None, sample_method = "gaussian"):
        features = np.zeros(len(names_list)).astype(bool)
        for i in range(len(names_list)):
            for name_to_include in names_to_include:
                if name_to_include in names_list[i]:
                    features[i] = 1
        return ChainComponent(features, confounding=confounding, name=group_name)
    

    def __init__(self, features, confounding=False, name=None, sample_method = "gaussian"):
        self.features = features
        self.confounding = confounding
        self.name = name
        if self.name is None:
            self.name = (i for i in range(len(features)) if features[i])
        self.sample_method = sample_method

    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other):
        return (self.features == other.features).all()
    
    def __str__(self) -> str:
        return str(self.name)
    

    

class CausalChainGraph:
    """Class representing a causal chain graph

    Parameters
    ----------
    components : list of ChainComponent
        List of components in the graph
    edges : list of tuples
        List of directed edges in the graph, where an edge from 
        component x to component y indicates that x causally precedes y
    dataset : np.array
        Background dataset to represent distribution used to generate interventional samples
    """

    
    def __init__(self, components, edges, dataset):       

        self.dataset = dataset
        self._mean = np.nanmean(dataset, axis=0)
        self._cov = np.cov(dataset, rowvar=False)
        self.components = components

        self._check_validity()
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(components)
        self.graph.add_edges_from(edges)
        self._confounding_components = [c for c in components if c.confounding]
        self._non_confounding_components = [c for c in components if not c.confounding]

        self._vars = np.ones(dataset.shape[1]).astype(bool)

        self.order, self._parents = self._generate_traversal_order()
        self._isolated_features = np.zeros_like(self._mean).astype(bool)
        for component in self.components:
            if self._is_isolated(component):
                self._isolated_features = np.logical_or(self._isolated_features, component.features)


    def _is_isolated(self, component):
        return self.graph.in_degree(component) == 0 and (component.confounding or \
                sum(component.features) == 1)


    def _check_validity(self):
        covered = np.zeros_like(self._mean).astype(bool)
        for component in self.components:
            if (covered * (component.features)).any():
                raise ValueError("Components must be disjoint")
            covered = np.logical_or(covered, component.features)
            

    def _generate_traversal_order(self):
        order = []
        predecessors = {}
        for node in nx.topological_sort(self.graph):
            parent_set = np.zeros_like(self._mean).astype(bool)
            for parent in self.graph.predecessors(node):
                parent_set = np.logical_or(parent_set, parent.features)
            predecessors[node] = parent_set
            order.append(node)
        return order, predecessors


    def interventional_distribution(self, S, x, nsamples=1, in_place=False, const_features=[]):
        """Draw samples from the interventional distribution P(X_S' | do(X_S=x_S))
        Mutates the input x to reflect the intervened values

        Parameters
        ----------
        S : np.ndarray[bool]
            Set of features to intervene on
        x_fixed : np.array
            Values of the intervened features
        nsamples : int (default=1)
            Number of samples to draw. If in_place, uses x.shape[0] as the number of samples
        in_place : bool (default=False)
            Whether to mutate the input x or return a new array
        const_features : list (default=[])
            List of features known to be constant and therefore not to be sampled or conditioned on
        """

        # Features which never vary and have no causal interactions with other features can 
        # be samples directly from the background dataset and should not be conditioned on
        # if const_features == []:
        #     const_features = [0 for _ in range(len(self._mean))]
        # self._background_features = (np.array(const_features) * np.array(self._isolated_features)).astype(bool) 

        if not in_place:
            x = np.repeat(x.reshape(1, -1), nsamples, axis=0)
        else:
            nsamples = x.shape[0]

        Sbar = np.logical_xor(self._vars, S)

        # Traverse nodes in topological (causal) order
        for i in range(nsamples):
            for node in self.order:
                T = node.features * Sbar
                # If the node's features are all in S, the distribution is pre-specified by the intervention
                if not T.any():
                    continue

                parents = self._parents[node]
                condition_on = parents 
                if not node.confounding:
                    condition_on = condition_on + (node.features * S) 

                # If the node is isolated, there is nothing to condition on. Sample from the background distribution
                if not condition_on.any():
                    x[i:i+1,Sbar*node.features] = self.dataset[np.random.choice(self.dataset.shape[0], 1), :][:, Sbar*node.features]
                else:
                    # Sample from the distribution and update x with the sampled features   
                    if node.sample_method == "gaussian":     
                        x[i:i+1,Sbar*node.features] = self.gaussian_conditional_distribution(condition_on, T, x[i:i+1,:])
                    elif node.sample_method == "max":
                        x[i:i+1,Sbar*node.features] = self.max_condition_distribution(condition_on, T, x[i:i+1,:])
                    elif node.sample_method == "min":
                        x[i:i+1,Sbar*node.features] = self.min_condition_distribution(condition_on, T, x[i:i+1,:])
                    else:
                        raise ValueError("Invalid sample method")
        # Sample isolated features from the background distribution
        # Constant features are already set to their appropriate value
        # x[:, self._background_features] = self.dataset[np.random.choice(self.dataset.shape[0], nsamples), :][:, self._background_features]

        return x



    def gaussian_conditional_distribution(self, S, T, x):
        """Generates the parameters of the conditional distribution P(X_T | X_S=x_S)

        Parameters
        ----------
        S : np.ndarray[bool]
            Set of conditioning features
        T : np.ndarray[bool]
            Set of features conditioned on
        x : np.array
            Values of the features conditioned on
        """
        x_S = x[:,S]
        mu_S = self._mean[S]
        mu_T = self._mean[T]
        cov_SS = self._cov[S][:, S]
        cov_TS = self._cov[T][:, S]
        cov_TT = self._cov[T][:, T]

        lu, piv = lu_factor(cov_SS)
        cov_SS_inv_cov_TS_T = lu_solve((lu, piv), cov_TS.T) # cov_SS^-1 cov_TS^T
        cov_SS_inv_mean_diff = lu_solve((lu, piv), (x_S - mu_S).T) # cov_SS^-1 (x_S - mu_S)


        mu_T_given_S = mu_T + (cov_TS @ cov_SS_inv_mean_diff).reshape(-1)
        cov_T_given_S = cov_TT - cov_TS @ cov_SS_inv_cov_TS_T


        return np.random.multivariate_normal(mu_T_given_S, cov_T_given_S)
        

    def max_condition_distribution(self, S, T, x):
        """
        Sample features in T conditioned from rows in dataset where S is at least x_S
        """
        valid_samples = np.where(self.dataset[:, S] >= x[S])
        return valid_samples[np.random.choice(valid_samples.shape[0]), T]
    

    def min_condition_distribution(self, S, T, x):
        """
        Sample features in T conditioned from rows in dataset where S is at most x_S
        """
        valid_samples = np.where(self.dataset[:, S] <= x[S])
        return valid_samples[np.random.choice(valid_samples.shape[0]), T]



    def save(self, filename):
        """Save the graph to a file

        Parameters
        ----------
        filename : str
            Name of the file to save the graph to
        """
        nx.write_gpickle(self.graph, filename)