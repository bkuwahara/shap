import networkx as nx
import numpy as np


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



class FeatureSet:
    """Simple implementation of a set using Boolean numpy arrays

    Parameters
    ----------
    features : np.array
        Boolean array representing the set of features present in the set
    """

    def __init__(self, features):
        self.features = features.astype(bool)

    
    def intersection(self, other):
        return FeatureSet(self.features * other.features)


    def union(self, other):
        return FeatureSet(np.bitwise_or(self.features, other.features))
    

    def __sub__(self, other):
        return FeatureSet(np.bitwise_xor(self.features, other.features))
    
    def any(self):
        return self.features.any()
    
    def from_set(features, M):
        return FeatureSet(np.array([1 if f in features else 0 for f in range(M)]))
    
    def __str__(self) -> str:
        return str(self.features)
    
    def __len__(self) -> int:
        return len(self.features)


class ChainComponent:
    """Class representing a component in a causal chain graph

    Parameters
    ----------
    features : np.ndarray[bool]
        Set of features in the component
    confounding : bool (default=False)
        Whether the features in the component are confounded by unobserved variables
    """
    def from_set(features, M, confounding=False, name=None):
        S = np.array([1 if f in features else 0 for f in range(M)]).astype(bool)
        return ChainComponent(S, confounding=confounding, name=name)

    def __init__(self, features, confounding=False, name=None):
        self.features = features
        self.confounding = confounding
        self.name = name
        if self.name is None:
            self.name = (i for i in range(len(features)) if features[i])

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
        self._isolated_components = [c for c in self.components if self.graph.in_degree(c) == 0 and self.graph.out_degree(c) == 0]


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


    def interventional_distribution(self, S, x, nsamples=1, in_place=False):
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
        """

        if not in_place:
            x = np.repeat(x.reshape(1, -1), nsamples, axis=0)
        else:
            nsamples = x.shape[0]

        T = np.logical_xor(self._vars, S)

        # Traverse nodes in topological (causal) order
        for i in range(nsamples):
            for node in self.order:
                dist = (self._confounding_distribution(node, self._parents[node], S, x[i:i+1,:]) if node.confounding 
                        else self._non_confounding_distribution(node, self._parents[node], S, x[i:i+1,:]))
                
                # Check if the distribution is undefined (i.e. the node's features are all in S, so are predefined)
                if dist is None:
                    continue
                mu, Sigma = dist

                # Sample from the distribution and update x with the sampled features
                
                x[i:i+1,T*node.features] = np.random.multivariate_normal(mu, Sigma)
        return x



    def _confounding_distribution(self, node, parents, S, x):
        """Generate the parameters of the multivariate normal distribution
        P(X_{node intersect S'} | X_{parents intersect S}, X_{parents intersect S'})
        where node is part of a confounded component

        Parameters
        ----------
        node : ChainComponent
            Node in the causal chain graph
        parents : np.ndarray[bool]
            Set of parents of the node
        S : np.ndarray[bool]
            Set of intervened features
        x : np.array    
            Values of the features conditioned on
        """
        Sbar = np.logical_xor(self._vars, S)
        T = node.features * Sbar

        # If the node's features are all in S, the distribution is pre-specified by the intervention
        if not T.any():
            return None

        # If the node has no parents, the distribution is simply the marginal distribution
        if not parents.any():
            mu = self._mean[T.features]
            Sigma = self._cov[T.features][:, T.features]
            return mu, Sigma
        
        mu, Sigma = self.conditional_distribution(parents, T, x)
        return mu, Sigma


    def _non_confounding_distribution(self, node, parents, S, x):
        """Generate the parameters of the multivariate normal distribution
        P(X_{node intersect S'} | X_{parents intersect S}, X_{parents intersect S'}, x_{node intersect S})
        where node is part of a non-confounded component

        Parameters
        ----------
        node : ChainComponent
            Node in the causal chain graph
        parents : np.ndarray[bool]
            Set of parents of the node
        S : np.ndarray[bool]
            Set of intervened features
        x : np.array    
            Values of the features conditioned on
        """
        Sbar = np.logical_xor(self._vars, S)
        T = node.features * Sbar
        if not T.any():
            return None
        
        tau_S = node.features * S

        mu, Sigma = self.conditional_distribution(np.logical_or(parents, tau_S), T, x)
        return mu, Sigma
    

    def conditional_distribution(self, S, T, x):
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

        mu_T_given_S = mu_T + (cov_TS @ np.linalg.inv(cov_SS) @ ((x_S - mu_S).reshape(-1, 1))).reshape(-1)
        cov_T_given_S = cov_TT - cov_TS @ np.linalg.inv(cov_SS) @ cov_TS.T

        return mu_T_given_S, cov_T_given_S
    
