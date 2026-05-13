
from al_graph_research.models.altered_laplace import AlteredLaplace
from al_graph_research.models.altered_poisson import AlteredPoisson

class Model:

    def __init__(self, model_type, W, class_priors=None, **kwargs):
        """
        Wrapper class for constructing and fitting graph-based semi-supervised
        learning models.

        This class provides a lightweight interface for creating either a signed
        Laplace-learning model or a signed Poisson-learning model using a shared
        graph and shared configuration arguments.

        The model type is selected through the `model_type` argument and the
        corresponding model object is constructed lazily through `build()`.

        Parameters
        ----------
        model_type : {"laplace", "poisson"}
            Type of graph-based SSL model to construct.

            - "laplace" constructs an `AlteredLaplace` model.
            - "poisson" constructs an `AlteredPoisson` model.

        W : numpy array, scipy sparse matrix, or graphlearning graph object
            Graph adjacency or weight matrix.

        class_priors : numpy array, optional
            Optional class priors passed to the underlying SSL model.

        **kwargs
            Additional keyword arguments passed directly to the selected model
            constructor.

            Examples include:
            - normalization
            - tol
            - order
            - mean_shift
            - reweighting

        Notes
        -----
        This wrapper is primarily intended to simplify experimentation by allowing
        Laplace and Poisson models to share a common interface.

        Examples
        --------
        Construct and fit a signed Laplace model:

        >>> model = Model(
        ...     model_type="laplace",
        ...     W=W,
        ...     normalization="normalized"
        ... )
        >>> scores = model.fit(train_indices, train_labels)

        Construct and fit a signed Poisson model:

        >>> model = Model(
        ...     model_type="poisson",
        ...     W=W,
        ...     normalization="combinatorial"
        ... )
        >>> scores = model.fit(train_indices, train_labels)
        """
        
        if model_type not in {"laplace", "poisson"}:
            raise ValueError("model_type must be 'laplace' or 'poisson'.")
        else:
            self.model_type = model_type # "laplace" or "poisson"
        self.W = W
        self.class_priors = class_priors
        self.kwargs = kwargs

    def build(self):
        """
        Construct the selected graph SSL model.

        Returns
        -------
        AlteredLaplace or AlteredPoisson
            Instantiated graph SSL model corresponding to `self.model_type`.

        Raises
        ------
        ValueError
            If `self.model_type` is not recognized.
        """
        if self.model_type == "laplace":
            return AlteredLaplace(self.W, class_priors=self.class_priors, **self.kwargs)
        if self.model_type == "poisson":
            return AlteredPoisson(self.W, class_priors=self.class_priors, **self.kwargs)
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, train_indices, train_labels):
        """
        Fit the selected graph SSL model and return prediction scores.

        Parameters
        ----------
        train_indices : array-like
            Indices of labeled training nodes.

        train_labels : array-like
            Labels corresponding to the training indices.

        Returns
        -------
        numpy.ndarray
            Model score vector over all graph nodes.
        """
        model = self.build()
        return model._fit(train_indices, train_labels)