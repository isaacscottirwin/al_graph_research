
from al_graph_research.models.altered_laplace import AlteredLaplace

from al_graph_research.models.altered_poisson import AlteredPoisson

class Model:

    def __init__(self, model_type, W, class_priors=None, **kwargs):
        if model_type not in {"laplace", "poisson"}:
            raise ValueError("model_type must be 'laplace' or 'poisson'.")
        else:
            self.model_type = model_type # "laplace" or "poisson"
        self.W = W
        self.class_priors = class_priors
        self.kwargs = kwargs

    def build(self):
        if self.model_type == "laplace":
            return AlteredLaplace(self.W, class_priors=self.class_priors, **self.kwargs)
        if self.model_type == "poisson":
            return AlteredPoisson(self.W, class_priors=self.class_priors, **self.kwargs)
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, train_indices, train_labels):
        model = self.build()
        return model._fit(train_indices, train_labels)