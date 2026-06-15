from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
from sklearn.base import BaseEstimator, ClassifierMixin

jax.config.update(jax_enable_x64=True)


@dataclass
class CircuitConfig:
    n_input_copies: int = 2
    n_layers: int = 4
    convergence_interval: int = 200
    max_steps: int = 10000
    learning_rate: float = 0.001
    batch_size: int = 32
    max_vmap: int = None
    jit: bool = True
    scaling: float = 1.0
    random_state: int = 42
    dev_type: str = "default.qubit"
    qnode_kwargs: dict = field(default_factory=lambda: {"interface": "jax-jit"})


class CircuitCentricClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, config: CircuitConfig = None):
        if config is None:
            config = CircuitConfig()
        self.__dict__.update(config.__dict__)
        # attributes that do not depend on data
        self.n_input_copies = config.n_input_copies
        self.n_layers = config.n_layers
        self.convergence_interval = config.convergence_interval
        self.max_steps = config.max_steps
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.dev_type = config.dev_type
        self.qnode_kwargs = config.qnode_kwargs
        self.jit = config.jit
        self.scaling = config.scaling
        self.random_state = config.random_state
        self.rng = np.random.default_rng(config.random_state)

        if config.max_vmap is None:
            self.max_vmap = self.batch_size
        else:
            self.max_vmap = config.max_vmap

        # data-dependant attributes
        # which will be initialised by calling "fit"
        self.params_ = None  # Dictionary containing the trainable parameters
        self.n_qubits_ = None
        self.scaler = None
        self.circuit = None

    def generate_key(self) -> jax.Array:
        """
        Generates a random key used in sampling batches.

        Returns:
            jax.Array: _description_
        """
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def construct_model(self):
        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(params, x):
            # classically compute the copies of inputs
            # since AmplitudeEmbedding cannot be called
            # multiple times on jax device
            tensor_x = x
            for i in range(self.n_input_copies - 1):
                tensor_x = jnp.kron(tensor_x, x)

            qml.AmplitudeEmbedding(tensor_x, wires=range(self.n_qubits_))
            qml.StronglyEntanglingLayers(params["weights"], wires=range(self.n_qubits_))
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

        if self.jit:
            circuit = jax.jit(circuit)

        def circuit_plus_bias(params, x):
            return circuit(params, x) + params["b"]

        if self.jit:
            circuit_plus_bias = jax.jit(circuit_plus_bias)
        self.forward = jax.vmap(circuit_plus_bias, in_axes=(None, 0))
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)

        return self.forward

    def initialize(self, n_features, classes=None):
        """Initialize attributes that depend on the number of features and the class labels.
        Args:
            n_features (int): Number of features that the classifier expects
            classes (array-like): class labels
        """
        if classes is None:
            classes = [-1, 1]

        self.classes_ = classes
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ == 2
        assert 1 in self.classes_ and -1 in self.classes_

        n_qubits_per_copy = int(np.ceil(np.log2(n_features)))
        self.n_qubits_ = self.n_input_copies * n_qubits_per_copy

        self.construct_model()
        self.initialize_params()

    def initialize_params(self):
        # initialise the trainable parameters
        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits_)
        weights = jax.random.uniform(self.generate_key(), minval=0, maxval=2 * np.pi, shape=shape)
        b = jnp.array(0.01)
        self.params_ = {"weights": weights, "b": b}

    def fit(self, X, y):
        """Fit the model to data X and labels y.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
            y (np.ndarray): Labels of shape (n_samples,)
        """

        self.initialize(n_features=X.shape[1], classes=np.unique(y))

        X = self.transform(X)

        optimizer = optax.adam

        def loss_fn(params, X, y):
            pred = self.forward(params, X)  # jnp.stack([self.forward(params, x) for x in X])
            return jnp.mean(optax.l2_loss(pred, y))

        if self.jit:
            loss_fn = jax.jit(loss_fn)
        self.params_ = train(
            self,
            loss_fn,
            optimizer,
            X,
            y,
            self.generate_key,
            convergence_interval=self.convergence_interval,
        )

        return self

    def predict(self, X):
        """Predict labels for batch of data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
        """
        predictions = self.predict_proba(X)
        mapped_predictions = np.argmax(predictions, axis=1)
        return np.take(self.classes_, mapped_predictions)

    def predict_proba(self, X):
        """Predict label probabilities for data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred_proba (np.ndarray): Predicted label probabilities of shape
            (n_samples, n_classes)
        """
        X = self.transform(X)
        predictions = self.chunked_forward(self.params_, X)
        predictions_2d = np.c_[(1 - predictions) / 2, (1 + predictions) / 2]
        return predictions_2d

    def transform(self, X, preprocess=False):
        """
        The feature vectors padded to the next power of 2 and then normalised.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
        """
        n_features = X.shape[1]
        X = X * self.scaling

        n_qubits_per_copy = int(np.ceil(np.log2(n_features)))
        max_n_features = 2**n_qubits_per_copy
        n_padding = max_n_features - n_features
        padding = np.ones(shape=(len(X), n_padding)) / max_n_features

        X_padded = np.c_[X, padding]
        X_normalised = np.divide(X_padded, np.expand_dims(np.linalg.norm(X_padded, axis=1), axis=1))
        return X_normalised
