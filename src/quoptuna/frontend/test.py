# from a dict get a model from create model
from quoptuna.backend.models import create_model

model_dict = {
    "C": 10,
    "alpha": 0.0001,
    "batch_size": 32,
    "degree": 2,
    "encoding_layers": 1,
    "eta0": 0.1,
    "filter_name": "sharpen",
    "gamma": 0.01,
    "gamma_factor": 0.1,
    "hidden_layer_sizes": "(10, 10, 10, 10)",
    "kernel_shape": 2,
    "learning_rate": 0.01,
    "max_vmap": 1,
    "model_type": "DressedQuantumCircuitClassifierSeparable",
    "n_episodes": 500,
    "n_input_copies": 1,
    "n_layers": 1,
    "n_qchannels": 10,
    "n_qfeatures": "full",
    "observable_type": "half",
    "qkernel_shape": 2,
    "repeats": 5,
    "t": 1,
    "temperature": 100,
    "trotter_steps": 1,
    "visible_qubits": "full",
}
model_type = model_dict.pop("model_type")
model = create_model(model_type=model_type, **model_dict)
