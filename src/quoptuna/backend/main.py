from quoptuna.backend.models.pennylane_models import CircuitCentricClassifier


def main():
    model = CircuitCentricClassifier()
    import numpy as np

    X = np.random.rand(100, 10)
    # y is a binary classification problem with 2 classes 1 and -1
    y = np.random.choice([1, -1], size=100)
    model.fit(X, y)
    print(model.predict(X))
    print(model)


if __name__ == "__main__":
    main()
    # test the model with a sample  test data
