"""
System endpoints (health, info, etc.)
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "QuOptuna Next API is running",
    }


@router.get("/info")
async def system_info():
    """Get system information"""
    return {
        "version": "2.0.0",
        "quantum_models": 18,
        "classical_models": 8,
        "total_models": 26,
    }


@router.get("/models")
async def list_available_models():
    """List all available models"""
    return {
        "quantum": [
            "Data Reuploading",
            "Circuit Centric",
            "Dressed Quantum Circuit",
            "Quantum Kitchen Sinks",
            "IQP Variational",
            "IQP Kernel",
            "Projected Quantum Kernel",
            "Quantum Metric Learning",
            "Vanilla QNN",
            "Quantum Boltzmann Machine",
            "Tree Tensor Network",
            "WeiNet",
            "Quanvolutional Neural Network",
            "Separable",
            "Convolutional Neural Network",
        ],
        "classical": [
            "Support Vector Classifier",
            "Multi-layer Perceptron",
            "Perceptron",
            "Random Forest",
            "Gradient Boosting",
            "AdaBoost",
            "Logistic Regression",
            "Decision Tree",
        ],
    }
