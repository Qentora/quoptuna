# QuOptuna Next 🚀

A modern, full-stack application for quantum machine learning optimization with **drag-and-drop workflow building**, inspired by langflow's architecture.

## ✨ Features

### 🎨 Modern UI/UX
- **Drag & Drop Workflow Builder** - Visually design ML pipelines with React Flow
- **Dark/Light Mode** - Beautiful design system with Tailwind CSS
- **Responsive Design** - Works seamlessly on desktop, tablet, and mobile
- **Real-time Updates** - Live optimization progress via WebSocket
- **Interactive Dashboards** - Rich data visualizations with Recharts and Plotly

### 🧠 Powerful ML Capabilities
- **26 ML Models** - 18 quantum models + 8 classical models
- **Hyperparameter Optimization** - Powered by Optuna
- **Explainability Analysis** - SHAP integration for model insights
- **AI-Powered Reports** - Generate insights with GPT-4, Claude, or Gemini

### ⚡ Modern Tech Stack

#### Frontend
- React 18 + TypeScript
- Vite (lightning-fast HMR)
- React Flow (drag & drop)
- Tailwind CSS + shadcn/ui
- Zustand (state management)
- TanStack Query (data fetching)

#### Backend
- FastAPI (async, high-performance)
- Pydantic (type safety)
- Optuna (optimization)
- PennyLane (quantum ML)
- SHAP (explainability)
- LangChain (LLM integration)

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18+ and npm
- **Python** 3.10+
- **Docker** (optional, recommended)

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/Qentora/quoptuna.git
cd quoptuna

# Start all services
docker-compose up

# Access the application
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/api/docs
```

### Option 2: Local Development

#### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev

# Open http://localhost:5173
```

#### Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Start server
uvicorn app.main:app --reload

# Open http://localhost:8000/api/docs
```

## 📖 Usage Guide

### 1. Create a Workflow

Navigate to **Workflow Builder** and design your ML pipeline:

```
[Upload CSV] → [Feature Selection] → [Train/Test Split] → [StandardScaler]
                                                               ↓
                                                          [Quantum Model]
                                                               ↓
                                                       [Optuna Optimization]
                                                               ↓
                                                          [SHAP Analysis]
                                                               ↓
                                                        [Generate Report]
```

**Drag nodes from the palette** → **Connect them** → **Configure each node** → **Run!**

### 2. Upload Data

Go to **Data Explorer** to:
- Upload CSV files
- Browse UCI ML Repository
- Preview dataset statistics
- Manage your datasets

### 3. Run Optimization

Click **Run** on your workflow to:
- Start Optuna hyperparameter tuning
- Monitor progress in real-time
- View trial results
- Explore optimization history

### 4. Analyze Results

Navigate to **Analytics** to:
- Generate SHAP visualizations
- Create AI-powered reports
- Compare model performance
- Export results

## 🗂️ Project Structure

```
quoptuna/
├── frontend/                    # React frontend
│   ├── src/
│   │   ├── components/         # Reusable components
│   │   │   ├── ui/            # shadcn/ui components
│   │   │   ├── workflow/      # Workflow builder nodes
│   │   │   ├── charts/        # Visualization components
│   │   │   └── layout/        # Layout components
│   │   ├── pages/             # Route pages
│   │   │   ├── Dashboard.tsx
│   │   │   ├── WorkflowBuilder.tsx
│   │   │   ├── DataExplorer.tsx
│   │   │   ├── Models.tsx
│   │   │   ├── Analytics.tsx
│   │   │   └── Settings.tsx
│   │   ├── stores/            # Zustand stores
│   │   ├── types/             # TypeScript types
│   │   └── lib/               # Utilities & API client
│   ├── package.json
│   └── vite.config.ts
│
├── backend/                     # FastAPI backend
│   ├── app/
│   │   ├── api/v1/            # API endpoints
│   │   │   ├── data.py
│   │   │   ├── workflows.py
│   │   │   ├── optimize.py
│   │   │   ├── analysis.py
│   │   │   └── system.py
│   │   ├── core/              # Core configuration
│   │   ├── services/          # Business logic
│   │   ├── schemas/           # Pydantic models
│   │   └── main.py            # FastAPI app
│   └── pyproject.toml
│
├── docker-compose.yml
├── NEW_FRONTEND_DESIGN.md      # Architecture design doc
└── README_NEXT.md              # This file
```

## 🎯 Available Node Types

### Data Nodes
- **Upload CSV** - Import local datasets
- **UCI Dataset** - Fetch from repository
- **Data Preview** - View statistics
- **Feature Selection** - Choose X and y

### Preprocessing Nodes
- **Train/Test Split** - Split data (default 75/25)
- **Standard Scaler** - Normalize features
- **Label Encoding** - Encode target labels

### Model Nodes
- **Quantum Models** - 18 PennyLane-based models
- **Classical Models** - 8 Scikit-learn models

### Optimization Nodes
- **Optuna Config** - Configure study parameters
- **Run Optimization** - Execute hyperparameter tuning

### Analysis Nodes
- **SHAP Analysis** - Generate explainability plots
- **Confusion Matrix** - Classification metrics
- **Feature Importance** - Rank features

### Output Nodes
- **Export Model** - Save trained models
- **Generate Report** - AI-powered insights

## 📊 Quantum Models (18)

1. Data Reuploading
2. Circuit Centric
3. Dressed Quantum Circuit
4. Quantum Kitchen Sinks
5. IQP Variational
6. IQP Kernel
7. Projected Quantum Kernel
8. Quantum Metric Learning
9. Vanilla QNN
10. Quantum Boltzmann Machine
11. Tree Tensor Network
12. WeiNet
13. Quanvolutional Neural Network
14. Separable
15. Convolutional Neural Network
16. (and more...)

## 🖥️ Classical Models (8)

1. Support Vector Classifier
2. Multi-layer Perceptron
3. Perceptron
4. Random Forest
5. Gradient Boosting
6. AdaBoost
7. Logistic Regression
8. Decision Tree

## 🔌 API Documentation

Access the interactive API docs at:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

### Key Endpoints

```
POST   /api/v1/data/upload           # Upload CSV
GET    /api/v1/data/uci              # List UCI datasets
POST   /api/v1/workflows             # Create workflow
POST   /api/v1/workflows/{id}/run    # Execute workflow
POST   /api/v1/optimize              # Start optimization
GET    /api/v1/optimize/{id}/trials  # Get trial history
POST   /api/v1/analysis/shap         # Generate SHAP analysis
POST   /api/v1/analysis/report       # Generate AI report
GET    /api/v1/health                # Health check
```

## 🛠️ Development

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Start dev server with HMR
npm run dev

# Type check
npm run type-check

# Build for production
npm run build

# Preview production build
npm run preview
```

### Backend Development

```bash
cd backend

# Install with dev dependencies
pip install -e ".[dev]"

# Run with auto-reload
uvicorn app.main:app --reload

# Run tests
pytest

# Format code
ruff format .

# Lint code
ruff check .
```

## 🐳 Docker Commands

```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild images
docker-compose build

# Remove volumes
docker-compose down -v
```

## 🌟 Key Differences from Streamlit Version

| Feature | Streamlit | QuOptuna Next |
|---------|-----------|---------------|
| **UX** | Linear page navigation | Drag-and-drop visual workflows |
| **Interactivity** | Limited, page reloads | Fully interactive React SPA |
| **Real-time** | Polling/rerun | WebSocket live updates |
| **Customization** | Basic theming | Full design system control |
| **Performance** | Server-side rendering | Client-side with smart caching |
| **Type Safety** | Python only | End-to-end TypeScript + Pydantic |
| **API** | None | Full REST API + WebSocket |
| **Workflows** | Manual multi-page steps | Reusable visual workflows |
| **Mobile** | Poor responsiveness | Fully responsive design |
| **State** | Session-based | Persistent with Zustand |

## 📝 Configuration

### Environment Variables

Create `.env` files in frontend and backend directories:

**Backend `.env`:**
```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
DATABASE_URL=sqlite:///./quoptuna.db
UPLOAD_DIR=./uploads
```

**Frontend `.env`:**
```env
VITE_API_URL=http://localhost:8000
```

## 🎨 Customization

### Add Custom Nodes

1. Define node type in `frontend/src/types/workflow.ts`
2. Add to palette in `frontend/src/components/workflow/NodePalette.tsx`
3. Implement backend logic in `backend/app/services/`

### Add Custom Models

1. Create model class in `src/quoptuna/backend/models.py`
2. Register in model factory
3. Add to frontend model list

## 🔧 Troubleshooting

### Frontend not connecting to backend?
- Check CORS settings in `backend/app/core/config.py`
- Verify `VITE_API_URL` in frontend `.env`
- Ensure backend is running on correct port

### Workflow execution failing?
- Check backend logs: `docker-compose logs backend`
- Verify dataset is uploaded correctly
- Ensure all nodes are connected properly

### Build errors?
- Clear node_modules: `rm -rf node_modules && npm install`
- Clear Python cache: `find . -type d -name __pycache__ -delete`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Langflow** - Inspiration for drag-and-drop architecture
- **React Flow** - Excellent workflow builder library
- **shadcn/ui** - Beautiful component library
- **FastAPI** - Modern Python web framework
- **Optuna** - Hyperparameter optimization
- **PennyLane** - Quantum machine learning

## 📧 Support

- **GitHub Issues**: https://github.com/Qentora/quoptuna/issues
- **Documentation**: See `NEW_FRONTEND_DESIGN.md` for architecture details
- **API Docs**: http://localhost:8000/api/docs

---

**Built with ❤️ by the QuOptuna Team**

*Quantum Machine Learning Made Visual*
