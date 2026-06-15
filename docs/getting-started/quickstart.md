# QuOptuna Next - Quick Start Guide 🚀

Get up and running with QuOptuna Next in under 5 minutes!

## 🎯 What You'll Get

A modern, drag-and-drop interface for building quantum ML workflows with real-time optimization tracking.

## 📋 Prerequisites

Choose one:
- **Docker** (easiest) - Just Docker and docker-compose
- **Local Development** - Node.js 18+ and Python 3.10+

## 🚀 Option 1: Docker (Recommended)

### Step 1: Start the Services

```bash
cd /path/to/quoptuna
docker-compose up
```

That's it! 🎉

### Step 2: Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/api/docs

## 💻 Option 2: Local Development

### Step 1: Start Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Start server
uvicorn app.main:app --reload
```

Backend running at http://localhost:8000 ✓

### Step 2: Start Frontend

```bash
# New terminal
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Frontend running at http://localhost:5173 ✓

## 🎨 Using the Application

### 1. Create Your First Workflow

1. Navigate to **Workflow Builder**
2. Drag nodes from the left palette onto the canvas
3. Connect nodes by dragging from output (bottom) to input (top) handles

**Example Workflow:**
```
Upload CSV → Feature Selection → Train/Test Split → Quantum Model → Optimization → SHAP Analysis
```

### 2. Configure Nodes

Click any node to configure its parameters:
- Dataset selection
- Model type
- Hyperparameter ranges
- Analysis options

### 3. Run the Workflow

Click **Run** in the toolbar to execute your pipeline!

## 📊 Example: Iris Classification

### Using the Interface:

1. **Data Explorer** → Upload `iris.csv` or fetch from UCI
2. **Workflow Builder** → Create this pipeline:
   ```
   [UCI Dataset: Iris]
      ↓
   [Feature Selection: sepal_length, sepal_width, petal_length, petal_width → species]
      ↓
   [Train/Test Split: 75/25]
      ↓
   [Standard Scaler]
      ↓
   [Quantum Model: Data Reuploading]
      ↓
   [Optuna Config: 50 trials]
      ↓
   [SHAP Analysis: bar, beeswarm, violin]
      ↓
   [Generate Report: GPT-4]
   ```
3. **Run** → Monitor real-time progress
4. **Analytics** → View SHAP insights and AI report

## 🔧 Development Commands

### Frontend

```bash
cd frontend

# Development
npm run dev          # Start dev server
npm run build        # Build for production
npm run preview      # Preview production build
npm run type-check   # TypeScript check

# Clean install
rm -rf node_modules package-lock.json && npm install
```

### Backend

```bash
cd backend

# Development
uvicorn app.main:app --reload        # Start with auto-reload
python -m pytest                      # Run tests
ruff check .                          # Lint code
ruff format .                         # Format code

# Clean environment
rm -rf .venv && python -m venv .venv
```

### Docker

```bash
# Start services
docker-compose up              # Foreground
docker-compose up -d           # Background

# View logs
docker-compose logs -f         # All services
docker-compose logs -f frontend  # Frontend only
docker-compose logs -f backend   # Backend only

# Stop and clean
docker-compose down            # Stop services
docker-compose down -v         # Stop and remove volumes
docker-compose build           # Rebuild images
```

## 🎯 Next Steps

### Explore Features

1. **Model Library** → Browse 26 available models
2. **Data Explorer** → Upload datasets or fetch from UCI
3. **Analytics** → Generate SHAP visualizations
4. **Settings** → Configure API keys for LLM reports

### Customize

- **Add Custom Nodes**: Edit `frontend/src/components/workflow/NodePalette.tsx`
- **Add Models**: Extend `src/quoptuna/backend/models.py`
- **Modify Theme**: Update `frontend/src/index.css`
- **Add Endpoints**: Create in `backend/app/api/v1/`

### Learn More

- 📖 **Full Documentation**: See `README_NEXT.md`
- 🏗️ **Architecture Guide**: See `NEW_FRONTEND_DESIGN.md`
- 📚 **API Reference**: http://localhost:8000/api/docs

## 🐛 Troubleshooting

### Port Already in Use?

```bash
# Change frontend port
cd frontend
# Edit vite.config.ts: server: { port: 3000 }

# Change backend port
cd backend
# Start with: uvicorn app.main:app --port 8001
```

### Dependencies Not Installing?

```bash
# Frontend
cd frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install

# Backend
cd backend
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### Docker Issues?

```bash
# Full clean restart
docker-compose down -v
docker system prune -a
docker-compose build --no-cache
docker-compose up
```

## 💡 Tips

1. **Save Workflows** - Click Save to reuse your pipelines
2. **Keyboard Shortcuts** -
   - `Delete` - Remove selected node
   - `Ctrl/Cmd + S` - Save workflow
   - `Ctrl/Cmd + Z` - Undo
3. **Node Status** - Watch nodes change color during execution:
   - Gray = Idle
   - Blue = Running (animated)
   - Green = Complete
   - Red = Error
4. **Use Templates** - Start with example workflows in Dashboard

## 🎓 Tutorial: Your First Quantum ML Workflow

**Goal**: Train a quantum model on the Iris dataset

**Time**: 5 minutes

1. **Upload Data**
   - Go to Data Explorer
   - Click "UCI Repository"
   - Select "Iris Dataset"

2. **Build Workflow**
   - Go to Workflow Builder
   - Drag: UCI Dataset → Feature Selection → Train/Test Split → Standard Scaler → Quantum Model → Optuna Config → SHAP Analysis
   - Connect all nodes

3. **Configure**
   - UCI Dataset: Select "iris"
   - Feature Selection: X = [sepal_length, sepal_width, petal_length, petal_width], y = species
   - Quantum Model: Type = "Data Reuploading"
   - Optuna Config: Trials = 20

4. **Execute**
   - Click Run
   - Watch real-time progress
   - View results in Analytics

5. **Analyze**
   - Go to Analytics
   - See SHAP plots
   - Generate AI report

**Congratulations!** 🎉 You've built your first quantum ML workflow!

## 📞 Get Help

- **Issues**: https://github.com/Qentora/quoptuna/issues
- **Discussions**: https://github.com/Qentora/quoptuna/discussions
- **API Docs**: http://localhost:8000/api/docs

---

Happy Quantum Machine Learning! 🌟
