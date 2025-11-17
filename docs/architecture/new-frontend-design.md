# QuOptuna Next: Modern Full-Stack Architecture

## Overview
A modern, intuitive full-stack application for quantum machine learning optimization with drag-and-drop workflow building, inspired by langflow's architecture.

## Tech Stack

### Frontend
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite (fast HMR, optimized builds)
- **UI Library**: shadcn/ui + Tailwind CSS
- **Drag & Drop**: React Flow (visual workflow builder)
- **State Management**: Zustand (lightweight, modern)
- **Charts**: Recharts + Plotly.js
- **API Client**: TanStack Query (React Query)
- **Form Handling**: React Hook Form + Zod validation
- **File Upload**: react-dropzone

### Backend
- **Framework**: FastAPI (async, high-performance)
- **API Docs**: Auto-generated OpenAPI/Swagger
- **WebSockets**: FastAPI WebSocket for real-time updates
- **Task Queue**: Celery + Redis (for long-running optimizations)
- **Database**: SQLite (Optuna) + PostgreSQL (metadata)
- **CORS**: FastAPI middleware

### Infrastructure
- **Monorepo**: pnpm workspace (frontend) + Python packages (backend)
- **Containerization**: Docker + docker-compose
- **Dev Server**: Vite dev server + uvicorn --reload
- **Type Safety**: TypeScript (frontend) + Pydantic (backend)

## Architecture Design

### 1. Visual Workflow Builder (Drag & Drop)

Users can visually build ML pipelines by dragging and connecting nodes:

**Node Types:**
1. **Data Nodes**
   - Upload CSV
   - Fetch from UCI
   - Data Preview
   - Feature Selection

2. **Preprocessing Nodes**
   - Train/Test Split
   - StandardScaler
   - Label Encoding
   - Feature Engineering

3. **Model Nodes**
   - Quantum Models (18 types)
   - Classical Models (8 types)
   - Ensemble Methods

4. **Optimization Nodes**
   - Optuna Study Config
   - Hyperparameter Ranges
   - Objective Function
   - Run Optimization

5. **Analysis Nodes**
   - SHAP Analysis
   - Confusion Matrix
   - ROC Curve
   - Feature Importance

6. **Output Nodes**
   - Export Model
   - Generate Report
   - Save Results

**Workflow Example:**
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

### 2. Page Structure

#### Dashboard (/)
- Recent workflows
- Quick actions
- System status
- Performance metrics

#### Workflow Builder (/workflow)
- Canvas with React Flow
- Node palette (left sidebar)
- Property panel (right sidebar)
- Toolbar (top): Save, Run, Export, Share

#### Data Explorer (/data)
- Uploaded datasets
- UCI repository browser
- Data preview with statistics
- Feature correlation heatmap

#### Experiments (/experiments)
- List of optimization runs
- Filter by status, model type, dataset
- Comparison view (side-by-side)
- Export results

#### Models (/models)
- Saved models library
- Model card with metadata
- Performance metrics
- Download/deploy options

#### Analytics (/analytics)
- SHAP visualizations
- Interactive plots
- AI-powered insights
- Report builder

#### Settings (/settings)
- API keys (OpenAI, Anthropic, Google)
- Database configuration
- Compute preferences
- Theme customization

### 3. Key Features

#### 🎨 Modern UI/UX
- **Dark/Light Mode**: System preference or manual toggle
- **Responsive Design**: Mobile, tablet, desktop optimized
- **Keyboard Shortcuts**: Power user workflows
- **Drag & Drop**: Intuitive workflow building
- **Live Preview**: Real-time data/result updates
- **Toast Notifications**: Success, error, info messages

#### ⚡ Performance
- **Code Splitting**: Lazy load routes and components
- **Virtual Scrolling**: Handle large datasets
- **Debounced Search**: Optimized filtering
- **WebSocket Updates**: Real-time optimization progress
- **Caching**: React Query for smart data caching

#### 🔒 Type Safety
- **End-to-End Types**: TypeScript ↔ Pydantic
- **Auto-generated API Client**: From OpenAPI spec
- **Runtime Validation**: Zod schemas
- **Type Guards**: Safer data handling

## API Design

### REST Endpoints

```typescript
// Data Management
POST   /api/v1/data/upload           // Upload CSV
GET    /api/v1/data/uci              // List UCI datasets
GET    /api/v1/data/uci/{id}         // Fetch specific dataset
GET    /api/v1/data/{id}             // Get dataset info
DELETE /api/v1/data/{id}             // Delete dataset

// Workflows
POST   /api/v1/workflows             // Create workflow
GET    /api/v1/workflows             // List workflows
GET    /api/v1/workflows/{id}        // Get workflow
PUT    /api/v1/workflows/{id}        // Update workflow
DELETE /api/v1/workflows/{id}        // Delete workflow
POST   /api/v1/workflows/{id}/run    // Execute workflow

// Optimization
POST   /api/v1/optimize              // Start optimization
GET    /api/v1/optimize/{id}         // Get optimization status
GET    /api/v1/optimize/{id}/trials  // Get trial history
DELETE /api/v1/optimize/{id}         // Cancel optimization

// Models
GET    /api/v1/models                // List available models
GET    /api/v1/models/{type}         // Get model info
POST   /api/v1/models/save           // Save trained model
GET    /api/v1/models/saved          // List saved models

// Analysis
POST   /api/v1/analysis/shap         // Generate SHAP analysis
POST   /api/v1/analysis/report       // Generate AI report
GET    /api/v1/analysis/{id}         // Get analysis results

// System
GET    /api/v1/health                // Health check
GET    /api/v1/info                  // System info
```

### WebSocket Endpoints

```typescript
WS /ws/optimize/{optimization_id}    // Real-time optimization updates
WS /ws/workflow/{workflow_id}        // Workflow execution status
```

### WebSocket Message Format

```typescript
interface OptimizationUpdate {
  type: 'trial_start' | 'trial_complete' | 'study_complete' | 'error';
  data: {
    trial_number: number;
    params: Record<string, any>;
    value: number;
    state: 'running' | 'complete' | 'pruned' | 'failed';
    timestamp: string;
  };
}
```

## Directory Structure

```
quoptuna/
├── frontend/                    # React frontend
│   ├── src/
│   │   ├── components/         # Reusable components
│   │   │   ├── ui/            # shadcn/ui components
│   │   │   ├── workflow/      # React Flow nodes
│   │   │   ├── charts/        # Visualization components
│   │   │   └── layout/        # Layout components
│   │   ├── pages/             # Route pages
│   │   │   ├── Dashboard.tsx
│   │   │   ├── WorkflowBuilder.tsx
│   │   │   ├── DataExplorer.tsx
│   │   │   ├── Experiments.tsx
│   │   │   ├── Models.tsx
│   │   │   ├── Analytics.tsx
│   │   │   └── Settings.tsx
│   │   ├── lib/               # Utilities
│   │   │   ├── api.ts         # API client
│   │   │   ├── websocket.ts   # WebSocket client
│   │   │   └── utils.ts       # Helper functions
│   │   ├── stores/            # Zustand stores
│   │   │   ├── workflow.ts
│   │   │   ├── data.ts
│   │   │   └── settings.ts
│   │   ├── types/             # TypeScript types
│   │   │   ├── api.ts
│   │   │   ├── workflow.ts
│   │   │   └── models.ts
│   │   ├── hooks/             # Custom hooks
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── public/
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
│
├── backend/                     # FastAPI backend
│   ├── app/
│   │   ├── api/               # API routes
│   │   │   ├── v1/
│   │   │   │   ├── data.py
│   │   │   │   ├── workflows.py
│   │   │   │   ├── optimize.py
│   │   │   │   ├── models.py
│   │   │   │   ├── analysis.py
│   │   │   │   └── system.py
│   │   │   └── deps.py        # Dependencies
│   │   ├── core/              # Core functionality
│   │   │   ├── config.py
│   │   │   ├── security.py
│   │   │   └── websocket.py
│   │   ├── services/          # Business logic
│   │   │   ├── data_service.py
│   │   │   ├── optimization_service.py
│   │   │   ├── analysis_service.py
│   │   │   └── workflow_service.py
│   │   ├── schemas/           # Pydantic schemas
│   │   │   ├── data.py
│   │   │   ├── workflow.py
│   │   │   ├── optimization.py
│   │   │   └── analysis.py
│   │   ├── models/            # Database models
│   │   ├── tasks/             # Celery tasks
│   │   └── main.py            # FastAPI app
│   └── pyproject.toml
│
├── docker-compose.yml
└── README_NEXT.md
```

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [x] Architecture design
- [ ] Setup Vite + React + TypeScript
- [ ] Setup FastAPI backend structure
- [ ] Basic routing (React Router)
- [ ] API client setup (React Query)
- [ ] Docker configuration

### Phase 2: Core Features (Week 2)
- [ ] Data upload/preview UI
- [ ] UCI dataset browser
- [ ] FastAPI data endpoints
- [ ] Basic workflow canvas (React Flow)
- [ ] Node palette and types

### Phase 3: Workflow Builder (Week 3)
- [ ] Complete node library (all 6 types)
- [ ] Node connection validation
- [ ] Property panel (node configuration)
- [ ] Workflow save/load
- [ ] Workflow execution engine

### Phase 4: Optimization (Week 4)
- [ ] Optuna integration
- [ ] WebSocket real-time updates
- [ ] Optimization dashboard
- [ ] Trial visualization
- [ ] Parameter importance plots

### Phase 5: Analysis & Reporting (Week 5)
- [ ] SHAP integration
- [ ] Interactive plots (Plotly.js)
- [ ] AI report generation (LangChain)
- [ ] Export functionality
- [ ] Model comparison tools

### Phase 6: Polish & Deploy (Week 6)
- [ ] Dark/light mode
- [ ] Keyboard shortcuts
- [ ] Performance optimization
- [ ] Error handling
- [ ] User documentation
- [ ] Deployment guide

## Benefits Over Streamlit

| Feature | Streamlit | QuOptuna Next |
|---------|-----------|---------------|
| **UX** | Linear page navigation | Drag-and-drop visual workflow |
| **Interactivity** | Limited | Fully interactive React app |
| **Real-time** | Polling/rerun | WebSocket live updates |
| **Customization** | Limited theming | Full design system control |
| **Performance** | Page reloads | SPA with smart caching |
| **Type Safety** | Python only | End-to-end TypeScript + Python |
| **Workflow** | Manual steps | Reusable visual workflows |
| **Collaboration** | Single session | Multi-user (future) |
| **API** | None | Full REST API + WebSocket |
| **Mobile** | Poor | Responsive design |

## Development Commands

```bash
# Frontend development
cd frontend
pnpm install
pnpm dev              # http://localhost:5173

# Backend development
cd backend
uv sync
uv run uvicorn app.main:app --reload  # http://localhost:8000

# Full stack (Docker)
docker-compose up     # Frontend + Backend + Redis + PostgreSQL

# Type generation (OpenAPI → TypeScript)
pnpm generate:api

# Build for production
pnpm build            # Frontend
uv build              # Backend
```

## Next Steps

1. **Get approval** on architecture design
2. **Initialize projects** (Vite + FastAPI)
3. **Setup development environment** (Docker compose)
4. **Implement Phase 1** (Foundation)
5. **Iterative development** following phases
6. **User testing** and feedback
7. **Production deployment**

---

**This design provides:**
- ✨ Modern, intuitive UI/UX
- 🎯 Drag-and-drop workflow building
- ⚡ Real-time updates
- 🔒 Type-safe development
- 📦 Modular, scalable architecture
- 🚀 Production-ready stack
