# Running QuOptuna Without Docker

## Backend Setup

### 1. Create Virtual Environment
```bash
cd /home/user/quoptuna
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Main QuOptuna Package
```bash
# Install the main quoptuna package in editable mode
pip install -e .
```

### 3. Install Backend Dependencies
```bash
cd backend
pip install -e .
```

### 4. Run Backend Server
```bash
# From the backend directory
cd /home/user/quoptuna/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Backend will be available at:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

## Frontend Setup

### 1. Install Node Dependencies
```bash
cd /home/user/quoptuna/frontend
npm install
```

### 2. Run Frontend Dev Server
```bash
npm run dev
```

**Frontend will be available at:**
- UI: http://localhost:5173

## Quick Start Script

Create a file `start-dev.sh`:

```bash
#!/bin/bash

# Start backend in background
echo "Starting backend..."
cd /home/user/quoptuna
source venv/bin/activate
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Start frontend in background
echo "Starting frontend..."
cd /home/user/quoptuna/frontend
npm run dev &
FRONTEND_PID=$!

echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Backend running at: http://localhost:8000"
echo "Frontend running at: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
```

Make it executable:
```bash
chmod +x start-dev.sh
./start-dev.sh
```

## Troubleshooting

### Backend Issues

**Import errors:**
```bash
# Make sure both packages are installed
pip install -e .  # From /home/user/quoptuna
pip install -e .  # From /home/user/quoptuna/backend
```

**Port already in use:**
```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### Frontend Issues

**Module not found:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Port already in use:**
```bash
# Find and kill process on port 5173
lsof -ti:5173 | xargs kill -9
```

## Environment Variables

### Backend (.env in /home/user/quoptuna/backend)
```env
DATABASE_URL=sqlite:///./quoptuna.db
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
UPLOAD_DIR=./uploads
```

### Frontend (.env in /home/user/quoptuna/frontend)
```env
VITE_API_URL=http://localhost:8000
```
