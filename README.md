# NeuroSeg AI: Brain Tumor Segmentation

NeuroSeg AI is an end-to-end deep learning web application designed for brain tumor detection and segmentation in MRI scans. The system features a modular FastAPI backend, an interactive React frontend built with Vite and Tailwind CSS, and is fully containerized with Docker.

Additionally, this repository includes the core research Jupyter Notebook containing the Enhanced U-Net model architecture training logic and validation results.

---

## 🌟 Key Features

*   **Deep Learning Segmentation**: Employs an Enhanced U-Net model optimized for brain MRI scans.
*   **Interactive Web UI**: Elegant, premium dark-themed dashboard allowing users to drag and drop or upload MRI scans.
*   **Pre-loaded Sample Scans**: Quick buttons to try out sample brain scans instantly.
*   **Detailed Analytics**: Calculates and displays essential performance metrics including **Dice Coefficient**, **Accuracy**, and **IoU (Intersection-over-Union)**.
*   **History Logs**: Displays a history of all analyzed scans with dates and corresponding metrics, fetched dynamically from an SQLite database.
*   **Fully Dockerized**: Easily run the entire stack locally with Docker Compose.

---

## 🛠️ Tech Stack

### Backend
*   **FastAPI**: Modern, high-performance web framework for building APIs.
*   **SQLite**: Lightweight SQL database to store segmentation history.
*   **Uvicorn**: High-performance ASGI web server.

### Frontend
*   **React (TypeScript)**: Interactive components and layout.
*   **Vite**: Next-generation frontend build tool.
*   **Tailwind CSS & shadcn/ui**: Modern, responsive UI design.
*   **Lucide Icons**: Crisp and premium icon assets.

### Containers
*   **Docker & Docker Compose**: Unified build and container deployment configuration.

---

## 📁 Repository Structure

```text
├── Brain-Tumor-Segmentation/
│   ├── app/
│   │   ├── backend/
│   │   │   ├── core/               # Database and configuration logic
│   │   │   ├── dependencies/       # Dependency injection (Auth, DB)
│   │   │   ├── routers/            # API endpoints (segmentation, entities)
│   │   │   ├── services/           # DB queries and processing business logic
│   │   │   ├── Dockerfile          # Backend container setup
│   │   │   ├── main.py             # FastAPI entrypoint
│   │   │   └── requirements.txt    # Python dependencies
│   │   ├── frontend/
│   │   │   ├── src/                # React components, pages, and hooks
│   │   │   ├── public/             # Sample images and public assets
│   │   │   ├── Dockerfile          # Frontend container setup
│   │   │   ├── package.json        # Node dependencies
│   │   │   └── vite.config.ts      # Vite build settings
│   │   └── docker-compose.yml      # Orchestration for both frontend & backend
│   ├── brain-tumor-segmentation-unet-dice-coef-89-6.ipynb  # U-Net training & research notebook
│   └── README.md                   # Project documentation
```

---

## 🚀 Getting Started

Ensure you have [Docker](https://www.docker.com/) and [Git](https://git-scm.com/) installed on your machine.

### Method 1: Running with Docker Compose (Recommended)

To build and run the entire application (both Backend and Frontend) inside containers, run:

1. Clone this repository to your local machine.
2. In the project root (where `docker-compose.yml` is located), execute:
   ```bash
   docker-compose up --build
   ```
3. Once running, open your browser and navigate to:
   *   **Frontend**: `http://localhost:3000`
   *   **Backend API**: `http://localhost:8000`

---

### Method 2: Running Locally (Manual Setup)

If you prefer to run the components directly on your host machine:

#### 1. Backend Setup
1. Navigate to the backend folder:
   ```bash
   cd app/backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the FastAPI server:
   ```bash
   python main.py
   ```
   *The backend will boot up at `http://localhost:8000`.*

#### 2. Frontend Setup
1. Navigate to the frontend folder:
   ```bash
   cd app/frontend
   ```
2. Install npm packages:
   ```bash
   pnpm install  # or npm install
   ```
3. Start the development server:
   ```bash
   pnpm run dev  # or npm run dev
   ```
   *The frontend dev server will start at `http://localhost:3000`.*

---

## 🧠 Model & Notebook Research

The root of this repository contains the research file:
`brain-tumor-segmentation-unet-dice-coef-89-6.ipynb`

This notebook covers:
1. **Data Preprocessing**: Handling MRI scans and normalization.
2. **Model Definition**: Enhanced U-Net architecture with multi-scale skip connections.
3. **Training & Optimization**: Custom loss functions integrating Binary Cross-Entropy (BCE) and Dice Loss.
4. **Validation**: Reaching an **89.6% Dice Coefficient** validation score on the dataset.
