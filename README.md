# NeuroSeg AI: Brain Tumor Segmentation

<div align="center">
  <h3>An end-to-end deep learning web application designed for brain tumor detection and segmentation in MRI scans.</h3>
</div>

---

## 🌟 Key Features

*   **Deep Learning Segmentation**: Employs an Enhanced U-Net model optimized for brain MRI scans.
*   **Interactive Web UI**: Elegant, premium dark-themed dashboard allowing users to drag and drop or upload MRI scans.
*   **Detailed Analytics**: Calculates and displays essential inference statistics including **Tumor Detection**, **Coverage Percentage**, and **Tumor Probability Mapping**.
*   **Evaluation Endpoint**: Specialized endpoint for computing true **Dice Coefficient**, **Accuracy**, and **IoU (Intersection-over-Union)** when a ground-truth mask is provided.
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

---

## 📁 Monorepo Structure

```text
├── backend/               # FastAPI Backend Service
│   ├── core/              # Database and configuration logic
│   ├── routers/           # API endpoints (segmentation, entities)
│   ├── services/          # DB queries and processing business logic
│   ├── Dockerfile         # Backend container setup
│   └── main.py            # FastAPI entrypoint
├── frontend/              # React & Vite Frontend Service
│   ├── src/               # React components, pages, and hooks
│   ├── public/            # Sample images and public assets
│   ├── Dockerfile         # Frontend container setup
│   └── package.json       # Node dependencies
├── notebooks/             # Core Research & Model Development
│   └── brain-tumor-segmentation-unet-dice-coef-89-6.ipynb
├── docs/                  # Reference Material & Papers
│   └── Automatic Biomedical Brain Tumor Segmentation...pdf
├── test_images/           # Sample scans for inference testing
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

Ensure you have [Docker](https://www.docker.com/) and [Git](https://git-scm.com/) installed on your machine.

### Method 1: Running with Docker Compose (Recommended)

1. Clone this repository to your local machine.
2. In the project root, execute:
   ```bash
   docker-compose up --build
   ```
3. Once running, open your browser and navigate to:
   *   **Frontend**: `http://localhost:3000`
   *   **Backend API**: `http://localhost:8000`

### Method 2: Running Locally (Manual Setup)

#### 1. Backend Setup
1. Navigate to the backend folder:
   ```bash
   cd backend
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
   cd frontend
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

The `notebooks/` directory contains the research file:
`brain-tumor-segmentation-unet-dice-coef-89-6.ipynb`

This notebook covers:
1. **Data Preprocessing**: Handling MRI scans and normalization.
2. **Model Definition**: Enhanced U-Net architecture with multi-scale skip connections.
3. **Training & Optimization**: Custom loss functions integrating Binary Cross-Entropy (BCE) and Dice Loss.
4. **Validation & Testing**: Computes per-sample, thresholded metrics on a held-out test set with zero patient leakage.

---

## 📊 Metric Interpretation

It is important to distinguish between **supervised evaluation** and **normal inference**:

*   **Supervised Evaluation Metrics (Dice, IoU, Accuracy, Precision, Recall)**: These metrics *require a ground-truth mask* to compare against the model's prediction. These are computed during notebook training and by the `/api/evaluate` endpoint.
*   **Normal Inference Statistics (Tumor Detected, Coverage, Mean Probability)**: When a user uploads a new MRI scan without a ground-truth mask, the system provides honest, prediction-derived statistics based on the model's sigmoid output.
