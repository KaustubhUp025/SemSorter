FROM python:3.10-slim

# ── System deps for MuJoCo EGL rendering ─────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    libglvnd0 \
    libglx0 \
    libx11-6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ── Create working directory ──────────────────────────────────────────────────
WORKDIR /app

# ── Copy requirements first (layer caching) ──────────────────────────────────
COPY requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt

# ── Copy project ──────────────────────────────────────────────────────────────
COPY . .

# ── MuJoCo environment ────────────────────────────────────────────────────────
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "SemSorter.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
