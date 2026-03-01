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
    libopengl0 \
    libosmesa6 \
    libegl-mesa0 \
    libgl1-mesa-dri \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Create working directory ──────────────────────────────────────────────────
WORKDIR /app

# ── Copy requirements first (layer caching) ──────────────────────────────────
COPY requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt

# ── Copy project ──────────────────────────────────────────────────────────────
COPY . .

# ── Download MuJoCo Menagerie (Panda arm) ─────────────────────────────────────
RUN git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git \
    && cd mujoco_menagerie \
    && git sparse-checkout set franka_emika_panda \
    && rm -rf .git

# ── MuJoCo environment ────────────────────────────────────────────────────────
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Limit Threads to save memory on Render Free Tier ──────────────────────────
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "SemSorter.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
