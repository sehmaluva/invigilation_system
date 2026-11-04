FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# Use `python -m pip` so packages land in the same interpreter that starts
# the server. `&&` stops the build if either install fails.
# CPU PyTorch comes from the PyTorch index; the rest comes from requirements.txt.
RUN python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && python -m pip install --no-cache-dir -r requirements.txt
COPY . .
# This is the same import uvicorn runs at startup. If a library such as
# sqlalchemy is missing, the build fails here instead of the container crashing.
RUN python -c "import app.main"
EXPOSE 8000
# So the startup model/threshold log shows up immediately in `docker logs`.
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
