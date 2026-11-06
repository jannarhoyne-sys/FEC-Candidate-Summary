# Small official Python image (same version as the app)
FROM python:3.12-slim

WORKDIR /app

# Install packages - Docker can reuse this step when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app, the model and its data
COPY . .

# Heroku sets PORT itself - locally, EC2 and SageMaker use 8080
ENV PORT=8080
EXPOSE 8080

# Start the app with gunicorn. ENTRYPOINT (not CMD) so SageMaker's "serve" argument can't replace it
#Original Code failed - rewrote to use gunicorn with threads and timeout
ENTRYPOINT ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 2 --threads 4 --timeout 120 app:app"]