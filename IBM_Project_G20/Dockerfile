# Use an official lightweight Python image as the base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt ./
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY .env ./
COPY movies.json ./
COPY application.py ./
COPY assets/ ./assets/
COPY templates/ ./templates/
COPY emotion_recognition_cnn_rnn_model_final.h5 ./

# Expose the application port
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "-b", "0.0.0.0:5000", "application:application"]