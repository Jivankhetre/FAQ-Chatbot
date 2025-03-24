# Use an official lightweight Python image.
FROM python:3.11.4

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/gemini_cred.json"

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Run the application
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]