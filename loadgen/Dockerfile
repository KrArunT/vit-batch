# Use a base image with build tools
FROM aruntiwary/echoswift-auto:latest
#FROM python:3.12

# Install system dependencies
#Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Ensure dependencies are installed
RUN pip install --no-cache-dir -r requirements.txt

# Ensure entrypoint script is executable
RUN chmod +x entrypoint.sh

# Expose a port (if your application needs it)
EXPOSE 8080

# Set the default command to start the container
ENTRYPOINT ["./entrypoint.sh"]
~                              
