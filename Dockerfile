# If using alpine - check the other separate docker file

# set base image (host OS) using debian
FROM python:3.7-slim
RUN apt-get update -y && apt-get install -y python-pip python-dev build-essential && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# copy the dependencies file to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# set the working directory in the container
# Copy and create only after pip has setup the structure tree
RUN mkdir /app
WORKDIR /app

#Copy all files
COPY . /app

# copy the content of the local src directory to the working directory
COPY . .

# command to run on container start
CMD ["python","./test_offline_analysis_methods.py"]
