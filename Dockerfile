#FROM python:3.7-alpine
FROM python:3.7-slim

RUN apt-get update -y && apt-get install -y python-pip python-dev build-essential && apt-get clean
#RUN apk update && apk add --update --no-cache build-base py3-pip python3-dev python3

COPY requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy and create only after pip has setup the structure tree
RUN mkdir /app
WORKDIR /app

#Copy all files
COPY . /app

# Run
CMD ["python","./test_offline_analysis_methods.py"]
