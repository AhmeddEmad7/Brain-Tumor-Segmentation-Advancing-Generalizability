FROM hazem11/pytorch24_monai14 AS base

# Set environment variables
ENV DCMQI_VERSION=1.3.2
ENV DCMQI_DEB=dcmqi-${DCMQI_VERSION}-ubuntu-20.04-amd64.deb

# Install dependencies and DCMQI
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://github.com/QIICR/dcmqi/releases/download/v${DCMQI_VERSION}/${DCMQI_DEB} && \
    apt-get install -y ./${DCMQI_DEB} && \
    rm -f ${DCMQI_DEB} && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Verify installation
RUN itkimage2segimage --help

# Set working directory
WORKDIR /usr/src/app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Command to run the application
CMD ["python", "main.py"]