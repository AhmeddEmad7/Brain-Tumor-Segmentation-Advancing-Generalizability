# Use the official Ubuntu 20.04 LTS image as the base image
FROM ubuntu:20.04

# Update the package repository and install required packages
RUN apt-get update && \
    apt-get install -y wget nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install Orthanc directly
RUN wget https://orthanc.uclouvain.be/downloads/linux-standard-base/orthanc/1.12.1/Orthanc --output-document /usr/sbin/Orthanc && \
    chmod +x /usr/sbin/Orthanc && \
    mkdir -p /usr/share/orthanc/plugins

# Download the plugins
RUN wget https://orthanc.uclouvain.be/downloads/linux-standard-base/orthanc/1.12.1/libServeFolders.so --output-document /usr/share/orthanc/plugins/libServeFolders.so && \
    wget https://orthanc.uclouvain.be/downloads/linux-standard-base/orthanc/1.12.1/libModalityWorklists.so --output-document /usr/share/orthanc/plugins/libModalityWorklists.so && \
    wget https://orthanc.uclouvain.be/downloads/linux-standard-base/orthanc-dicomweb/1.15/libOrthancDicomWeb.so --output-document /usr/share/orthanc/plugins/libOrthancDicomWeb.so && \
    wget https://orthanc.uclouvain.be/downloads/linux-standard-base/orthanc-neuro/1.0/libOrthancNeuro.so --output-document /usr/share/orthanc/plugins/libOrthancNeuro.so && \
    wget https://orthanc.uclouvain.be/downloads/linux-standard-base/orthanc-webviewer/2.8/libOrthancWebViewer.so --output-document /usr/share/orthanc/plugins/libOrthancWebViewer.so && \
    wget https://orthanc.uclouvain.be/downloads/linux-standard-base/orthanc-postgresql/5.1/libOrthancPostgreSQLIndex.so --output-document /usr/share/orthanc/plugins/libOrthancPostgreSQLIndex.so && \
    wget https://orthanc.uclouvain.be/downloads/linux-standard-base/orthanc-postgresql/5.1/libOrthancPostgreSQLStorage.so --output-document /usr/share/orthanc/plugins/libOrthancPostgreSQLStorage.so

# Copy your Orthanc configuration file into the container
COPY ./config/orthanc.json /etc/orthanc/orthanc.json

# Expose the default Orthanc port (4242) if needed
EXPOSE 4242

# Expose the Https server
EXPOSE 8042

## Start Orthanc
CMD ["/usr/sbin/Orthanc", "/etc/orthanc/orthanc.json"]
