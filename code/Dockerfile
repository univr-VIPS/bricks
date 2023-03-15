# Start from a core stack version, already includes libs like numpy scipy and many others
FROM jupyter/scipy-notebook:lab-3.0.15
LABEL maintainer="Filippo"
ENV JUPYTER_ENABLE_LAB=yes

#pymesh
RUN cd /tmp && \
    git clone https://github.com/PyMesh/PyMesh.git && \
    cd PyMesh &&\
    git submodule update --init && \
    pip install -r python/requirements.txt
USER root 
RUN apt-get update && \
    apt-get install -y \
    libeigen3-dev \
    libgmp-dev \
    libgmpxx4ldbl \
    libmpfr-dev \
    libboost-dev \
    libboost-thread-dev \
    libtbb-dev \
    cmake \
    build-essential
RUN cd /tmp/PyMesh && \
    ./setup.py build && \
    ./setup.py install
USER jovyan

# Dependencies - from https://jupyter-docker-stacks.readthedocs.io/en/latest/using/recipes.html#using-pip-install-or-conda-install-in-a-child-docker-image
# Install from requirements.txt file
COPY requirements.txt /tmp/
RUN conda config --add channels conda-forge # add condaforge channel
RUN conda config --set channel_priority strict # increase priority 
RUN conda install --yes --file /tmp/requirements.txt

#trimesh
RUN pip install trimesh[easy]


