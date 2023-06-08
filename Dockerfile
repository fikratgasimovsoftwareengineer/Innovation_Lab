# GCC support can be specified at major, minor, or micro version
# (e.g. 8, 8.2 or 8.2.0).
# See https://hub.docker.com/r/library/gcc/ for all supported GCC
# tags from Docker Hub.
# See https://docs.docker.com/samples/library/gcc/ for more on how to use this image
#FROM gcc:latest
FROM ubuntu:20.04
FROM tensorflow/tensorflow:latest-gpu
FROM tensorflow/tensorflow:latest-gpu-jupyter
WORKDIR /tensorfl_lab
# These commands copy your files into the specified directory in the image
# and set that as the working location

COPY . /Innovation_Lab/Innovation_Lab	

RUN echo "****Upgading Pip3"
RUN python3 -m pip install --upgrade pip

RUN echo "**Install Git**"
RUN apt-get update && apt-get install -y git

RUN echo "***Installing LIBGL***"
RUN apt-get update && apt-get install -y build-essential libgl1-mesa-dev

RUN echo "**apt utils"
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils && \
	apt-get install -y build-essential
	


RUN echo "NVIDIA VALOHAI CONFIGURATION"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility


RUN apt-get clean && \
	rm -rf /var/lib/apt/lists/*


RUN echo "******BUILDING FINISHED Successfully********************"
# This command compiles your app using GCC, adjust for your source code


# This command runs your application, comment out this line to compile only
#CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

#LABEL Name=tensorflowtensorrt Version=0.0.1
