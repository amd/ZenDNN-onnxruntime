#!/bin/bash

# Parse Arguments
while getopts i:h:m: parameter
do case "${parameter}"
in 
i) DOCKER_IMAGE=${OPTARG};;
h) HOME_DIR=${OPTARG};;
m) MODEL_FILEPATH=${OPTARG};;
esac
done 

# Variables
DOCKER_HOME_DIR='/perf/'

docker run --gpus all -v $HOME_DIR:$DOCKER_HOME_DIR $DOCKER_IMAGE /bin/bash $DOCKER_HOME_DIR'tester.sh' -h $DOCKER_HOME_DIR -m $MODEL_FILEPATH

