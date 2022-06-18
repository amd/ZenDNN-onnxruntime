#!/bin/bash

while getopts h:m: parameter
do case "${parameter}"
in
h) HOME_DIR=${OPTARG};;
m) MODEL_FILEPATH=${OPTARG};;
esac
done 

# files to download info
FLOAT_16="float16.py"
FLOAT_16_LINK="https://raw.githubusercontent.com/microsoft/onnxconverter-common/master/onnxconverter_common/float16.py"

download_files() {
    wget --no-check-certificate -c $FLOAT_16_LINK 
}

cd $HOME_DIR

apt update
apt-get install -y --no-install-recommends pciutils
pip install --upgrade pip
pip install -r requirements.txt    
download_files
python tester.py -m $MODEL_FILEPATH
