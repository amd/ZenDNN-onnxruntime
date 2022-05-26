#!/bin/bash

while getopts d:o:m:e:b:a: parameter
do case "${parameter}"
in
d) PERF_DIR=${OPTARG};;
o) OPTION=${OPTARG};;
m) MODEL_PATH=${OPTARG};;
e) EP_LIST=${OPTARG};;
b) BUILD_ORT=${OPTARG};;
a) OPTIONAL_ARGS=${OPTARG};;
esac
done 

# add ep list
RUN_EPS=""
if [ ! -z "$EP_LIST" ]
then 
    RUN_EPS=" -e $EP_LIST"
    OPTIONAL_ARGS=$OPTIONAL_ARGS$RUN_EPS
fi

# change dir if docker
if [ ! -z $PERF_DIR ]
then 
    echo 'changing to '$PERF_DIR
    cd $PERF_DIR 
fi 

# metadata
FAIL_MODEL_FILE=".fail_model_map"
LATENCY_FILE=".latency_map"
METRICS_FILE=".metrics_map"
SESSION_FILE=".session_map"

# files to download info
FLOAT_16="float16.py"
FLOAT_16_LINK="https://raw.githubusercontent.com/microsoft/onnxconverter-common/master/onnxconverter_common/float16.py"

cleanup_files() {
    rm -f $FAIL_MODEL_FILE
    rm -f $LATENCY_FILE
    rm -f $METRICS_FILE
    rm -f $SESSION_FILE
    rm -f $FLOAT_16
    #rm -rf result/$OPTION
}

download_files() {
    wget --no-check-certificate -c $FLOAT_16_LINK 
}

setup() {
    apt update
    apt-get install -y --no-install-recommends pciutils
    pip install --upgrade pip 
    pip install -r requirements.txt    
    if [ "$BUILD_ORT" = "False" ]
    then
        echo 'installing the nightly wheel file'
        ls Release/dist/* | xargs -n 1 pip install
    fi
    cleanup_files
    download_files
}

setup

# Validate all model/EPs once.
echo "=============================="
echo "===    Validation start    ==="
echo "=============================="
python benchmark_wrapper.py -r validate -m $MODEL_PATH -o result/$OPTION $OPTIONAL_ARGS 

# Run bechmark 50 times.
NUM_DATA_POINTS=50
for ((i = 0 ; i < $NUM_DATA_POINTS ; i++)); do
    echo "=============================="
    echo "=== Benchmark run $i start ==="
    echo "=============================="
    STARTTIME=`date +%s` ;
    python benchmark_wrapper.py -r benchmark -t 4 -m $MODEL_PATH --upload_index $i -o result/$OPTION $OPTIONAL_ARGS
    rm -f $LATENCY_FILE
    rm -f $METRICS_FILE
    #rm -f $SESSION_FILE
    ENDTIME=`date +%s` ;
    echo "==============================================="
    echo "=== Benchmark run $i took $(($ENDTIME - $STARTTIME)) sec ===\n"
    echo "==============================================="
done
