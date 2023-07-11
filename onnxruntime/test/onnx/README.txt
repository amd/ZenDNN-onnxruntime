/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
* LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
* OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
* WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
*******************************************************************************/

onnx_test_runner [options...] <data_root>
Options:
        -j [models]: Specifies the number of models to run simultaneously.
        -c [runs]: Specifies the number of Session::Run() to invoke simultaneously for each model.
        -n [test_case_name]: Specifies a single test case to run.
        -p [PLANNER_TYPE]: PLANNER_TYPE could be 'seq' or 'simple'. Default: 'simple'.
        -e [EXECUTION_PROVIDER]: EXECUTION_PROVIDER could be 'cpu', 'cuda', 'dnnl', 'zendnn', 'tensorrt' or 'acl'. Default: 'cpu'.
        -h: help

The debug version of this program depends on dbghelp.dll. Please make sure it's in your PATH.

How to run node tests:
1. Install onnx from onnxruntime\cmake\external\onnx

2. Execute test data generator:
       backend-test-tools generate-data -o <some_empty_folder>
   e.g.
       backend-test-tools generate-data -o C:\testdata
    backend-test-tools is a tool under C:\Python35\Scripts (If your python was installed to C:\Python35)

3. compile onnx_test_runner and run
      onnx_test_runner <test_data_dir>
	e.g.
	  onnx_test_runner C:\testdata\node


How to run model tests:
1. Download the test data from Azure
   You can get the latest url from tools/ci_build/github/azure-pipelines/templates/set-test-data-variables-step.yml
   After downloading, please unzip the downloaded file

2. compile onnx_test_runner and run
   onnx_test_runner <test_data_dir>
   e.g.
	 onnx_test_runner C:\testdata
