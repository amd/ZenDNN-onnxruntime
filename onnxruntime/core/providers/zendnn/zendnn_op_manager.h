/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

/*******************************************************************************
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

#pragma once
#include "zendnn_node_capability.h"
#include <map>
#include <string>
#include <memory>

namespace onnxruntime {
class ZendnnOpManager {
  public:
    ZendnnOpManager();

    /**
     * This will check if the ORT node is Supported by the ZENDNN execution provider
     *
     * Several things will be checked from the node
     * - Is the OpType is regestered with the ZENDNN execution provider?
     * - Are the tensor dimensions Supported by the ZENDNN execution provider
     * - Are operator attributes Supported by the ZENDNN execution provider
     *
     * @param node the node that is being checked
     *
     * @return true if the node is Supported by the ZENDNN execution provider
     *         false is returned otherwise.
     */
    bool IsNodeSupported(const Node *node, const GraphViewer &graph_viewer) const;

    /**
     * Find out if the OpType is one of the OpTypes Supported by the ZENDNN execution provider
     *
     * This only looks at the OpType it does not look at other factors that may mean
     * the operator is not Supported.
     *
     * @param opType the name of the operator i.e. "Add" or "Conv" etc.
     *
     * @return true is the OpType is one of those Supported by the ZENDNN execution provider
     *         false is returned otherwise.
     */
    bool IsOpTypeAvalible(const std::string &opType) const;

  private:
    std::map<std::string, std::unique_ptr<ZendnnNodeCapability>> zendnn_ops_map_;
};
}  // namespace onnxruntime