// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This file is used to make a minimal export of JSEP to make it able to load in a browser test environment.

import {env} from 'onnxruntime-common';

import {init} from './init';

export {env as ortEnv, init as jsepInitOnly};
