import abc
import enum
import numpy as np
import onnx
import onnxruntime as ort

from dataclasses import dataclass
from onnx import parser, version_converter
from pathlib import Path
from typing import List, Tuple, Union

from PIL import Image

# The ONNX graph parser has it's own map of names just to be special
# https://github.com/onnx/onnx/blob/604af9cb28f63a6b9924237dcb91530649233db9/onnx/defs/parser.h#L72
TENSOR_TYPE_TO_ONNX_GRAPH_TYPE = {
    int(onnx.TensorProto.FLOAT): 'float',
    int(onnx.TensorProto.UINT8): 'uint8',
    int(onnx.TensorProto.INT8): 'int8',
    int(onnx.TensorProto.UINT16): 'uint16',
    int(onnx.TensorProto.INT16): 'int16',
    int(onnx.TensorProto.INT32): 'int32',
    int(onnx.TensorProto.INT64): 'int64',
    int(onnx.TensorProto.STRING): 'string',
    int(onnx.TensorProto.BOOL): 'bool',
    int(onnx.TensorProto.FLOAT16): 'float16',
    int(onnx.TensorProto.DOUBLE): 'double',
    int(onnx.TensorProto.UINT32): 'uint32',
    int(onnx.TensorProto.UINT64): 'uint64',
    int(onnx.TensorProto.COMPLEX64): 'complex64',
    int(onnx.TensorProto.COMPLEX128): 'complex128',
    int(onnx.TensorProto.BFLOAT16): 'bfloat16',
}

# We need to use an opset that's valid for the pre/post processing operators we add.
# Could alternatively use onnx.defs.onnx_opset_version to match the onnx version installed, but that's not deterministic
PRE_POST_PROCESSING_ONNX_OPSET = 16


# Create an checker context that includes the ort-ext domain so that custom ops don't cause failure
def _create_custom_op_checker_context():
    context = onnx.checker.C.CheckerContext()
    context.ir_version = onnx.checker.DEFAULT_CONTEXT.ir_version

    # arbitrary default of ONNX v16 and example custom op domain.
    onnx_version = PRE_POST_PROCESSING_ONNX_OPSET
    context.opset_imports = {'': onnx_version, 'ortext': 1}

    return context


@dataclass
class IoMapEntry:
    """Entry to map the output index from a producer to the input index of a consumer."""
    # optional producer value
    #   producer is inferred from previous step if not provided.
    #   producer Step will be search for by PrePostProcessor using name if str is provided
    producer: Union["Step", str] = None
    producer_idx: int = 0
    consumer_idx: int = 0


class Step(object):
    """Base class for a pre or post processing step"""

    _step_num = 0  # unique step number so we can prefix the naming in the graph created for the step
    _prefix = '_ppp'
    _custom_op_checker_context = _create_custom_op_checker_context()

    # If debug is True each step will add any consumed inputs to the outputs of the merged graph so they are outputs of
    # the complete pre/post processing graph and can be used for debugging. The pre/post processing graph will be
    # saved to preprocessing.onnx/postprocessing.onnx. The final merge with the original model will NOT occur as it
    # currently is not able to handle the additional outputs when merging.
    debug = False


    def __init__(self, inputs: List[str], outputs: List[str], name: str = None):
        self.step_num = Step._step_num
        self.input_names = inputs
        self.output_names = outputs
        self.name = name if name else f'{self.__class__.__name__}'
        self._prefix = f'{Step._prefix}{self.step_num}_'

        Step._step_num += 1

    def connect(self, entry: IoMapEntry):
        """
        Connect the value name from a previous step to an input of this step so we use matching value names.
        This makes joining the GraphProto created by each step trivial.
        """
        assert(len(entry.producer.output_names) >= entry.producer_idx)
        assert(len(self.input_names) >= entry.consumer_idx)
        assert(isinstance(entry.producer, Step))
        self.input_names[entry.consumer_idx] = entry.producer.output_names[entry.producer_idx]

    def apply(self, graph: onnx.GraphProto):
        """Append the nodes that implement this step to the provided graph."""

        graph_for_step = self._create_graph_for_step(graph)

        # TODO: The onnx renamer breaks the graph input/output name mapping between steps as rename_inputs and
        # rename_outputs only applies if we're not updating all the other values (rename_edges=True)
        # It's better to rename each graph so we guarantee no clashes though
        onnx.compose.add_prefix_graph(graph_for_step, self._prefix, inplace=True)
        result = self.__merge(graph, graph_for_step)

        # update self.output_names to the prefixed names so that when we connect later Steps the values match
        new_outputs = [self._prefix + o for o in self.output_names]
        result_outputs = [o.name for o in result.output]

        # sanity check that all of our outputs are in the merged graph
        for o in new_outputs:
            assert(o in result_outputs)

        self.output_names = new_outputs

        return result

    @abc.abstractmethod
    def _create_graph_for_step(self, graph: onnx.GraphProto):
        pass

    def __merge(self, first: onnx.GraphProto, second: onnx.GraphProto):
        # We prefixed everything in `second` so allow for that when connection the two graphs
        io_map = []
        for o in first.output:
            # apply the same prefix to the output from the previous step to match the prefixed graph from this step
            prefixed_output = self._prefix + o.name
            for i in second.input:
                if i.name == prefixed_output:
                    io_map.append((o.name, i.name))

        # merge with existing graph
        merged_graph = onnx.compose.merge_graphs(first, second, io_map)

        # if debugging add output from previous step to graph outputs
        if Step.debug:
            for o in first.output:
                if o.name in self.input_names:
                    merged_graph.output.append(o)

        return merged_graph

    @staticmethod
    def _elem_type_str(type: int):
        return TENSOR_TYPE_TO_ONNX_GRAPH_TYPE[type]

    @staticmethod
    def _shape_to_str(shape: onnx.TensorShapeProto):
        def dim_to_str(dim):
            if dim.HasField("dim_value"):
                return str(dim.dim_value)
            elif dim.HasField("dim_param"):
                return dim.dim_param
            else:
                return ""

        shape_str = ','.join([dim_to_str(dim) for dim in shape.dim])
        return shape_str

    def _input_tensor_type(self, graph: onnx.GraphProto, input_num: int) -> onnx.TensorProto:
        """Get the onnx.TensorProto for the input from the outputs of the graph we're appending to."""

        input_type = None
        for o in graph.output:
            if o.name == self.input_names[input_num]:
                input_type = o.type.tensor_type
                break

        if not input_type:
            raise ValueError(f"Input {self.input_names[input_num]} was not found in outputs of graph.")

        return input_type

    def _get_input_type_and_shape_strs(self, graph: onnx.GraphProto, input_num: int) -> Tuple[str, str]:
        input_type = self._input_tensor_type(graph, input_num)
        return Step._elem_type_str(input_type.elem_type), Step._shape_to_str(input_type.shape)


class PrePostProcessor:
    """
    Class to handle running all the pre/post processing steps and updating the model.
    """
    def __init__(self, inputs: List[onnx.ValueInfoProto] = None, outputs: List[onnx.ValueInfoProto] = None):
        self.pre_processors = []
        self.post_processors = []

        # Connections for each pre/post processor. 1:1 mapping with entries in pre_processors/post_processors
        self._pre_processor_connections = []  # type: List[List[IoMapEntry]]
        self._post_processor_connections = []  # type: List[List[IoMapEntry]]

        # explicitly join outputs from Steps in pre_processors to inputs of the original model
        # format is Step or step name, step_idx, name of graph input/output
        # Pre-processing we connect Step output to original model:
        #   - step_idx is for Step.output_names, and name is in graph.input
        # Post-processing we connect the original model output to the Step input
        #   - step_idx is for Step.input_names, and name is in graph.output
        self._pre_processing_joins = None  # type: Union[None,List[Tuple[Union[Step,str], int, str]]]
        self._post_processing_joins = None  # type: Union[None,List[Tuple[Union[Step,str], int, str]]]

        self._inputs = inputs if inputs else []
        self._outputs = outputs if outputs else []

    def add_pre_processing(self, items: List[Union[Step, Tuple[Step, List[IoMapEntry]]]]):
        """
        Add the pre-processing steps.
        Options are:
          Add Step with default connection of outputs from the previous step (if available) to inputs of this step.
          Add tuple of Step or step name and io_map for connections between two steps.
            Previous step is inferred if IoMapEntry.producer is not specified.
        """
        self.__add_processing(self.pre_processors, self._pre_processor_connections, items)

    def add_post_processing(self, items: List[Union[Step, Tuple[Step, List[IoMapEntry]]]]):
        self.__add_processing(self.post_processors, self._post_processor_connections, items)

    def add_joins(self,
                  preprocessing_joins: List[Tuple[Step, int, str]] = None,
                  postprocessing_joins: List[Tuple[Step, int, str]] = None):
        if preprocessing_joins:
            for step, step_idx, graph_input in preprocessing_joins:
                assert(step and step_idx <= len(step.output_names))

            self._pre_processing_joins = preprocessing_joins

        if postprocessing_joins:
            for step, step_idx, graph_output in postprocessing_joins:
                assert(step and step_idx <= len(step.input_names))

            self._post_processing_joins = postprocessing_joins

    def _add_connection(self, consumer: Step, entry: IoMapEntry):
        producer = self.__producer_from_step_or_str(entry.producer)

        if not((producer in self.pre_processors or producer in self.post_processors) or
                (consumer in self.pre_processors or consumer in self.post_processors)):
            raise ValueError("Producer and Consumer processors must both be registered")

        if producer in self.pre_processors:
            if (consumer in self.pre_processors and
                    self.pre_processors.index(producer) > self.pre_processors.index(consumer)):
                raise ValueError("Producer was registered after consumer and cannot be connected")
        elif producer in self.post_processors:
            if consumer not in self.post_processors:
                raise ValueError("Cannot connect pre-processor consumer with post-processor producer")
            elif self.post_processors.index(producer) > self.post_processors.index(consumer):
                raise ValueError("Producer was registered after consumer and cannot be connected")

        assert(isinstance(producer, Step))
        consumer.connect(entry)

    def run(self, model: onnx.ModelProto):
        pre_process_graph = None
        post_process_graph = None

        # update to the ONNX opset we're using
        model_opset = [entry.version for entry in model.opset_import
                       if entry.domain == '' or entry.domain == 'ai.onnx'][0]
        if (model_opset > PRE_POST_PROCESSING_ONNX_OPSET):
            # It will probably work if the user updates PRE_POST_PROCESSING_ONNX_OPSET to match the model
            # but there are no guarantees.
            # Would only break if ONNX operators used in the pre/post processing graphs have had spec changes.
            raise ValueError(f"Model opset is {model_opset} which is newer than the opset used by this script.")

        model = onnx.version_converter.convert_version(model, PRE_POST_PROCESSING_ONNX_OPSET)

        def name_nodes(new_graph: onnx.GraphProto, prefix: str):
            idx = 0
            for n in new_graph.node:
                if not n.name:
                    n.name = prefix + str(idx)
                    idx += 1

        def connect_and_run(graph: onnx.GraphProto, processor: Step, connections: List[IoMapEntry]):
            for connection in connections:
                assert(connection.producer)
                self._add_connection(processor, connection)

            return processor.apply(graph)

        graph = model.graph
        # add pre-processing
        if self.pre_processors:
            # create empty graph with pass through of the requested input name
            pre_process_graph = onnx.GraphProto()
            for i in self._inputs:
                pre_process_graph.input.append(i)
                pre_process_graph.output.append(i)

            for idx, step in enumerate(self.pre_processors):
                pre_process_graph = connect_and_run(pre_process_graph, step, self._pre_processor_connections[idx])

            # name all the nodes for easier debugging
            name_nodes(pre_process_graph, "pre_process_")

            # dump preprocessing graph if debugging
            if Step.debug:
                pre_process_model = onnx.helper.make_model(pre_process_graph, opset_imports=model.opset_import)
                onnx.save_model(pre_process_model, r'preprocessing.onnx')

            if not self._pre_processing_joins:
                # default to 1:1 between outputs of last step with inputs of model
                last_step = self.pre_processors[-1]
                num_entries = min(len(last_step.output_names), len(graph.input))
                self._pre_processing_joins = [(last_step, i, graph.input[i].name) for i in range(0, num_entries)]

            # map the pre-processing outputs to graph inputs
            io_map = []  # type: List[Tuple[str, str]]
            for step, step_idx, graph_input in self._pre_processing_joins:
                io_map.append((step.output_names[step_idx], graph_input))

            graph = onnx.compose.merge_graphs(pre_process_graph, graph, io_map)

        # add post-processing
        if self.post_processors:
            orig_model_outputs = [o.name for o in model.graph.output]
            graph_outputs = [o.name for o in graph.output]  # this may have additional outputs from pre-processing

            # create default joins if needed
            if not self._post_processing_joins:
                # default to 1:1 between outputs of original model with inputs of first post-processing step
                first_step = self.post_processors[0]
                num_entries = min(len(first_step.input_names), len(orig_model_outputs))
                self._post_processing_joins = [(first_step, i, orig_model_outputs[i]) for i in range(0, num_entries)]

            # update the input names for the steps to match the values produced by the model
            for step, step_idx, graph_output in self._post_processing_joins:
                assert(graph_output in graph_outputs)
                step.input_names[step_idx] = graph_output

            # create empty graph with the values that will be available to the post-processing
            # we do this so we can create a standalone graph for easier debugging
            post_process_graph = onnx.GraphProto()
            for o in graph.output:
                post_process_graph.input.append(o)
                post_process_graph.output.append(o)

            for idx, step in enumerate(self.post_processors):
                post_process_graph = connect_and_run(post_process_graph, step, self._post_processor_connections[idx])

            name_nodes(pre_process_graph, "post_process_")

            if Step.debug:
                post_process_model = onnx.helper.make_model(post_process_graph, opset_imports=model.opset_import)
                onnx.save_model(post_process_model, r'postprocessing.onnx')

            # io_map should be 1:1 with the post-processing graph given we updated the step input names to match
            io_map = [(o, o) for o in graph_outputs]
            graph = onnx.compose.merge_graphs(graph, post_process_graph, io_map)

        new_model = onnx.helper.make_model(graph)
        custom_op_import = new_model.opset_import.add()
        custom_op_import.domain = 'ortext'
        custom_op_import.version = 1
        onnx.checker.check_model(new_model)

        if Step.debug:
            print("Step.debug is set and merged model will not be valid as the joining logic does not handle "
                  "the additional debug outputs.")
            new_model = None

        return new_model

    def __add_processing(self,
                         processors: List[Step],
                         processor_connections: List[List[IoMapEntry]],
                         items: List[Union[Step, Tuple[Union[Step,str], List[IoMapEntry]]]]):
        """
        Add the pre/post processing steps and join with existing steps.

        Args:
            processors: List of processors to add
            processor_connections: Manual connections to create between the pre/post processing graph and the model
            items: List of processors to add.
                   If a Step instances are provided the step will be joined with the immediately previous step.
                   Explicit IoMapEntries can also be provided to join with arbitrary previous steps. The previous step
                   instance can be provided, or it can be lookup up by name.
        """

        for item in items:
            step = None
            explicit_io_map_entries = None

            if isinstance(item, Step):
                step = item
            elif isinstance(item, tuple):
                step_or_str, explicit_io_map_entries = item
                step = self.__producer_from_step_or_str(step_or_str)
            else:
                raise ValueError("Unexpected type " + str(type(item)))

            # start with implicit joins and replace with explicitly provided ones
            # this allows the user to specify the minimum number of manual joins
            io_map_entries = [None] * len(step.input_names)  # type: List[Union[None,IoMapEntry]]
            prev_step = None if len(processors) == 0 else processors[-1]
            if prev_step:
                # default is connecting as many outputs from the previous step as possible
                for i in range(0, min(len(prev_step.output_names), len(step.input_names))):
                    io_map_entries[i] = IoMapEntry(prev_step, i, i)

            # add explicit connections
            if explicit_io_map_entries:
                for entry in explicit_io_map_entries:
                    if not entry.producer:
                        producer = prev_step
                    else:
                        producer = self.__producer_from_step_or_str(entry.producer)

                    io_map_entries[entry.consumer_idx] = IoMapEntry(producer, entry.producer_idx, entry.consumer_idx)

            processors.append(step)
            processor_connections.append([entry for entry in io_map_entries if entry is not None])

    def __producer_from_step_or_str(self, entry: Union[Step,str]):
        if isinstance(entry, Step):
            return entry
        if isinstance(entry, str):
            # search for existing pre or post processing step by name.
            match = (next((s for s in self.pre_processors if s.name == entry), None) or
                     next((s for s in self.post_processors if s.name == entry), None))

            if not match:
                raise ValueError(f'Step named {entry} was not found')

            return match

#
# Pre/Post processing steps
#
class ConvertImageToRGB(Step):
    def __init__(self, name: str = None):
        super().__init__(['image'], ['rgb_data'], name)

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        output_shape_str = f'torgb_ppp_{self.step_num}_h, torgb_ppp_{self.step_num}_w, torgb_ppp_{self.step_num}_c'

        converter_graph = onnx.parser.parse_graph(f'''\
            convert_to_rgb ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{output_shape_str}C] {self.output_names[0]})  
            {{
                {self.output_names[0]} = ortext.ConvertImageToRGB ({self.input_names[0]})
            }}
            ''')

        onnx.checker.check_graph(converter_graph, Step._custom_op_checker_context)
        return converter_graph


class Resize(Step):
    def __init__(self, height: int, width: int, layout: str = "HWC", name: str = None):
        super().__init__(['image'], ['resized_image'], name)
        self.height = height
        self.width = width
        self._layout = layout

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        dims = input_shape_str.split(',')

        # adjust for layout
        # resize will use the largest ratio so both sides won't necessary match the requested height and width.
        # use symbolic names for the output dims as we have to provide values. prefix the names to try and
        # avoid any clashes
        scales_constant_str = 'k1f = Constant <value = float[1] {1.0}> ()'
        if self._layout == 'HWC':
            assert(len(dims) == 3)
            split_str = "h, w, c"
            scales_str = "ratio_resize, ratio_resize, k1f"
            output_shape_str = f'resize_ppp_{self.step_num}_h, resize_ppp_{self.step_num}_w, {dims[-1]}'
        elif self._layout == 'CHW':
            assert(len(dims) == 3)
            split_str = "c, h, w"
            scales_str = "k1f, ratio_resize, ratio_resize"
            output_shape_str = f'{dims[0]}, resize_ppp_{self.step_num}_h, resize_ppp_{self.step_num}_w'
        elif self._layout == 'HW':
            assert(len(dims) == 2)
            split_str = 'h, w'
            scales_str = "ratio_resize, ratio_resize"
            scales_constant_str = ''
            output_shape_str = f'resize_ppp_{self.step_num}_h, resize_ppp_{self.step_num}_w'
        else:
            raise ValueError(f'Unsupported layout of {self._layout}')

        resize_graph = onnx.parser.parse_graph(f'''\
            resize ({input_type_str}[{input_shape_str}] {self.input_names[0]}) => 
                ({input_type_str}[{output_shape_str}] {self.output_names[0]})
            {{
                target_size = Constant <value = float[2] {{{float(self.height)}, {float(self.width)}}}> ()
                image_shape = Shape ({self.input_names[0]})
                {split_str} = Split <axis = 0> (image_shape)
                hw = Concat <axis = 0> (h, w)
                hw_f = Cast <to = 1> (hw)
                target_size_f = Cast <to = 1> (target_size)
                ratios = Div(target_size_f, hw_f)
                ratio_resize = ReduceMax(ratios)

                {scales_constant_str}
                scales_resize = Concat <axis = 0> ({scales_str})
                {self.output_names[0]} = Resize <mode = \"linear\", 
                                                 nearest_mode = \"floor\",
                                                 coordinate_transformation_mode = \"pytorch_half_pixel\"> 
                                                 ({self.input_names[0]}, , scales_resize)
            }}
            ''')

        onnx.checker.check_graph(resize_graph)
        return resize_graph


class CenterCrop(Step):
    def __init__(self, height: int, width: int, name: str = None):
        super().__init__(['image'], ['cropped_image'], name)
        self.height = height
        self.width = width

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        dims = input_shape_str.split(',')
        output_shape_str = f'{self.height}, {self.width}, {dims[-1]}'

        crop_graph = onnx.parser.parse_graph(f'''\
            crop ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{output_shape_str}] {self.output_names[0]})
            {{
                target_crop = Constant <value = int64[2] {{{self.height}, {self.width}}}> ()
                k2 = Constant <value = int64[1] {{2}}> ()
                axes = Constant <value = int64[2] {{0, 1}}> ()
                x_shape = Shape ({self.input_names[0]})
                h, w, c = Split <axis = 0> (x_shape)
                hw = Concat <axis = 0> (h, w)
                hw_diff = Sub (hw, target_crop)
                start_xy = Div (hw_diff, k2)
                end_xy = Add (start_xy, target_crop)
                {self.output_names[0]} = Slice ({self.input_names[0]}, start_xy, end_xy, axes)
            }}
            ''')

        onnx.checker.check_graph(crop_graph)
        return crop_graph


class ImageBytesToFloat(Step):
    """
    Convert uint8 image data to float by dividing by 255.
    """
    def __init__(self, name: str = None):
        super().__init__(['data'], ['flaot_data'], name)

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)

        byte_to_float_graph = onnx.parser.parse_graph(f'''\
            byte_to_float ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => (float[{input_shape_str}] {self.output_names[0]})
            {{
                k255 = Constant <value = float[1] {{255.0}}> ()
                {self.output_names[0]} = Div({self.input_names[0]}, k255)
            }}
            ''')

        onnx.checker.check_graph(byte_to_float_graph)
        return byte_to_float_graph


class FloatToImageBytes(Step):
    """
    Reverse ImageBytesToFloat by multiplying by 255 and casting to uint8
    """
    def __init__(self, name: str = None):
        super().__init__(['float_data'], ['uint8_data'], name)

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)

        float_to_byte_graphs = onnx.parser.parse_graph(f'''\
            float_to_type (float[{input_shape_str}] {self.input_names[0]}) 
                => (uint8[{input_shape_str}] {self.output_names[0]})
            {{
                k255 = Constant <value = float[1] {{255.0}}> ()
                input_x_255 = Mul({self.input_names[0]}, k255)
                {self.output_names[0]} = Cast <to = {int(onnx.TensorProto.UINT8)}>(input_x_255)
            }}
            ''')

        onnx.checker.check_graph(float_to_byte_graphs)
        return float_to_byte_graphs


class Normalize(Step):
    def __init__(self, normalization_values: List[Tuple[float, float]], hwc_layout=True, name: str = None):
        """
        Provide normalization values as pairs of mean and stddev.
        Per-channel normalization requires 3 tuples.
        If a single tuple is provided the values will be applied to all channels.

        Layout can be HWC or CHW. Set hwc_layout to False for CHW.
        """
        super().__init__(['data'], ['normalized_data'], name)

        # duplicate for each channel if needed
        if len(normalization_values) == 1:
            normalization_values *= 3

        assert(len(normalization_values) == 3)
        self.normalization_values = normalization_values
        self.hwc_layout = hwc_layout

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        mean0 = self.normalization_values[0][0]
        mean1 = self.normalization_values[1][0]
        mean2 = self.normalization_values[2][0]
        stddev0 = self.normalization_values[0][1]
        stddev1 = self.normalization_values[1][1]
        stddev2 = self.normalization_values[2][1]

        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        values_shape = '3' if self.hwc_layout else '3, 1, 1'

        normalize_graph = onnx.parser.parse_graph(f'''\
            normalize ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => (float[{input_shape_str}] {self.output_names[0]})
            {{
                kMean = Constant <value = float[{values_shape}] {{{mean0}, {mean1}, {mean2}}}> ()
                kStddev = Constant <value = float[{values_shape}] {{{stddev0}, {stddev1}, {stddev2}}}> ()                
                fp32_input = Cast <to = 1> ({self.input_names[0]})
                tmp1 = Sub(fp32_input, kMean)
                {self.output_names[0]} = Div(tmp1, kStddev)
            }}
            ''')

        onnx.checker.check_graph(normalize_graph)
        return normalize_graph


class Unsqueeze(Step):
    def __init__(self, axes: List[int], name: str = None):
        super().__init__(['data'], ['expanded'], name)
        self._axes = axes

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        dims = input_shape_str.split(',')

        for idx in self._axes:
            dims.insert(idx, '1')

        output_shape_str = ','.join(dims)
        axes_strs = [str(axis) for axis in self._axes]

        unsqueeze_graph = onnx.parser.parse_graph(f'''\
            unsqueeze ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{output_shape_str}] {self.output_names[0]})  
            {{
                axes = Constant <value = int64[{len(self._axes)}] {{{','.join(axes_strs)}}}> ()
                {self.output_names[0]} = Unsqueeze({self.input_names[0]}, axes)
            }}
            ''')

        onnx.checker.check_graph(unsqueeze_graph)
        return unsqueeze_graph


class Squeeze(Step):
    def __init__(self, axes: List[int] = None, name: str = None):
        super().__init__(['data'], ['squeezed'], name)
        self._axes = axes

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        dims = input_shape_str.split(',')
        output_dims = [dim for idx, dim in enumerate(dims) if idx not in self._axes]
        output_shape_str = ','.join(output_dims)

        if self._axes:
            axes_strs = [str(axis) for axis in self._axes]
            graph_str = f'''\
            axes = Constant <value = int64[{len(self._axes)}] {{{','.join(axes_strs)}}}> ()
            {self.output_names[0]} = Squeeze({self.input_names[0]}, axes)
            '''
        else:
            graph_str = f"{self.output_names[0]} = Squeeze({self.input_names[0]})"

        squeeze_graph = onnx.parser.parse_graph(f'''\
            squeeze ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{output_shape_str}] {self.output_names[0]})  
            {{
                {graph_str}
            }}
            ''')

        onnx.checker.check_graph(squeeze_graph)
        return squeeze_graph


class Transpose(Step):
    def __init__(self, perms: List[int], name: str = None):
        super().__init__(['X'], ['Y'], name)
        self.perms = perms

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        perms_str = ','.join([str(idx) for idx in self.perms])
        dims = input_shape_str.split(',')
        output_dims = [dims[axis] for axis in self.perms]
        output_shape_str = ','.join(output_dims)

        transpose_graph = onnx.parser.parse_graph(f'''\
            transpose ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{output_shape_str}] {self.output_names[0]})  
            {{
                {self.output_names[0]} = Transpose <perm = [{perms_str}]> ({self.input_names[0]})
            }}
            ''')

        onnx.checker.check_graph(transpose_graph)
        return transpose_graph


class NhwcToNchw(Transpose):
    def __init__(self, name: str = None):
        super().__init__([0, 3, 1, 2], name)


class HwcToChw(Transpose):
    def __init__(self, name: str = None):
        super().__init__([2, 0, 1], name)


class RGBToYCbCr(Step):
    def __init__(self, name: str = None):
        super().__init__(['rgb_data'], ['Y', 'Cb', 'Cr'], name)

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        # each output is {h, w}. TBD if input is CHW or HWC though. Once we figure that out we could copy values from
        # the input shape
        output_shape_str = f'ToYCbCr_ppp_{self.step_num}_h, ToYCbCr_ppp_{self.step_num}_w'
        assert(input_type_str == "uint8")

        converter_graph = onnx.parser.parse_graph(f'''\
            RGB_to_YCbCr ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{output_shape_str}] {self.output_names[0]},
                    {input_type_str}[{output_shape_str}] {self.output_names[1]},
                    {input_type_str}[{output_shape_str}] {self.output_names[2]})  
            {{
                {','.join(self.output_names)} = ortext.RGBToYCbCr ({self.input_names[0]})
            }}
            ''')

        onnx.checker.check_graph(converter_graph, Step._custom_op_checker_context)
        return converter_graph


class YCbCrToRGB(Step):
    def __init__(self, name: str = None):
        super().__init__(['Y', 'Cb', 'Cr'], ['rgb_data'], name)

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str0, input_shape_str0 = self._get_input_type_and_shape_strs(graph, 0)
        input_type_str1, input_shape_str1 = self._get_input_type_and_shape_strs(graph, 1)
        input_type_str2, input_shape_str2 = self._get_input_type_and_shape_strs(graph, 2)
        assert(input_type_str0 == input_type_str1 and input_type_str0 == input_type_str2)
        assert(len(input_shape_str0.split(',')) == len(input_shape_str1.split(',')) and
               len(input_shape_str0.split(',')) == len(input_shape_str2.split(',')))

        output_shape_str = f'3,{input_shape_str0}'

        converter_graph = onnx.parser.parse_graph(f'''\
            YCbCr_to_RGB ({input_type_str0}[{input_shape_str0}] {self.input_names[0]},
                          {input_type_str1}[{input_shape_str1}] {self.input_names[1]},
                          {input_type_str2}[{input_shape_str2}] {self.input_names[2]}) 
                => ({input_type_str0}[{output_shape_str}] {self.output_names[0]})  
            {{
                {self.output_names[0]} = ortext.YCbCrToRGB ({','.join(self.input_names)})
            }}
            ''')

        onnx.checker.check_graph(converter_graph, Step._custom_op_checker_context)
        return converter_graph


def create_value_info_for_image_bytes(name: str):
    # create a ValueInfoProto for a buffer of bytes containing an input image. could be jpeg/png/bmp
    input_type = onnx.helper.make_tensor_type_proto(elem_type=onnx.TensorProto.UINT8, shape=['NumBytes'])
    return onnx.helper.make_value_info(name, input_type)


class ModelSource(enum.Enum):
    PYTORCH = 0
    TENSORFLOW = 1
    OTHER = 2


def mobilenet(model_file: Path, output_file: Path, model_source: ModelSource = ModelSource.PYTORCH):
    model = onnx.load(str(model_file.resolve(strict=True)))
    inputs = [create_value_info_for_image_bytes('image')]

    if model_source == ModelSource.PYTORCH:
        normalization_params = [(0.485, 0.229), (0.456, 0.224), (0.406, 0.225)]
    else:
        # for the sake of this example use mean/stddev from the TF Lite tasks example metadata.
        # As we convert to float with divide by 255 first use 0.5 for each (== 127.5/255)
        # https://github.com/tensorflow/examples/blob/9eb657f949c2e8ec8592a9576811db38a86dcbc0/lite/examples/image_classification/metadata/metadata_writer_for_image_classifier.py#L64
        normalization_params = [(0.5, 0.5)]

    runner = PrePostProcessor(inputs)
    runner.add_pre_processing([
        ConvertImageToRGB(),  # custom op to convert jpg/png to RGB (output is HWC)
        Resize(256, 256),
        CenterCrop(244, 244),
        ImageBytesToFloat(),
        Normalize(normalization_params),
        Unsqueeze([0])  # add batch dim of 1 to match model input shape
    ])

    new_model = runner.run(model)

    onnx.save_model(new_model, str(output_file.resolve()))


def superresolution(model_file: Path, output_file: Path):
    # TODO: There seems to be a split with some super resolution models processing RGB input and some processing
    # the Y channel after converting to YCbCr.
    # For the sake of this example implementation we do the trickier YCbCr processing as that involves joining the
    # Cb and Cr channels with the model output to create the resized image.
    model = onnx.load(str(model_file.resolve(strict=True)))

    inputs = [create_value_info_for_image_bytes('image')]

    # assuming input is *CHW, infer the input sizes from the model. requires model input and output has a fixed
    # size for the input and output height and width. user would have to specify otherwise.
    model_input_shape = model.graph.input[0].type.tensor_type.shape
    model_output_shape = model.graph.output[0].type.tensor_type.shape
    assert(model_input_shape.dim[-1].HasField("dim_value"))
    assert(model_input_shape.dim[-2].HasField("dim_value"))
    assert(model_output_shape.dim[-1].HasField("dim_value"))
    assert(model_output_shape.dim[-2].HasField("dim_value"))

    w_in = model_input_shape.dim[-1].dim_value
    h_in = model_input_shape.dim[-2].dim_value
    h_out = model_output_shape.dim[-2].dim_value
    w_out = model_output_shape.dim[-1].dim_value

    pipeline = PrePostProcessor(inputs)
    pipeline.add_pre_processing([
        ConvertImageToRGB(),  # jpg/png image to RGB (HWC)
        Resize(h_in, w_in),
        CenterCrop(h_in, w_in),
        RGBToYCbCr(),         # this produces Y, Cb and Cr outputs. each has shape {h_in, w_in}
        ImageBytesToFloat(),  # we only need to process the Y channel from the previous step so default mapping is fine
        Unsqueeze([0, 1]),    # add batch and channels dim so shape is {1, h_in, w_in}
    ])

    # Post-processing is complicated here. resize the Cb and Cr outputs from the pre-processing to match
    # the model output size, merge those with the Y` model output, and convert back to RGB.

    # create the Steps we need to use in the manual connections
    pipeline.add_post_processing([
        FloatToImageBytes(),
        Squeeze([0, 1], 'RemoveBatchAndChannelsDims'),
        # Verbose example with param names for IoMapEntry to clarify
        (Resize(h_out, w_out, 'HW', 'Resize_Cb'), [IoMapEntry(producer='RGBToYCbCr', producer_idx=1, consumer_idx=0)]),
        (Resize(h_out, w_out, 'HW', 'Resize_Cr'), [IoMapEntry('RGBToYCbCr', 2, 0)]),
        (YCbCrToRGB(), [IoMapEntry('RemoveBatchAndChannelsDims', 0, 0),  # Y' with shape {h, w}
                        IoMapEntry('Resize_Cb', 0, 1),
                        IoMapEntry('Resize_Cr', 0, 2)]),
        # TODO: Convert from RGB to original input format (jpg/png)
    ])

    new_model = pipeline.run(model)
    onnx.save_model(new_model, str(output_file.resolve()))


def update_mobilenet():
    mobilenet_path = r'D:\temp\prepostprocessor_poc\pytorch_mobilenet_v2.onnx'
    mobilenet_withppp_path = r'D:\temp\prepostprocessor_poc\pytorch_mobilenet_v2.with_preprocessing.onnx'

    mobilenet(Path(mobilenet_path), Path(mobilenet_withppp_path))


def update_mobilenet2():
    mobilenet_path = r'D:\temp\prepostprocessor_poc\mobilenetv2-7.onnx'
    mobilenet_withppp_path = r'D:\temp\prepostprocessor_poc\mobilenetv2-7.with_preprocessing.onnx'

    mobilenet(Path(mobilenet_path), Path(mobilenet_withppp_path))


def update_superresolution():
    superresolution_path = r'D:\temp\prepostprocessor_poc\pt_super_resolution.onnx'
    superresolution_withppp_path = r'D:\temp\prepostprocessor_poc\pt_super_resolution.with_preprocessing.onnx'
    superresolution(Path(superresolution_path), Path(superresolution_withppp_path))


def main():
    update_mobilenet2()
    #update_superresolution()


if __name__ == '__main__':
    main()
