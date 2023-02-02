class GraphTransformerRegistry:
    _TRANSFORMER_FUNCS = []

    @classmethod
    def register(cls, fn):
        cls._TRANSFORMER_FUNCS.append(fn)

    @classmethod
    def transform_all(cls, graph):
        for fn in cls._TRANSFORMER_FUNCS:
            fn(graph)


def register_graph_transformer():
    def gradient_wrapper(fn):
        GraphTransformerRegistry.register(fn)
        return fn

    return gradient_wrapper
