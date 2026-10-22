"""Triton Python backend for the shared hash encoder.

The image sets PYTHONPATH to this repo's src directory. Swap HashEncoder for a
sentence-transformers model behind the same TEXT -> EMBEDDING contract.
"""

import numpy as np

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        from mlops_cluster.embeddings import load_torch_encoder

        self._encoder = load_torch_encoder()

    def execute(self, requests):
        responses = []
        for request in requests:
            tensor = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
            texts = []
            for row in tensor:
                value = row[0]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                texts.append(str(value))
            vectors = np.asarray(self._encoder.encode(texts), dtype=np.float32)
            out = pb_utils.Tensor("EMBEDDING", vectors)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses
