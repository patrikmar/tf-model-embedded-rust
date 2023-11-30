import numpy as np
import tensorflow as tf
from typing import Dict

class DummyMarketModel(tf.Module):
    def __init__(self):
        super().__init__()

        inputs = {
            "mediation_name": tf.string,
            "country": tf.string,
            "model_type": tf.string,
            "game_id": tf.string,
            "session_depth": tf.int64,
            "max_bid": tf.uint64,
            "original_rev_share": tf.float64
        }

        self.predict = tf.function(
            func=self._predict,
            input_signature=[{
                feature: tf.TensorSpec(shape=[None], dtype=dtype, name=feature) for (feature, dtype) in inputs.items()
            }]
        )

    def _predict(self, features: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        return {
            "optimized_rev_share": np.float64(0.1),
            "log_str": "some_log",
            "prob_win_for_optimized_bid": np.float64(0.3),
            "prob_win_for_economical_bid": np.float64(0.2),
            "optimized_bid": np.uint64(1234),
            "economical_bid": np.uint64(1000),
            "model_name": "baseline",
            "model_version": "1.0-a",
            "model_timestamp": "20230127095932",
        }

if __name__ == "__main__":
    model = DummyMarketModel()
    tf.saved_model.save(model, "dummymodel")