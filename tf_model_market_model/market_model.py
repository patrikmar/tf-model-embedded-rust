import numpy as np
import tensorflow as tf
from typing import Dict

class DummyMarketModel(tf.Module):
    def __init__(self):
        super().__init__()

        inputs = {
            "country": tf.string,
            "game_id": tf.string,
            "max_bid": tf.uint64,
            "mediation_name": tf.string,
            "model_type": tf.string,
            "original_rev_share": tf.float64,
            "session_depth": tf.int64,
        }

        self.predict = tf.function(
            func=self._predict,
            input_signature=[{
                feature: tf.TensorSpec(shape=[None], dtype=dtype, name=feature) for (feature, dtype) in inputs.items()
            }]
        )

    def _predict(self, features: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        return {
            "economical_bid": np.uint64(1000),
            "log_str": "some_log",
            "optimized_rev_share": np.float64(0.1),
            "model_name": "baseline",
            "model_timestamp": "20230127095932",
            "model_version": "1.0-a",
            "optimized_bid": np.uint64(1234),
            "prob_win_for_economical_bid": np.float64(0.2),
            "prob_win_for_optimized_bid": np.float64(0.3),
        }

if __name__ == "__main__":
    model = DummyMarketModel()
    tf.saved_model.save(model, "dummymodel")