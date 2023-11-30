#!/bin/bash

docker run -it -v $(pwd):/app -w /app --rm tensorflow/tensorflow \
   python market_model.py


