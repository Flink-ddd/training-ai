#!/bin/bash

python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --tokenizer mistralai/Mistral-7B-Instruct-v0.1 \
  --port 8000 \
  --dtype float32
