#!/bin/bash

curl -X POST http://localhost:8888/chat \
  -H "Content-Type: application/json" \
  -d '{"user_input": "什么是vLLM？"}'
