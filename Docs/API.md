# Chirrup API Documentation

## Overview

Chirrup provides a high-performance, OpenAI-compatible API for RWKV model inference. The service supports streaming responses, continuous batching, and state caching for optimal performance.

## Starting the Service

### Basic Startup

```bash
# Start with default configuration
PYTHON_GIL=0 uv run python -m chirrup.web_service.app --model_path /path/to/your/model.pth
```

### Command Line Parameters

The API service supports the following command-line parameters:

| Parameter            | Type   | Default                                          | Description                         |
| -------------------- | ------ | ------------------------------------------------ | ----------------------------------- |
| `--model_path`       | string | `../models/rwkv7-g0a3-7.2b-20251029-ctx8192.pth` | Path to the model file              |
| `--vocab_path`       | string | `./Albatross/reference/rwkv_vocab_v20230424.txt` | Path to the vocabulary file         |
| `--vocab_size`       | int    | `65536`                                          | Vocabulary size                     |
| `--head_size`        | int    | `64`                                             | Head size for the model             |
| `--worker_num`       | int    | `1`                                              | Number of worker processes (min: 1) |
| `--batch_size`       | int    | `24`                                             | Batch size for processing (min: 1)  |
| `--state_cache_size` | int    | `50`                                             | State cache size (min: 0)           |
| `--host`             | string | `127.0.0.1`                                      | Server host address                 |
| `--port`             | int    | `8000`                                           | Server port (range: 1-65535)        |

### Example Configurations

```bash
# Configuration for 4 GPU machine to provide web service
PYTHON_GIL=0 uv run python -m chirrup.web_service.app \
  --model_path /path/to/your/model.pth \
  --worker_num 4 \
  --batch_size 32 \
  --state_cache_size 100 \
  --host 0.0.0.0 \
  --port 8000

# Single GPU
PYTHON_GIL=0 uv run python -m chirrup.web_service.app \
  --model_path /models/rwkv7-g1a3-1.5b.pth \
  --worker_num 1 \
  --batch_size 16 \
  --state_cache_size 50
```

## API Endpoints

### Base URL

```
http://localhost:8000
```

### Health Check

**Endpoint:** `GET /health`

Check if the service is running and the model is loaded.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": 1703123456,
  "model_loaded": true
}
```

### List Models

**Endpoint:** `GET /v1/models`

List available models.

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "rwkv-latest",
      "object": "model",
      "created": 1703123456,
      "owned_by": "chirrup"
    },
    {
      "id": "rwkv-latest:thinking",
      "object": "model",
      "created": 1703123456,
      "owned_by": "chirrup"
    },
    {
      "id": "rwkv-latest:no-thinking",
      "object": "model",
      "created": 1703123456,
      "owned_by": "chirrup"
    }
  ]
}
```

### Chat Completion

**Endpoint:** `POST /v1/chat/completions`

Create a chat completion, supporting both streaming and non-streaming responses.

**Request Body:**

```json
{
  "model": "rwkv-latest",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Hello, how are you?" }
  ],
  "stream": false,
  "temperature": 1.0,
  "top_p": 0.8,
  "max_tokens": 2048,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stop": null,
  "pad_zero": true
}
```

**Parameters:**

| Parameter           | Type         | Default       | Description                          |
| ------------------- | ------------ | ------------- | ------------------------------------ |
| `model`             | string       | `rwkv-latest` | Model name to use                    |
| `messages`          | array        | Required      | Array of message objects             |
| `stream`            | boolean      | `false`       | Enable streaming response            |
| `temperature`       | float        | `1.0`         | Sampling temperature (0.0-2.0)       |
| `top_p`             | float        | `0.8`         | Nucleus sampling parameter (0.0-1.0) |
| `max_tokens`        | int          | `2048`        | Maximum tokens to generate (min: 1)  |
| `presence_penalty`  | float        | `0.0`         | Presence penalty (0.0-2.0)           |
| `frequency_penalty` | float        | `0.0`         | Frequency penalty (0.0-2.0)          |
| `stop`              | string/array | `null`        | Stop words/sequences                 |
| `pad_zero`          | boolean      | `true`        | Pad prompt with zero token           |

**Model Variants:**

- `rwkv-latest`: Standard model with quick thinking
- `rwkv-latest:thinking`: Model with detailed thinking process
- `rwkv-latest:no-thinking`: Model without thinking process

**Non-Streaming Response:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1703123456,
  "model": "rwkv-latest",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking. How can I assist you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 15,
    "total_tokens": 35
  }
}
```

**Streaming Response:**

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1703123456,"model":"rwkv-latest","choices":[{"index":0,"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1703123456,"model":"rwkv-latest","choices":[{"index":0,"delta":{"content":"!"}}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1703123456,"model":"rwkv-latest","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Batch Translation

**Endpoint:** `POST /v1/batch/translate`

Translate multiple texts in a batch.

**Request Body:**

```json
{
  "source_lang": "en",
  "target_lang": "zh-CN",
  "text_list": ["Hello, world!", "How are you today?"],
  "placeholders": []
}
```

**Parameters:**

| Parameter      | Type   | Default  | Description                 |
| -------------- | ------ | -------- | --------------------------- |
| `source_lang`  | string | `auto`   | Source language code        |
| `target_lang`  | string | Required | Target language code        |
| `text_list`    | array  | Required | Array of texts to translate |
| `placeholders` | array  | `[]`     | Placeholder strings         |

**Supported Languages:**

- `zh-CN`: Chinese (Simplified)
- `zh-TW`: Chinese (Traditional)
- `en`: English
- `ja`: Japanese
- `fr`: French
- `de`: German
- `es`: Spanish
- `ru`: Russian

**Response:**

```json
{
  "translations": [
    {
      "text": "你好，世界！",
      "detected_source_lang": "en"
    },
    {
      "text": "你今天好吗？",
      "detected_source_lang": "en"
    }
  ],
  "id": "chatcmpl-def456",
  "created": 1703123456
}
```

### Batch Rollout

**Endpoint:** `POST /v1/batch/rollout`

Generate multiple completions in a batch.

**Request Body:**

```json
{
  "model": "rwkv-latest",
  "contents": ["The weather today is", "My favorite hobby is"],
  "stream": false,
  "temperature": 1.0,
  "top_p": 0.8,
  "max_tokens": 100,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stop_tokens": [0],
  "pad_zero": true
}
```

**Parameters:**

| Parameter           | Type    | Default       | Description                          |
| ------------------- | ------- | ------------- | ------------------------------------ |
| `model`             | string  | `rwkv-latest` | Model name to use                    |
| `contents`          | array   | Required      | Array of prompt strings              |
| `stream`            | boolean | `false`       | Enable streaming response            |
| `temperature`       | float   | `1.0`         | Sampling temperature (0.0-2.0)       |
| `top_p`             | float   | `0.8`         | Nucleus sampling parameter (0.0-1.0) |
| `max_tokens`        | int     | `2048`        | Maximum tokens to generate (min: 1)  |
| `presence_penalty`  | float   | `0.0`         | Presence penalty (0.0-2.0)           |
| `frequency_penalty` | float   | `0.0`         | Frequency penalty (0.0-2.0)          |
| `stop_tokens`       | array   | `[0]`         | Stop token IDs                       |
| `pad_zero`          | boolean | `true`        | Pad prompt with zero token           |

**Non-Streaming Response:**

```json
{
  "id": "chatcmpl-ghi789",
  "object": "batch.rollout.chunk",
  "created": 1703123456,
  "model": "rwkv-latest",
  "choices": [
    {
      "index": 0,
      "delta": { "content": " sunny and warm with a gentle breeze." }
    },
    {
      "index": 1,
      "delta": { "content": " reading books and exploring new technologies." }
    }
  ]
}
```

## Usage Examples

### Using curl

**Chat Completion:**

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv-latest",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'
```

**Streaming Chat Completion:**

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv-latest",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true,
    "max_tokens": 200
  }'
```

### Using Python

**OpenAI Client:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-required"
)

# Non-streaming
response = client.chat.completions.create(
    model="rwkv-latest",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=100
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="rwkv-latest",
    messages=[
        {"role": "user", "content": "Tell me a story"}
    ],
    stream=True,
    max_tokens=200
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

**Requests Library:**

```python
import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "rwkv-latest",
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print(result["choices"][0]["message"]["content"])
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "message": "Model not loaded",
    "type": "invalid_request_error",
    "code": 503
  }
}
```

### Common Error Codes

| Status Code | Description                              |
| ----------- | ---------------------------------------- |
| 400         | Bad Request - Invalid parameters         |
| 503         | Service Unavailable - Model not loaded   |
| 500         | Internal Server Error - Unexpected error |

### Common Issues and Solutions

1. **Model Loading Failed**

   - Check model path is correct
   - Ensure model file is accessible
   - Verify sufficient GPU memory

2. **Request Timeout**

   - Reduce `max_tokens` for faster responses
   - Increase `batch_size` for better throughput
   - Check system resources

3. **Memory Issues**
   - Reduce `batch_size`
   - Reduce `state_cache_size`
   - Use smaller model if possible

## Performance Optimization

### Recommended Settings

### State Caching

The service includes state caching to improve performance for repeated prompts with similar prefixes. The cache size can be adjusted based on available memory and usage patterns.

- Larger cache: Better performance for repetitive workloads
- Smaller cache: Lower memory usage

### Continuous Batching

The service automatically batches incoming requests to maximize GPU utilization. The `batch_size` parameter controls the maximum batch size processed simultaneously.

## Monitoring

### Health Check

Monitor service health using the `/health` endpoint:

```bash
curl http://localhost:8000/health
```

### Common Issues

1. **Service won't start**

   - Check Python environment
   - Verify all dependencies are installed
   - Check model file permissions

2. **Can't connect to the service**

   - Check service is running
   - Check weather the `--host` parameter is set to `0.0.0.0`
   - Check port availability

3. **Poor performance**

   - Increase `batch_size`
   - Enable state caching
   - Check GPU utilization

4. **Memory errors**

   - Reduce `batch_size`
   - Reduce `state_cache_size`
   - Use smaller model
