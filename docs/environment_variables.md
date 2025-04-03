# Environment Variables Configuration Guide

This document explains how to configure environment variables for both the frontend and backend components of the Marketing ChatBot application.

## Backend Configuration

### Available Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COMPANY_NAME` | The name of the company using the chatbot | "MSquared" |
| `OPENAI_API_KEY` | OpenAI API key for LLM access | None (Required) |
| `CACHE_PATH` | Path to the chat cache database | "data/cache/chat_cache.db" |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | "INFO" |

### Setting Environment Variables

For development, you can set these variables in your shell before running the application:

```bash
export COMPANY_NAME=MyCompany
export OPENAI_API_KEY=your-api-key
```

For production deployment, add them to your environment configuration or Docker configuration.

## Frontend Configuration

### Available Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REACT_APP_COMPANY_NAME` | The name of the company using the chatbot | "MSquared" |
| `REACT_APP_API_BASE_URL` | The base URL for the backend API | "http://localhost:8005" |
| `REACT_APP_ENABLE_DEBUG_MODE` | Whether to enable debug mode | false |

### Setting Environment Variables

Create a `.env` file in the root of the frontend directory with your configuration:

```
REACT_APP_COMPANY_NAME=MyCompany
REACT_APP_API_BASE_URL=http://localhost:8005
REACT_APP_ENABLE_DEBUG_MODE=false
```

For containerized deployment, pass these variables when starting the container:

```bash
docker run -p 80:80 \
  -e REACT_APP_COMPANY_NAME=MyCompany \
  -e REACT_APP_API_BASE_URL=http://api.example.com \
  marketing-chatbot-frontend
```

## Accessing Environment Variables

### In Backend Code

```python
from app.config.chat_config import ChatConfig

company_name = ChatConfig.COMPANY_NAME
```

### In Frontend Code

```javascript
import config from '../config/config';

const companyName = config.COMPANY_NAME;
```

## Environment Variables and Building

### Backend

Backend environment variables are read at runtime, so they can be changed without rebuilding the application.

### Frontend

Frontend environment variables are baked into the build at build time. If you change environment variables after building, you'll need to rebuild the frontend.

For development with React's development server, environment variables in `.env` are read when you start the server with `npm start`.

For production, environment variables must be set before building with `npm run build` to be included in the build.