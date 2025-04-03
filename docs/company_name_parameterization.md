# Company Name Parameterization

This document outlines how the company name is parameterized throughout the application.

## Overview

To make the application configurable for different companies, we've replaced hardcoded references to "MSquared" with environment variables. This allows the application to be customized without code changes.

## Configuration

### Backend

The company name is configured in the backend through the `COMPANY_NAME` environment variable. If not specified, it defaults to "MSquared".

```python
# app/config/chat_config.py
class ChatConfig:
    # Company settings
    COMPANY_NAME = os.getenv("COMPANY_NAME", "MSquared")
```

### Frontend

The company name is configured in the frontend through the `REACT_APP_COMPANY_NAME` environment variable. If not specified, it defaults to "MSquared".

```javascript
// frontend/src/config/config.js
const config = {
    // Company configuration
    COMPANY_NAME: process.env.REACT_APP_COMPANY_NAME || "MSquared",
    // ...
};
```

## Usage

### In Backend

The company name is used in various places, including:

1. System prompts:
```python
# app/config/prompt_config.py
def get_system_prompt():
    prompt = prompts_data["system_prompt"]
    # Replace MSquared with the configured company name
    return prompt.replace("MSquared", ChatConfig.COMPANY_NAME)
```

2. File naming and storage paths

### In Frontend

The company name is used in:

1. Application title and document title
2. Header and branding elements
3. Chat interface titles
4. Document metadata (manifest.json)

## Implementation Details

### Backend Implementation

The `ChatConfig` class provides the central point for accessing the company name. All other components should reference `ChatConfig.COMPANY_NAME` rather than using hardcoded values.

### Frontend Implementation

The frontend uses a React Context Provider pattern to make the company name available throughout the application:

1. `config.js` reads the environment variable
2. `EnvironmentContext.js` provides the config to all components
3. Components use the `useEnvironment` hook to access the company name

```javascript
import { useEnvironment } from '../context/EnvironmentContext';

function MyComponent() {
  const { COMPANY_NAME } = useEnvironment();
  return <h1>{COMPANY_NAME} Dashboard</h1>;
}
```

## Docker Configuration

The Docker Compose file has been updated to include environment variables for both backend and frontend services:

```yaml
# docker-compose.yml
services:
  backend:
    environment:
      - COMPANY_NAME=${COMPANY_NAME:-MSquared}
  
  frontend:
    environment:
      - REACT_APP_COMPANY_NAME=${COMPANY_NAME:-MSquared}
```

## Testing

When testing the application with a custom company name:

1. Backend: Set the `COMPANY_NAME` environment variable
2. Frontend Development: Create a `.env` file with `REACT_APP_COMPANY_NAME`
3. Frontend Production: Set `REACT_APP_COMPANY_NAME` before building

## Notes

- Frontend environment variables are baked into the build at build time. If you change the company name, you'll need to rebuild the frontend.
- Backend environment variables are read at runtime and can be changed without rebuilding.