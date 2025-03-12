# ChatBot Example Frontend

This is a React frontend for the RAG-enabled Marketing ChatBot application. It provides interfaces for chatting, testing, and managing document indexing.

## Features
- **Chat Interface**: Modern chat UI with support for both RAG and standard responses
- **Response Mode Switching**: Toggle between RAG, standard, and comparison views
- **Response Memory**: Preserves both responses for retrospective comparison
- **Testing Tool**: Run and analyze test scenarios to compare RAG vs non-RAG performance
- **Google Drive Indexing**: Interface for adding documents from Google Drive to the knowledge base
- **Shopify Indexing**: Interface for adding products and articles from Shopify to the knowledge base

## Setup and Installation
1. Install dependencies:
   ```
   cd frontend
   npm install
   ```

2. Start the development server:
   ```
   npm start
   ```

3. The application will be available at http://localhost:3000

## Configuration
The application is configured to proxy API requests to http://localhost:8005 where the backend should be running. You can modify this in the `package.json` file if your backend is running on a different port.

## Pages
- **Chat**: Main chat interface with the marketing bot
  - Switch between RAG, Standard, and Compare modes
  - View both knowledge-enhanced and standard AI responses
  - Messages retain both response types for later comparison
- **Test Chat**: Run test scenarios and analyze results
  - Run either a single prompt or an automated test suites against semantic benchmarks
  - Compare RAG vs standard model performance metrics
  - View detailed side-by-side response comparisons
- **Google Drive Indexing**: Manage Google Drive document indexing
  - Index documents from specified folders
  - View and manage indexed document stats
- **Shopify Indexing**: Manage Shopify content indexing
  - Index products and blog posts
  - Filter and manage indexed content by type

## Technologies Used
- React.js
- Material UI for components
- Axios for API requests
- React Router for navigation

## Architecture
The application follows a modular architecture with clear separation of concerns:

### 1. Context (State Management)
- Each feature has its own context provider (`ChatContext`, `GoogleDriveContext`, etc.)
- Contexts use the reducer pattern for predictable state updates
- Actions are defined as constants for consistent state changes

### 2. Custom Hooks (Business Logic)
- Feature-specific hooks encapsulate business logic (`useChatActions`, `useGoogleDriveActions`, etc.)
- API interactions and complex logic are isolated from UI components
- Hooks use React's `useCallback` for optimized performance

### 3. UI Components (Presentation)
- Components are focused on presentation with minimal logic
- Each component has a single responsibility
- Components receive data and callbacks via props or context

### 4. Pages (Composition)
- Pages act as composition layers, bringing together components
- Context providers wrap page content
- Features are modular and independently maintainable

## Development Notes
- The application uses React's context API for state management
- Material UI's theming system is used for styling
- The proxy setting in package.json handles CORS issues during development
- The codebase follows the principle of separation of concerns

## Response Mode Handling
The chat interface provides three distinct modes for viewing AI responses:
1. **RAG Mode**: Shows only responses enhanced with the knowledge base
2. **Standard Mode**: Shows only the base AI responses without knowledge enhancement
3. **Compare Mode**: Shows both responses side-by-side for direct comparison

The interface maintains both response types internally, allowing you to switch views even after receiving responses. This is useful for analyzing how much the RAG process improves or changes responses.

## Session Management
- Each conversation has a unique session ID
- The backend maintains conversation history for each session
- Session-specific context helps improve response relevance
- The backend cache service can optionally consider session context when caching

## Error Handling

The frontend includes comprehensive error handling for API issues:
- Detailed error messages for failed requests
- Automatic session recovery when possible
- Graceful fallbacks for backend service issues

## Directory Structure

```
src/
├── components/        # UI components
│   ├── indexing/      # Components for indexing interfaces
│   ├── testing/       # Components for testing interface
│   └── ...            # Chat interface components
├── context/           # Context providers for state management
│   ├── indexing/      # Indexing-specific contexts
│   ├── TestingContext.js
│   └── ChatContext.js
├── hooks/             # Custom hooks for business logic
│   ├── indexing/      # Indexing-specific hooks
│   ├── useTestingActions.js
│   └── useChatActions.js
├── pages/             # Main page components
│   ├── ChatPage.js
│   ├── TestingPage.js
│   ├── GoogleDriveIndexingPage.js
│   └── ShopifyIndexingPage.js
└── services/          # API services
    └── api.js         # API client and endpoints
```

This structure follows best practices for React application organization, making the codebase more maintainable, testable, and scalable.