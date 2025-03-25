import React, { createContext, useReducer, useContext, useRef } from 'react';

// Initial state
const initialState = {
  testResults: null,
  loading: false,
  error: null,
  customTestFile: '',
  showComparison: true,
  uploadedFile: null,
  uploadError: null,
  testMode: 'file', // 'file' or 'prompt'
  singlePrompt: '',
  expectedAnswer: '',
  
  // New fields for long-running jobs
  testJobId: null,
  jobStatus: null,
  jobProgress: 0,
  statusMessage: null,
  pollInterval: null
};

// Action types
export const ACTIONS = {
  SET_TEST_RESULTS: 'SET_TEST_RESULTS',
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  SET_CUSTOM_TEST_FILE: 'SET_CUSTOM_TEST_FILE',
  SET_SHOW_COMPARISON: 'SET_SHOW_COMPARISON',
  SET_UPLOADED_FILE: 'SET_UPLOADED_FILE',
  SET_UPLOAD_ERROR: 'SET_UPLOAD_ERROR',
  SET_TEST_MODE: 'SET_TEST_MODE',
  SET_SINGLE_PROMPT: 'SET_SINGLE_PROMPT',
  SET_EXPECTED_ANSWER: 'SET_EXPECTED_ANSWER',
  RESET_UPLOAD_STATE: 'RESET_UPLOAD_STATE',
  
  // New actions for long-running jobs
  SET_TEST_JOB_ID: 'SET_TEST_JOB_ID',
  SET_JOB_STATUS: 'SET_JOB_STATUS',
  SET_JOB_PROGRESS: 'SET_JOB_PROGRESS',
  SET_STATUS_MESSAGE: 'SET_STATUS_MESSAGE',
  SET_POLL_INTERVAL: 'SET_POLL_INTERVAL',
  
  // Special action to get state
  GET_STATE: 'GET_STATE'
};

// Reducer for state management
const testingReducer = (state, action) => {
  switch (action.type) {
    case ACTIONS.SET_TEST_RESULTS:
      return { ...state, testResults: action.payload };
    
    case ACTIONS.SET_LOADING:
      return { ...state, loading: action.payload };
    
    case ACTIONS.SET_ERROR:
      return { ...state, error: action.payload };
    
    case ACTIONS.SET_CUSTOM_TEST_FILE:
      return { ...state, customTestFile: action.payload };
    
    case ACTIONS.SET_SHOW_COMPARISON:
      return { ...state, showComparison: action.payload };
    
    case ACTIONS.SET_UPLOADED_FILE:
      return { ...state, uploadedFile: action.payload };
    
    case ACTIONS.SET_UPLOAD_ERROR:
      return { ...state, uploadError: action.payload };
    
    case ACTIONS.SET_TEST_MODE:
      return { ...state, testMode: action.payload };
    
    case ACTIONS.SET_SINGLE_PROMPT:
      return { ...state, singlePrompt: action.payload };
    
    case ACTIONS.SET_EXPECTED_ANSWER:
      return { ...state, expectedAnswer: action.payload };
    
    case ACTIONS.RESET_UPLOAD_STATE:
      return { 
        ...state, 
        uploadedFile: null, 
        uploadError: null,
        customTestFile: ''
      };
      
    // New actions for long-running jobs
    case ACTIONS.SET_TEST_JOB_ID:
      return { ...state, testJobId: action.payload };
      
    case ACTIONS.SET_JOB_STATUS:
      return { ...state, jobStatus: action.payload };
      
    case ACTIONS.SET_JOB_PROGRESS:
      return { ...state, jobProgress: action.payload };
      
    case ACTIONS.SET_STATUS_MESSAGE:
      return { ...state, statusMessage: action.payload };
      
    case ACTIONS.SET_POLL_INTERVAL:
      return { ...state, pollInterval: action.payload };
      
    case ACTIONS.GET_STATE:
      // Special action that calls the provided callback with the current state
      if (typeof action.payload === 'function') {
        action.payload(state);
      }
      return state;
    
    default:
      return state;
  }
};

// Create the context
export const TestingContext = createContext();

// Provider component
export const TestingProvider = ({ children }) => {
  const [state, dispatch] = useReducer(testingReducer, initialState);
  const fileInputRef = useRef(null);

  return (
    <TestingContext.Provider value={{ state, dispatch, fileInputRef }}>
      {children}
    </TestingContext.Provider>
  );
};

// Custom hook for using the context
export const useTestingContext = () => {
  const context = useContext(TestingContext);
  if (!context) {
    throw new Error('useTestingContext must be used within a TestingProvider');
  }
  return context;
};