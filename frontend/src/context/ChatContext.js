import React, { createContext, useReducer, useEffect, useRef, useContext, useState } from 'react';
import { getSessionId, loadSessionHistory } from '../services/sessionService';
import { chatApi } from '../services/api';

// Initial state for the chat context
const initialState = {
  messages: [],
  input: '',
  loading: false,
  responseMode: 'rag', // Three modes: "rag", "standard", "needl" (needl uses direct API responses)
  promptStyle: 'default', // Three styles: "default", "detailed", "concise"
  sessionId: null,
  error: null,
  systemPrompt: '',  // Store custom system prompt
  showSystemPrompt: false,  // Control visibility of system prompt editor
  isLoadingHistory: false,  // Flag for session history loading
  availableSessions: [],   // List of available sessions
  sessionMetadata: {}      // Metadata for the current session
};

// Action types
export const ACTIONS = {
  SET_INPUT: 'SET_INPUT',
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  SET_RESPONSE_MODE: 'SET_RESPONSE_MODE',
  SET_PROMPT_STYLE: 'SET_PROMPT_STYLE',
  ADD_USER_MESSAGE: 'ADD_USER_MESSAGE',
  ADD_ASSISTANT_MESSAGE: 'ADD_ASSISTANT_MESSAGE',
  ADD_ASSISTANT_MESSAGES: 'ADD_ASSISTANT_MESSAGES',
  SET_SESSION_ID: 'SET_SESSION_ID',
  CLEAR_CHAT: 'CLEAR_CHAT',
  REFRESH_MESSAGES: 'REFRESH_MESSAGES',
  SET_SYSTEM_PROMPT: 'SET_SYSTEM_PROMPT',
  TOGGLE_SYSTEM_PROMPT: 'TOGGLE_SYSTEM_PROMPT',
  
  // Session management actions
  SET_LOADING_HISTORY: 'SET_LOADING_HISTORY',
  SET_AVAILABLE_SESSIONS: 'SET_AVAILABLE_SESSIONS',
  SET_SESSION_METADATA: 'SET_SESSION_METADATA',
  ADD_SESSION_TAG: 'ADD_SESSION_TAG',
  REMOVE_SESSION_TAG: 'REMOVE_SESSION_TAG'
};

// Reducer for handling state updates
const chatReducer = (state, action) => {
  switch (action.type) {
    case ACTIONS.SET_INPUT:
      return { ...state, input: action.payload };
    
    case ACTIONS.SET_LOADING:
      return { ...state, loading: action.payload };
    
    case ACTIONS.SET_ERROR:
      return { ...state, error: action.payload };
    
    case ACTIONS.SET_RESPONSE_MODE:
      return { ...state, responseMode: action.payload };
      
    case ACTIONS.SET_PROMPT_STYLE:
      return { ...state, promptStyle: action.payload };
    
    case ACTIONS.ADD_USER_MESSAGE:
      return {
        ...state,
        messages: [...state.messages, { role: 'user', content: action.payload }]
      };
    
    case ACTIONS.ADD_ASSISTANT_MESSAGE:
      return {
        ...state,
        messages: [...state.messages, action.payload]
      };
    
    case ACTIONS.ADD_ASSISTANT_MESSAGES:
      return {
        ...state,
        messages: [...state.messages, ...action.payload]
      };
    
    case ACTIONS.SET_SESSION_ID:
      return { ...state, sessionId: action.payload };
    
    case ACTIONS.CLEAR_CHAT:
      return { ...state, messages: [], sessionId: null };
    
    case ACTIONS.REFRESH_MESSAGES:
      return { ...state, messages: action.payload };
      
    case ACTIONS.SET_SYSTEM_PROMPT:
      return { ...state, systemPrompt: action.payload };
      
    case ACTIONS.TOGGLE_SYSTEM_PROMPT:
      return { ...state, showSystemPrompt: !state.showSystemPrompt };
      
    // Session management actions
    case ACTIONS.SET_LOADING_HISTORY:
      return { ...state, isLoadingHistory: action.payload };
      
    case ACTIONS.SET_AVAILABLE_SESSIONS:
      return { ...state, availableSessions: action.payload };
      
    case ACTIONS.SET_SESSION_METADATA:
      return { ...state, sessionMetadata: action.payload };
      
    case ACTIONS.ADD_SESSION_TAG:
      return {
        ...state,
        sessionMetadata: {
          ...state.sessionMetadata,
          tags: [...(state.sessionMetadata.tags || []), action.payload]
        }
      };
      
    case ACTIONS.REMOVE_SESSION_TAG:
      return {
        ...state,
        sessionMetadata: {
          ...state.sessionMetadata,
          tags: (state.sessionMetadata.tags || []).filter(tag => tag !== action.payload)
        }
      };
    
    default:
      return state;
  }
};

// Create the context
export const ChatContext = createContext();

// Provider component
export const ChatProvider = ({ children }) => {
  const [state, dispatch] = useReducer(chatReducer, initialState);
  const chatEndRef = useRef(null);

  // Scroll to bottom when messages update
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [state.messages]);
  
  // Initialize session ID and load session history
  useEffect(() => {
    const initializeSession = async () => {
      try {
        // Get session ID from URL/localStorage or generate new one
        const sessionId = getSessionId();
        dispatch({ type: ACTIONS.SET_SESSION_ID, payload: sessionId });
        
        // Set loading state
        dispatch({ type: ACTIONS.SET_LOADING_HISTORY, payload: true });
        
        // Load session history if there's a session ID
        if (sessionId) {
          try {
            const sessionData = await loadSessionHistory(sessionId);
            
            // Update messages if there are any
            if (sessionData.messages && sessionData.messages.length > 0) {
              dispatch({ type: ACTIONS.REFRESH_MESSAGES, payload: sessionData.messages });
            }
            
            // Update session metadata
            if (sessionData.metadata) {
              dispatch({ type: ACTIONS.SET_SESSION_METADATA, payload: sessionData.metadata });
            }
          } catch (error) {
            console.error('Error loading session history:', error);
            dispatch({ type: ACTIONS.SET_ERROR, payload: 'Failed to load session history' });
          }
        }
      } catch (error) {
        console.error('Error initializing session:', error);
      } finally {
        dispatch({ type: ACTIONS.SET_LOADING_HISTORY, payload: false });
      }
    };

    initializeSession();
  }, []);
  
  // Load available sessions
  useEffect(() => {
    const loadAvailableSessions = async () => {
      try {
        const result = await chatApi.listSessions(1, 10);
        if (result && result.sessions) {
          dispatch({ type: ACTIONS.SET_AVAILABLE_SESSIONS, payload: result.sessions });
        }
      } catch (error) {
        console.error('Error loading available sessions:', error);
      }
    };
    
    loadAvailableSessions();
  }, []);

  // Load mode-specific messages when the response mode changes
  useEffect(() => {
    console.log('responseMode useEffect triggered:', { 
      mode: state.responseMode, 
      sessionId: state.sessionId, 
      isLoading: state.isLoadingHistory 
    });
    
    // Don't reload if we don't have a session ID yet or if we're already loading
    if (!state.sessionId || state.isLoadingHistory) return;
    
    const loadModeSpecificMessages = async () => {
      // Set loading state
      dispatch({ type: ACTIONS.SET_LOADING_HISTORY, payload: true });
      
      try {
        console.log('Loading session history for mode:', state.responseMode);
        // Load session with the current mode
        const sessionData = await loadSessionHistory(state.sessionId, state.responseMode);
        console.log('Session data loaded:', sessionData);
        
        // Always update messages, even with empty array
        // This ensures we clear messages when there aren't any for the current mode
        const messages = sessionData.messages || [];
        console.log(`Mode ${state.responseMode} returned ${messages.length} messages`);
        dispatch({ type: ACTIONS.REFRESH_MESSAGES, payload: messages });
      } catch (error) {
        console.error('Error loading mode-specific messages:', error);
      } finally {
        dispatch({ type: ACTIONS.SET_LOADING_HISTORY, payload: false });
      }
    };
    
    // Load mode-filtered messages
    loadModeSpecificMessages();
  }, [state.responseMode, state.sessionId, dispatch]);
  
  // Add session management functions
  const switchSession = async (newSessionId) => {
    try {
      // Set loading state
      dispatch({ type: ACTIONS.SET_LOADING_HISTORY, payload: true });
      dispatch({ type: ACTIONS.SET_ERROR, payload: null });
      
      // Update localStorage
      localStorage.setItem('chatSessionId', newSessionId);
      
      // Update session ID in state
      dispatch({ type: ACTIONS.SET_SESSION_ID, payload: newSessionId });
      
      // Clear existing messages
      dispatch({ type: ACTIONS.REFRESH_MESSAGES, payload: [] });
      
      // Load new session data
      try {
        const sessionData = await loadSessionHistory(newSessionId);
        
        // Update messages if there are any
        if (sessionData.messages && sessionData.messages.length > 0) {
          dispatch({ type: ACTIONS.REFRESH_MESSAGES, payload: sessionData.messages });
        }
        
        // Update session metadata
        if (sessionData.metadata) {
          dispatch({ type: ACTIONS.SET_SESSION_METADATA, payload: sessionData.metadata });
        }
      } catch (error) {
        console.error(`Error loading session ${newSessionId}:`, error);
        dispatch({ type: ACTIONS.SET_ERROR, payload: 'Failed to load session' });
      }
    } finally {
      dispatch({ type: ACTIONS.SET_LOADING_HISTORY, payload: false });
    }
  };
  
  const createNewSession = () => {
    // Generate a new session ID
    const newSessionId = getSessionId();
    
    // Clear the existing chat
    dispatch({ type: ACTIONS.CLEAR_CHAT });
    
    // Set the new session ID
    dispatch({ type: ACTIONS.SET_SESSION_ID, payload: newSessionId });
    
    // Update localStorage
    localStorage.setItem('chatSessionId', newSessionId);
    
    return newSessionId;
  };
  
  const addSessionTag = async (tag) => {
    if (!state.sessionId) return;
    
    try {
      await chatApi.addSessionTag(state.sessionId, tag);
      dispatch({ type: ACTIONS.ADD_SESSION_TAG, payload: tag });
    } catch (error) {
      console.error(`Error adding tag to session ${state.sessionId}:`, error);
      dispatch({ type: ACTIONS.SET_ERROR, payload: 'Failed to add tag' });
    }
  };
  
  const removeSessionTag = async (tag) => {
    if (!state.sessionId) return;
    
    try {
      await chatApi.removeSessionTag(state.sessionId, tag);
      dispatch({ type: ACTIONS.REMOVE_SESSION_TAG, payload: tag });
    } catch (error) {
      console.error(`Error removing tag from session ${state.sessionId}:`, error);
      dispatch({ type: ACTIONS.SET_ERROR, payload: 'Failed to remove tag' });
    }
  };
  
  const refreshSessionList = async () => {
    try {
      const result = await chatApi.listSessions(1, 10);
      if (result && result.sessions) {
        dispatch({ type: ACTIONS.SET_AVAILABLE_SESSIONS, payload: result.sessions });
      }
    } catch (error) {
      console.error('Error refreshing session list:', error);
    }
  };

  return (
    <ChatContext.Provider 
      value={{ 
        state, 
        dispatch, 
        chatEndRef,
        sessionManagement: {
          switchSession,
          createNewSession,
          addSessionTag,
          removeSessionTag,
          refreshSessionList
        }
      }}
    >
      {children}
    </ChatContext.Provider>
  );
};

// Custom hook for using the chat context
export const useChatContext = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChatContext must be used within a ChatProvider');
  }
  return context;
};