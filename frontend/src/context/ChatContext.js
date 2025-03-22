import React, { createContext, useReducer, useEffect, useRef, useContext } from 'react';

// Initial state for the chat context
const initialState = {
  messages: [],
  input: '',
  loading: false,
  responseMode: 'rag', // Three modes: "rag", "standard", "compare"
  promptStyle: 'default', // Three styles: "default", "detailed", "concise"
  sessionId: null,
  error: null,
  systemPrompt: '',  // Store custom system prompt
  showSystemPrompt: false  // Control visibility of system prompt editor
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
  TOGGLE_SYSTEM_PROMPT: 'TOGGLE_SYSTEM_PROMPT'
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

  // Reformat message display when response mode changes
  useEffect(() => {
    const reformatMessages = () => {
      // Create a new message array to avoid mutation
      const newMessages = [];

      // Process each message based on response mode
      for (const msg of state.messages) {
        // Skip null/undefined messages
        if (!msg) continue;
        
        // Always keep user messages unchanged
        if (msg.role === 'user') {
          newMessages.push(msg);
          continue;
        }
        
        // Handle assistant messages based on display mode
        if (msg.role === 'assistant') {
          // CASE 1: Compare mode - messages with type already set
          if (state.responseMode === 'compare') {
            // If message already has a type, keep it as is
            if (msg.type === 'rag' || msg.type === 'standard') {
              newMessages.push(msg);
              continue;
            }

            // This means that the initial call was either RAG or STANDARD mode
            // If message has hidden content, split into two messages
            if (msg.hiddenContent) {
              if (msg.originalMode === 'rag') {
                // RAG content is primary, standard is hidden
                newMessages.push({
                          role: 'assistant',
                          content: msg.content,
                          hiddenContent: msg.hiddenContent,
                          type: 'rag' });
                newMessages.push({
                          role: 'assistant',
                          content: msg.hiddenContent,
                          hiddenContent: msg.content,
                          type: 'standard' });
              } else {
                // Standard content is primary, RAG is hidden
                newMessages.push({
                          role: 'assistant',
                          content: msg.hiddenContent,
                          hiddenContent: msg.content,
                          type: 'rag' });
                newMessages.push({
                          role: 'assistant',
                          content: msg.content,
                          hiddenContent: msg.hiddenContent,
                          type: 'standard' });
              }
              continue;
            }

            console.assert(false, 'Message should have hidden content');
            // If no hidden content, just use existing content (single mode message)
            newMessages.push({ ...msg, type: 'standard' });
            continue;
          }
          
          // CASE 2: RAG mode
          if (state.responseMode === 'rag') {
            if (msg.type) {
              // If it's a typed message in compare mode, only keep RAG messages
              if (msg.type === 'rag') {
                newMessages.push({
                    role: 'assistant',
                    content: msg.content,
                    hiddenContent: msg.hiddenContent,
                    originalMode: 'rag'
                 });
              }
              continue;
            }

            // it is a typed message however not in compare mode
            // If it has hiddenContent, use the right content based on originalMode
            // always convert it to RAG type message
            if (msg.hiddenContent) {
              if (msg.originalMode === 'rag') {
                // RAG is already the primary content
                newMessages.push({ 
                  role: 'assistant', 
                  content: msg.content,
                  hiddenContent: msg.hiddenContent,
                  originalMode: 'rag'
                });
              } else {
                // Standard is primary, RAG is hidden, so swap
                newMessages.push({ 
                  role: 'assistant', 
                  content: msg.hiddenContent,
                  hiddenContent: msg.content,
                  originalMode: 'rag'
                });
              }
              continue;
            }

            console.assert(false, 'Message should have hidden content');
            // No type or hidden content, keep as is
            newMessages.push(msg);
            continue;
          }
          
          // CASE 3: Standard mode
          if (state.responseMode === 'standard') {
            if (msg.type) {
              // If it's a typed message in compare mode, only keep standard messages
              if (msg.type === 'standard') {
                newMessages.push({
                    role: 'assistant',
                    content: msg.content,
                    hiddenContent: msg.hiddenContent,
                    originalMode: 'standard'
                });
              }
              continue;
            }

            // it is a typed message however not in compare mode
            // If it has hiddenContent, use the right content based on originalMode
            // always convert it to standard type message
            if (msg.hiddenContent) {
              if (msg.originalMode === 'standard') {
                // Standard is already the primary content
                newMessages.push({ 
                  role: 'assistant', 
                  content: msg.content,
                  hiddenContent: msg.hiddenContent,
                  originalMode: 'standard'
                });
              } else {
                // RAG is primary, standard is hidden, so swap
                newMessages.push({ 
                  role: 'assistant', 
                  content: msg.hiddenContent,
                  hiddenContent: msg.content,
                  originalMode: 'standard'
                });
              }
              continue;
            }

            console.assert(false, "Message should have hidden content");
            // No type or hidden content, keep as is
            newMessages.push(msg);
          }
        } else {
          // Any other role, just keep it
          console.assert(false, 'Unknown role');
          newMessages.push(msg);
        }
      }
      
      dispatch({ type: ACTIONS.REFRESH_MESSAGES, payload: newMessages });
    };

    reformatMessages();
  }, [state.responseMode]);

  return (
    <ChatContext.Provider value={{ state, dispatch, chatEndRef }}>
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