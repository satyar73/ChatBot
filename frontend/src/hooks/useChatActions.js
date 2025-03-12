import { useCallback } from 'react';
import { chatApi } from '../services/api';
import { useChatContext, ACTIONS } from '../context/ChatContext';

/**
 * Custom hook that provides actions for the chat interface
 */
const useChatActions = () => {
  const { state, dispatch } = useChatContext();
  
  /**
   * Set the input value for the chat
   */
  const setInput = useCallback((value) => {
    dispatch({ type: ACTIONS.SET_INPUT, payload: value });
  }, [dispatch]);
  
  /**
   * Set the response mode (rag, standard, compare)
   */
  const setResponseMode = useCallback((mode) => {
    dispatch({ type: ACTIONS.SET_RESPONSE_MODE, payload: mode });
  }, [dispatch]);
  
  /**
   * Clear the error message
   */
  const clearError = useCallback(() => {
    dispatch({ type: ACTIONS.SET_ERROR, payload: null });
  }, [dispatch]);
  
  /**
   * Clear the chat history and session ID
   */
  const clearChat = useCallback(() => {
    dispatch({ type: ACTIONS.CLEAR_CHAT });
  }, [dispatch]);
  
  /**
   * Process responses from the API - guaranteed to be in the format:
   * response.response.output for RAG
   * response.response.no_rag_output for non-RAG/standard
   */
  const processResponses = useCallback((response) => {
    let ragResponse = null;
    let standardResponse = null;
    
    if (response.response) {
      ragResponse = response.response.output || null;
      standardResponse = response.response.no_rag_output || null;
    }
    
    // Ensure we have strings for display
    if (ragResponse && typeof ragResponse !== 'string') {
      ragResponse = JSON.stringify(ragResponse, null, 2);
    }
    
    if (standardResponse && typeof standardResponse !== 'string') {
      standardResponse = JSON.stringify(standardResponse, null, 2);
    }
    
    return { ragResponse, standardResponse };
  }, []);
  
  /**
   * Send a message to the chat API
   */
  const sendMessage = useCallback(async (e) => {
    e.preventDefault();
    
    if (!state.input.trim()) return;
    
    // Add user message to chat
    dispatch({ type: ACTIONS.ADD_USER_MESSAGE, payload: state.input });
    dispatch({ type: ACTIONS.SET_INPUT, payload: '' });
    dispatch({ type: ACTIONS.SET_LOADING, payload: true });
    dispatch({ type: ACTIONS.SET_ERROR, payload: null });
    
    try {
      // Send message to API with the current session ID and response mode
      const response = await chatApi.sendMessage(state.input, state.responseMode, state.sessionId);
      
      // Save or update the session ID for future requests
      if (response.session_id) {
        dispatch({ type: ACTIONS.SET_SESSION_ID, payload: response.session_id });
      }
      
      // Process the response data
      const { ragResponse, standardResponse } = processResponses(response);
      
      // Add responses based on the current display mode
      if (state.responseMode === "compare") {
        // Show both responses side by side
        const assistantMessages = [];
        
        if (ragResponse) {
          assistantMessages.push({ 
            role: 'assistant', 
            content: ragResponse, 
            type: 'rag' 
          });
        }
        
        if (standardResponse) {
          assistantMessages.push({ 
            role: 'assistant', 
            content: standardResponse, 
            type: 'standard' 
          });
        }
        
        if (assistantMessages.length > 0) {
          dispatch({ 
            type: ACTIONS.ADD_ASSISTANT_MESSAGES, 
            payload: assistantMessages 
          });
        }
      } else if (state.responseMode === "rag") {
        // Only show RAG response but store both if available
        dispatch({ 
          type: ACTIONS.ADD_ASSISTANT_MESSAGE, 
          payload: { 
            role: 'assistant', 
            content: ragResponse || "No RAG response available",
            hiddenContent: standardResponse,
            originalMode: state.responseMode 
          }
        });
      } else {
        // Only show standard response but store both if available
        dispatch({ 
          type: ACTIONS.ADD_ASSISTANT_MESSAGE, 
          payload: { 
            role: 'assistant', 
            content: standardResponse || "No standard response available",
            hiddenContent: ragResponse,
            originalMode: state.responseMode
          }
        });
      }
    } catch (error) {
      console.error('Chat error:', error);
      
      dispatch({ 
        type: ACTIONS.SET_ERROR, 
        payload: error.message || 'Failed to send message' 
      });
      
      dispatch({ 
        type: ACTIONS.ADD_ASSISTANT_MESSAGE, 
        payload: { 
          role: 'assistant', 
          content: 'Sorry, there was an error processing your request. ' + 
                  (error.response?.data?.detail || error.message || '')
        }
      });
    } finally {
      dispatch({ type: ACTIONS.SET_LOADING, payload: false });
    }
  }, [state.input, state.responseMode, state.sessionId, dispatch, processResponses]);
  
  return {
    sendMessage,
    setInput,
    setResponseMode,
    clearError,
    clearChat
  };
};

export default useChatActions;