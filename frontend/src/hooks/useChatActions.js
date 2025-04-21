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
    console.log('Mode switching to:', mode);
    // Only change mode if it's different from current mode
    if (state.responseMode !== mode) {
      dispatch({ type: ACTIONS.SET_RESPONSE_MODE, payload: mode });
    } else {
      console.log('Mode already set to', mode, '- skipping update');
    }
  }, [dispatch, state.responseMode]);
  
  /**
   * Set the prompt style (default, detailed, concise)
   */
  const setPromptStyle = useCallback((style) => {
    dispatch({ type: ACTIONS.SET_PROMPT_STYLE, payload: style });
  }, [dispatch]);
  
  /**
   * Set the system prompt
   */
  const setSystemPrompt = useCallback((prompt) => {
    dispatch({ type: ACTIONS.SET_SYSTEM_PROMPT, payload: prompt });
  }, [dispatch]);
  
  /**
   * Toggle the system prompt editor visibility
   */
  const toggleSystemPrompt = useCallback(() => {
    dispatch({ type: ACTIONS.TOGGLE_SYSTEM_PROMPT });
  }, [dispatch]);
  
  /**
   * Clear the error message
   */
  const clearError = useCallback(() => {
    dispatch({ type: ACTIONS.SET_ERROR, payload: null });
  }, [dispatch]);
  
  /**
   * Clear the chat history while preserving the session ID
   */
  const clearChat = useCallback(() => {
    if (window.confirm('Clear the current conversation? The session will remain available but messages will be cleared.')) {
      dispatch({ type: ACTIONS.REFRESH_MESSAGES, payload: [] });
    }
  }, [dispatch]);
  
  /**
   * Process responses from the API - simplified to work with just the output field.
   * The backend now returns a single response in the output field based on the mode.
   */
  const processResponses = useCallback((response) => {
    let output = null;
    let mode = null;
    
    if (response.response) {
      output = response.response.output || null;
      mode = response.response.mode || state.responseMode;
    }
    
    // Ensure we have a string for display
    if (output && typeof output !== 'string') {
      output = JSON.stringify(output, null, 2);
    }
    
    return { output, mode };
  }, [state.responseMode]);
  
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
      // Send message to API with the current session ID, response mode, system prompt, and prompt style
      const response = await chatApi.sendMessage(
        state.input, 
        state.responseMode, 
        state.sessionId,
        state.systemPrompt || null,
        state.promptStyle || "default"
      );
      
      // Save or update the session ID for future requests
      if (response.session_id) {
        dispatch({ type: ACTIONS.SET_SESSION_ID, payload: response.session_id });
      }
      
      // Process the response data
      const { output, mode } = processResponses(response);
      
      // Add responses based on the current display mode

      // Just use role: 'assistant' and add the text, no other flags or data
      if (state.responseMode === "needl") {
        // For Needl mode, display both the response and sources
        const content = output || "No Needl response available";
        
        // Include sources in the message if available
        const sources = response.sources || [];
        const sourceText = sources.length > 0 ? 
          "\n\n**Sources:**\n" + sources.map(s => `- [${s.title}](${s.url})`).join("\n") : 
          "";
          
        dispatch({
          type: ACTIONS.ADD_ASSISTANT_MESSAGE,
          payload: {
            role: 'assistant',
            content: content + sourceText,
            response_type: 'needl',  // Explicitly tag with response_type
            isNeedlResponse: true
          }
        });
      } else if (state.responseMode === "rag") {
        // Show RAG response
        dispatch({ 
          type: ACTIONS.ADD_ASSISTANT_MESSAGE, 
          payload: { 
            role: 'assistant', 
            content: output || "No RAG response available",
            originalMode: state.responseMode,
            response_type: "rag"  // Explicitly tag with response_type for filtering
          }
        });
      } else {
        // Show standard response
        dispatch({ 
          type: ACTIONS.ADD_ASSISTANT_MESSAGE, 
          payload: { 
            role: 'assistant', 
            content: output || "No standard response available",
            originalMode: state.responseMode,
            response_type: "no_rag"  // Explicitly tag with response_type for filtering
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
    setPromptStyle,
    setSystemPrompt,
    toggleSystemPrompt,
    clearError,
    clearChat
  };
};

export default useChatActions;