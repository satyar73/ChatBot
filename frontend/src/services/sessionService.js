/**
 * Service for managing chat sessions in the frontend
 */
import { v4 as uuidv4 } from 'uuid';
import { chatApi } from './api';

/**
 * Get the current session ID or create a new one if none exists
 * Checks URL parameters first, then localStorage, then generates a new ID
 * @returns {string} The session ID
 */
export const getSessionId = () => {
  // Check for session ID in URL params first
  const params = new URLSearchParams(window.location.search);
  const paramSessionId = params.get('session');
  
  if (paramSessionId) {
    localStorage.setItem('chatSessionId', paramSessionId);
    return paramSessionId;
  }
  
  // Check for existing session ID in localStorage
  const savedSessionId = localStorage.getItem('chatSessionId');
  if (savedSessionId) {
    return savedSessionId;
  }
  
  // Generate new session ID if none exists
  const newSessionId = uuidv4();
  localStorage.setItem('chatSessionId', newSessionId);
  return newSessionId;
};

/**
 * Load the history for a specific session
 * @param {string} sessionId - The session ID to load
 * @param {string} mode - Optional mode to filter messages by (rag, standard, needl, compare)
 * @returns {Promise<Object>} - The session data including messages and metadata
 */
export const loadSessionHistory = async (sessionId, mode = null) => {
  try {
    // Build query parameters if mode is specified
    const params = new URLSearchParams();
    if (mode) {
      // Map UI mode names to API mode names if needed
      let apiMode = mode;
      if (mode === 'standard') apiMode = 'no_rag';
      
      params.append('mode', apiMode);
    }
    
    const queryString = params.toString() ? `?${params.toString()}` : '';
    console.log(`Loading session from API: /chat/sessions/${sessionId}${queryString}`);
    const response = await chatApi.getSession(sessionId, mode);
    
    // Log the response data for debugging
    console.log(`Received session data for mode ${mode}:`, response);
    console.log(`Messages count: ${response.messages?.length || 0}`);
    if (response.messages && response.messages.length > 0) {
      console.log(`First message:`, response.messages[0]);
      
      // Count message types
      const userCount = response.messages.filter(m => m.role === 'user').length;
      const assistantCount = response.messages.filter(m => m.role === 'assistant').length;
      console.log(`Message types - User: ${userCount}, Assistant: ${assistantCount}`);
      
      // Check for response_type attribute
      const withResponseType = response.messages.filter(m => m.response_type).length;
      console.log(`Messages with response_type: ${withResponseType}`);
    }
    
    return response;
  } catch (error) {
    console.error('Failed to load session history:', error);
    return { messages: [], metadata: {} };
  }
};

/**
 * List available sessions with pagination
 * @param {number} page - Page number (1-indexed)
 * @param {number} pageSize - Number of sessions per page
 * @returns {Promise<Object>} - The session list response
 */
export const listSessions = async (page = 1, pageSize = 10) => {
  try {
    const response = await chatApi.listSessions(page, pageSize);
    return response;
  } catch (error) {
    console.error('Failed to list sessions:', error);
    return { sessions: [], total_count: 0, page: 1, page_size: pageSize };
  }
};

/**
 * Filter messages by response type/mode
 * @param {Array} messages - List of messages to filter
 * @param {string} mode - Mode to filter by (rag, no_rag, needl, etc.)
 * @returns {Array} - Filtered messages
 */
export const filterMessagesByMode = (messages, mode) => {
  if (!mode || mode === 'all') return messages;
  
  return messages.filter(msg => 
    // Always include user messages
    (msg.role === 'user') || 
    // For assistant messages, check if response_type matches or isn't specified
    (msg.role === 'assistant' && 
     (!msg.response_type || msg.response_type === mode))
  );
};

/**
 * Add a tag to a session
 * @param {string} sessionId - The session ID to tag
 * @param {string} tag - The tag to add
 * @returns {Promise<Object>} - The response
 */
export const addTagToSession = async (sessionId, tag) => {
  try {
    const response = await chatApi.addSessionTag(sessionId, tag);
    return response;
  } catch (error) {
    console.error(`Failed to add tag '${tag}' to session:`, error);
    throw error;
  }
};

/**
 * Remove a tag from a session
 * @param {string} sessionId - The session ID to modify
 * @param {string} tag - The tag to remove
 * @returns {Promise<Object>} - The response
 */
export const removeTagFromSession = async (sessionId, tag) => {
  try {
    const response = await chatApi.removeSessionTag(sessionId, tag);
    return response;
  } catch (error) {
    console.error(`Failed to remove tag '${tag}' from session:`, error);
    throw error;
  }
};

export default {
  getSessionId,
  loadSessionHistory,
  listSessions,
  filterMessagesByMode,
  addTagToSession,
  removeTagFromSession
};