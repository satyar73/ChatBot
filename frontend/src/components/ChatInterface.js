/**
 * ChatInterface Component - A dynamic chat application built with React and Material-UI.
 *
 * Key Features:
 * 1. **Dynamic Chat Messages**:
 *    - Chat messages (user and assistant) are stored in the `messages` state.
 *    - Messages are rendered dynamically, and the display adjusts based on message type (e.g., user, assistant, RAG, Standard).
 *
 * 2. **Response Modes**:
 *    - The chatbot supports three response modes:
 *      1. RAG (Retrieval-Augmented Generation): Advanced AI-generated responses.
 *      2. Standard: Basic responses without advanced augmentation.
 *      3. Compare: Displays both RAG and Standard responses side-by-side for comparison.
 *    - The mode can be toggled using buttons, and chat messages are dynamically updated for the chosen mode.
 *
 * 3. **React Tools (Hooks)**:
 *    - `useState`: Manages state for messages, user input, loading status, response modes, errors, etc.
 *    - `useEffect`: Automatically reacts to changes:
 *      - Scrolls the chat box to the bottom when new messages are added.
 *      - Updates the displayed messages when the response mode is changed.
 *    - `useRef`: References the bottom of the chat box for smooth scrolling.
 *
 * 4. **handleSendMessage Function**:
 *    - Triggered when the user sends a message.
 *    - Sends the user's message and the selected mode to the backend API (`chatApi.sendMessage`).
 *    - Processes and formats the chatbot's response before adding it to the chat.
 *    - Displays error messages if something goes wrong (e.g., network issues).
 *
 * 5. **Error Handling**:
 *    - Errors during communication with the API are caught and displayed as a popup notification (Snackbar).
 *
 * 6. **Chat Display**:
 *    - Messages are styled dynamically based on type:
 *      - User messages are right-aligned with a specific background color.
 *      - Assistant messages are categorized and visually differentiated for RAG, Standard, and Compare modes.
 *    - Circular loaders are displayed when waiting for responses.
 *
 * 7. **Mode-Specific Message Processing**:
 *    - When the mode is changed, the `useEffect` hook evaluates previous messages and adjusts the display.
 *    - It ensures the right content (RAG, Standard, or both) is displayed based on the selected mode formatting.
 *
 * 8. **Additional Features**:
 *    - Real-time updates ensure a smooth user experience.
 *    - A clear button allows users to reset the chat and start fresh.
 *
 * Overall, this component creates a flexible and robust chat interface that dynamically adapts to user interactions and different response modes.
 */

import React, { useState, useRef, useEffect } from 'react';
import { 
  Box, TextField, Button, Paper, Typography, Avatar, 
  Grid, IconButton, CircularProgress, FormControlLabel, Switch,
  Snackbar, Alert
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import DeleteIcon from '@mui/icons-material/Delete';
import ReactMarkdown from 'react-markdown';
import { chatApi } from '../services/api';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  // Three modes: "rag", "standard", "compare"
  const [responseMode, setResponseMode] = useState("rag");
  const [sessionId, setSessionId] = useState(null);
  const [error, setError] = useState(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  // Simplified mode change handler that rebuilds messages instead of trying to transform them
  useEffect(() => {
    // Get the current messages
    setMessages(prevMessages => {
      // Create a new message array to avoid mutation
      const newMessages = [];

      console.log("Processing messages for mode change:", prevMessages);
      // Process each message based on response mode
      for (const msg of prevMessages) {
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
          if (responseMode === 'compare') {
            // If message already has a type, keep it as is
            if (msg.type === 'rag' || msg.type === 'standard') {
              newMessages.push(msg);
              continue;
            }
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
                    hiddenContent: msg.content,
                    type: 'standard' });
              }
              continue;
            }

            console.assert(false, "Message should have hidden content");
            // If no hidden content, just use existing content (single mode message)
            newMessages.push({ ...msg, type: 'standard' });
            continue;
          }
          
          // CASE 2: RAG mode
          if (responseMode === 'rag') {
            // If it's a typed message in compare mode, only keep RAG messages
            if (msg.type) {
              if (msg.type === 'rag') {
                newMessages.push({
                  role: 'assistant',
                  content: msg.content,
                  hiddenContent: msg.hiddenContent,
                  originalMode: responseMode
                  });
              }
              // 'standard' nodes will get dropped
              continue;
            }
            
            // If it has hiddenContent, use the right content based on originalMode
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

            console.assert(false, "Message should have hidden content or a type");
            // No type or hidden content, keep as is
            newMessages.push(msg);
            continue;
          }
          
          // CASE 3: Standard mode
          if (responseMode === 'standard') {
            // If it's a typed message in compare mode, only keep standard messages
            if (msg.type) {
              if (msg.type === 'standard') {
                newMessages.push({
                  role: 'assistant',
                  content: msg.content,
                  hiddenContent: msg.hiddenContent,
                  originalMode: responseMode
                });
              }
              // in this case rag node will be dropped
              continue;
            }
            
            // If it has hiddenContent, use the right content based on originalMode
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

            console.assert(false, "Message should have hidden content or a type");
            // No type or hidden content, keep as is
            newMessages.push(msg);
          }
        } else {
          // Any other role, just keep it
          newMessages.push(msg);
        }
      }
      
      return newMessages;
    });
  }, [responseMode]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message to chat
    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      // Send message to API with the current session ID and response mode
      const response = await chatApi.sendMessage(input, responseMode, sessionId);
      
      // Save or update the session ID for future requests
      if (response.session_id) {
        setSessionId(response.session_id);
      }
      
      // Process response data - ensure all content is strings
      const processContent = (content) => {
        if (typeof content === 'string') return content;
        return JSON.stringify(content, null, 2);
      };
      
      // Extract RAG and standard responses from the JSON response
      let ragResponse = null;
      let standardResponse = null;
      
      console.log("Raw API response:", response);
      
      // Check if we have a response.response object (nested structure)
      if (response.response && typeof response.response === 'object') {
        // Primary output based on the selected mode
        ragResponse = response.response.output;
        standardResponse = response.response.no_rag_output;
      }
      else {
        // Fallback if no recognizable format
        ragResponse = "Could not extract RAG response from server data";
        standardResponse = "Could not extract standard response from server data";
      }
      
      // Make sure we have strings, not objects
      if (typeof ragResponse !== 'string') {
        ragResponse = processContent(ragResponse);
      }
      
      if (typeof standardResponse !== 'string') {
        standardResponse = processContent(standardResponse);
      }
      
      console.log("Extracted RAG response:", ragResponse);
      console.log("Extracted standard response:", standardResponse);
      
      // Add responses based on the current display mode
      setMessages(prevMessages => {
        // Create a new array with all existing messages
        const newMessages = [...prevMessages];
        
        // Add appropriate message(s) based on the display mode
        if (responseMode === "compare") {
          // Show both responses side by side
          if (ragResponse) {
            newMessages.push({
              role: 'assistant',
              content: ragResponse,
              hiddenContent: standardResponse || "No standard response available",
              type: 'rag' });
          }
          if (standardResponse) {
            newMessages.push({
              role: 'assistant',
              content: standardResponse,
              hiddenContent: ragResponse || "No RAG response available",
              type: 'standard' });
          }
        } else if (responseMode === "rag") {
          // Only show RAG response but store both if available
          newMessages.push({ 
            role: 'assistant', 
            content: ragResponse || "No RAG response available",
            hiddenContent: standardResponse,
            originalMode: responseMode 
          });
        } else {
          // Only show standard response but store both if available
          newMessages.push({ 
            role: 'assistant', 
            content: standardResponse || "No standard response available",
            hiddenContent: ragResponse,
            originalMode: responseMode
          });
        }
        
        return newMessages;
      });
    } catch (error) {
      console.error('Chat error:', error);
      setError(error.message || 'Failed to send message');
      setMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: 'Sorry, there was an error processing your request. ' + 
                  (error.response?.data?.detail || error.message || '')
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    // Generate a new session ID when clearing the chat
    setSessionId(null);
  };

  const messageStyle = {
    user: {
      bgcolor: 'primary.light',
      color: 'white',
      alignSelf: 'flex-end',
      borderRadius: '18px 18px 0 18px',
    },
    assistant: {
      bgcolor: 'grey.100',
      color: 'text.primary',
      alignSelf: 'flex-start',
      borderRadius: '18px 18px 18px 0',
    },
    rag: {
      bgcolor: 'success.light',
      color: 'white',
      alignSelf: 'flex-start',
      borderRadius: '18px 18px 18px 0',
    },
    standard: {
      bgcolor: 'info.light',
      color: 'white',
      alignSelf: 'flex-start',
      borderRadius: '18px 18px 18px 0',
    }
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Snackbar 
        open={!!error} 
        autoHideDuration={6000} 
        onClose={() => setError(null)}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert onClose={() => setError(null)} severity="error">
          {error}
        </Alert>
      </Snackbar>

      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6">
          Chat with Marketing Bot
          {sessionId && (
            <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
              Session: {sessionId.substring(0, 8)}...
            </Typography>
          )}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {/* Response Mode Selector */}
          <Box 
            sx={{ 
              border: '1px solid #e0e0e0', 
              borderRadius: 1, 
              display: 'flex', 
              mr: 2,
              overflow: 'hidden'
            }}
          >
            <Button 
              size="small"
              onClick={() => setResponseMode("rag")}
              variant={responseMode === "rag" ? "contained" : "text"}
              color="success"
              sx={{ borderRadius: 0 }}
            >
              RAG
            </Button>
            <Button 
              size="small"
              onClick={() => setResponseMode("standard")}
              variant={responseMode === "standard" ? "contained" : "text"}
              color="info"
              sx={{ borderRadius: 0 }}
            >
              Standard
            </Button>
            <Button 
              size="small"
              onClick={() => setResponseMode("compare")}
              variant={responseMode === "compare" ? "contained" : "text"}
              color="secondary"
              sx={{ borderRadius: 0 }}
            >
              Compare
            </Button>
          </Box>
          
          <IconButton color="error" onClick={clearChat}>
            <DeleteIcon />
          </IconButton>
        </Box>
      </Box>
      
      {/* Chat Messages */}
      <Paper 
        elevation={3} 
        sx={{ 
          flexGrow: 1, 
          p: 2, 
          mb: 2, 
          overflow: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 2
        }}
      >
        {messages.length === 0 && (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Typography variant="body1" color="text.secondary">
              Start the conversation by sending a message
            </Typography>
          </Box>
        )}
        
        {messages.map((msg, index) => (
          <Box 
            key={index} 
            sx={{ 
              display: 'flex',
              flexDirection: 'row',
              gap: 1,
              justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
            }}
          >
            {msg.role !== 'user' && (
              <Avatar 
                sx={{ bgcolor: msg.type === 'rag' ? 'success.main' : msg.type === 'standard' ? 'info.main' : 'grey.500' }}
              >
                {msg.type === 'rag' ? 'R' : msg.type === 'standard' ? 'S' : 'A'}
              </Avatar>
            )}
            
            <Paper
              elevation={1}
              sx={{
                p: 2,
                maxWidth: '70%',
                ...messageStyle[msg.type || msg.role]
              }}
            >
              {msg.type && (
                <Typography variant="caption" sx={{ fontWeight: 'bold' }}>
                  {msg.type === 'rag' ? 'RAG Response' : 'Standard Response'}
                </Typography>
              )}
              <ReactMarkdown>{typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content, null, 2)}</ReactMarkdown>
            </Paper>
            
            {msg.role === 'user' && (
              <Avatar sx={{ bgcolor: 'primary.main' }}>U</Avatar>
            )}
          </Box>
        ))}
        
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
            <CircularProgress size={30} />
          </Box>
        )}
        
        <div ref={chatEndRef} />
      </Paper>
      
      {/* Message Input */}
      <Paper elevation={3} sx={{ p: 1 }}>
        <form onSubmit={handleSendMessage}>
          <Grid container spacing={1}>
            <Grid item xs>
              <TextField
                fullWidth
                placeholder="Type your message here..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                variant="outlined"
                disabled={loading}
                autoComplete="off"
              />
            </Grid>
            <Grid item>
              <Button
                type="submit"
                variant="contained"
                color="primary"
                endIcon={<SendIcon />}
                disabled={loading || !input.trim()}
              >
                Send
              </Button>
            </Grid>
          </Grid>
        </form>
      </Paper>
    </Box>
  );
};

export default ChatInterface;