import React from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Avatar,
  CircularProgress
} from '@mui/material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useChatContext } from '../context/ChatContext';

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
  },
  needl: {
    bgcolor: 'warning.light',
    color: 'white',
    alignSelf: 'flex-start',
    borderRadius: '18px 18px 18px 0',
  }
};

const ChatMessageList = () => {
  const { state, chatEndRef } = useChatContext();
  const { messages, loading, responseMode, isLoadingHistory } = state;
  
  console.log("CHAT MESSAGE LIST - Current messages:", messages);
  console.log("CHAT MESSAGE LIST - Current response mode:", responseMode);

  return (
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
      {messages.length === 0 && !isLoadingHistory && (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
          <Typography variant="body1" color="text.secondary">
            {state.sessionId ? 
              `No ${responseMode} messages in this session. Try sending a message or switching to another mode.` : 
              'Start the conversation by sending a message'}
          </Typography>
        </Box>
      )}
      
      {isLoadingHistory && (
        <Box sx={{ 
          display: 'flex', 
          flexDirection: 'column',
          justifyContent: 'center', 
          alignItems: 'center', 
          height: messages.length === 0 ? '100%' : 'auto',
          py: 3
        }}>
          <CircularProgress size={40} sx={{ mb: 2 }} />
          <Typography variant="body2" color="text.secondary">
            Loading {responseMode.charAt(0).toUpperCase() + responseMode.slice(1)} messages...
          </Typography>
        </Box>
      )}
      
      {messages.map((msg, index) => {
        // Log each message for debugging
        console.log(`Message ${index}:`, JSON.stringify(msg));
        
        // Assign the message type for styling and labels
        let messageType = msg.role;
        
        // For assistant messages, use more specific types based on available info
        if (msg.role === 'assistant') {
          // First check specific message properties
          if (msg.type) {
            // Use type property if present
            messageType = msg.type;
            console.log(`Message ${index} has type attribute: ${messageType}`);
          } else if (msg.response_type) {
            // Use response_type if present (API response format)
            messageType = msg.response_type === 'no_rag' ? 'standard' : msg.response_type;
            console.log(`Message ${index} has response_type: ${msg.response_type}, mapped to ${messageType}`);
          } else if (msg.originalMode) {
            // Use originalMode if present (legacy format)
            messageType = msg.originalMode === 'no_rag' ? 'standard' : msg.originalMode;
            console.log(`Message ${index} has originalMode: ${msg.originalMode}, mapped to ${messageType}`);
          } else {
            // If no specific message attributes, use the current response mode
            if (responseMode === 'standard') {
              messageType = 'standard';
            } else if (responseMode === 'needl') {
              messageType = 'needl';
            } else if (responseMode === 'rag') {
              messageType = 'rag';
            }
            console.log(`Message ${index} using current responseMode: ${responseMode}, mapped to ${messageType}`);
          }
          
          // Special handling for Needl responses (override other types)
          if (msg.isNeedlResponse || messageType === 'needl') {
            messageType = 'needl';
            console.log(`Message ${index} identified as Needl response`);
          }
        }

        return (
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
                sx={{ 
                  bgcolor: 
                    messageType === 'rag' ? 'success.main' : 
                    messageType === 'standard' ? 'info.main' : 
                    messageType === 'needl' ? 'warning.main' :
                    'grey.500' 
                }}
              >
                {messageType === 'rag' ? 'R' : 
                 messageType === 'standard' ? 'S' : 
                 messageType === 'needl' ? 'N' : 'A'}
              </Avatar>
            )}
            
            <Paper
              elevation={1}
              sx={{
                p: 2,
                maxWidth: '70%',
                ...messageStyle[messageType]
              }}
              className="markdown-content"
            >
              {/* Always show message type label for AI responses */}
              {msg.role === 'assistant' && (
                <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 1 }}>
                  {messageType === 'rag' ? 'RAG Response' : 
                   messageType === 'standard' ? 'Standard Response' :
                   messageType === 'needl' ? 'Needl Response' : 
                   'AI Response'}
                </Typography>
              )}
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content, null, 2)}</ReactMarkdown>
            </Paper>
            
            {msg.role === 'user' && (
              <Avatar sx={{ bgcolor: 'primary.main' }}>U</Avatar>
            )}
          </Box>
        );
      })}
      
      {loading && !isLoadingHistory && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
          <CircularProgress size={30} />
        </Box>
      )}
      
      <div ref={chatEndRef} />
    </Paper>
  );
};

export default ChatMessageList;