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
  }
};

const ChatMessageList = () => {
  const { state, chatEndRef } = useChatContext();
  const { messages, loading } = state;

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
            className="markdown-content"
          >
            {msg.type && (
              <Typography variant="caption" sx={{ fontWeight: 'bold' }}>
                {msg.type === 'rag' ? 'RAG Response' : 'Standard Response'}
              </Typography>
            )}
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content, null, 2)}</ReactMarkdown>
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
  );
};

export default ChatMessageList;