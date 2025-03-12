import React from 'react';
import { Box, Typography, Button, IconButton } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import { useChatContext } from '../context/ChatContext';
import useChatActions from '../hooks/useChatActions';

const ChatHeader = () => {
  const { state } = useChatContext();
  const { responseMode, sessionId } = state;
  const { setResponseMode, clearChat } = useChatActions();

  return (
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
  );
};

export default ChatHeader;