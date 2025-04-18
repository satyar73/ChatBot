import React from 'react';
import { Box, Typography, Button, IconButton, Tooltip, Menu, MenuItem } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import SettingsIcon from '@mui/icons-material/Settings';
import FormatSizeIcon from '@mui/icons-material/FormatSize';
import { useChatContext } from '../context/ChatContext';
import useChatActions from '../hooks/useChatActions';
import { useEnvironment } from '../context/EnvironmentContext';

const ChatHeader = () => {
  const { state } = useChatContext();
  const { responseMode, promptStyle, sessionId, systemPrompt } = state;
  const { setResponseMode, setPromptStyle, clearChat, toggleSystemPrompt } = useChatActions();
  const { COMPANY_NAME } = useEnvironment();
  
  // State for prompt style menu
  const [anchorEl, setAnchorEl] = React.useState(null);
  const open = Boolean(anchorEl);
  
  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };
  
  const handleClose = () => {
    setAnchorEl(null);
  };
  
  const handlePromptStyleChange = (style) => {
    setPromptStyle(style);
    handleClose();
  };

  return (
    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
      <Typography variant="h6">
        Chat with {COMPANY_NAME} Marketing Bot
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
          <Button 
            size="small"
            onClick={() => setResponseMode("needl")}
            variant={responseMode === "needl" ? "contained" : "text"}
            color="warning"
            sx={{ borderRadius: 0 }}
          >
            Needl
          </Button>
        </Box>
        
        {/* Prompt Style Selector with badge showing current style */}
        <Tooltip title={`Prompt Style: ${promptStyle.charAt(0).toUpperCase() + promptStyle.slice(1)}`}>
          <Box sx={{ position: 'relative', mr: 1 }}>
            <IconButton
              color={
                promptStyle === "default" ? "primary" : 
                promptStyle === "detailed" ? "success" : 
                "secondary"
              }
              onClick={handleClick}
              aria-controls={open ? "prompt-style-menu" : undefined}
              aria-haspopup="true"
              aria-expanded={open ? "true" : undefined}
            >
              <FormatSizeIcon />
            </IconButton>
            <Typography 
              variant="caption" 
              sx={{ 
                position: 'absolute', 
                top: -4, 
                right: -4, 
                backgroundColor: promptStyle === "default" ? "#1976d2" : 
                                promptStyle === "detailed" ? "#2e7d32" : 
                                "#9c27b0",
                color: 'white',
                borderRadius: '50%',
                width: 16,
                height: 16,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '0.6rem',
                fontWeight: 'bold'
              }}
            >
              {promptStyle.charAt(0).toUpperCase()}
            </Typography>
          </Box>
        </Tooltip>
        <Menu
          id="prompt-style-menu"
          anchorEl={anchorEl}
          open={open}
          onClose={handleClose}
        >
          <MenuItem 
            onClick={() => handlePromptStyleChange("default")}
            selected={promptStyle === "default"}
            sx={{ display: 'block' }}
          >
            <Typography variant="subtitle2">Default</Typography>
            <Typography variant="caption" color="text.secondary">
              Balanced responses with moderate detail
            </Typography>
          </MenuItem>
          <MenuItem 
            onClick={() => handlePromptStyleChange("detailed")}
            selected={promptStyle === "detailed"}
            sx={{ display: 'block' }}
          >
            <Typography variant="subtitle2">Detailed</Typography>
            <Typography variant="caption" color="text.secondary">
              Comprehensive responses with full explanations
            </Typography>
          </MenuItem>
          <MenuItem 
            onClick={() => handlePromptStyleChange("concise")}
            selected={promptStyle === "concise"}
            sx={{ display: 'block' }}
          >
            <Typography variant="subtitle2">Concise</Typography>
            <Typography variant="caption" color="text.secondary">
              Brief responses with essential information only
            </Typography>
          </MenuItem>
        </Menu>
        
        <Tooltip title="System Prompt Editor">
          <IconButton 
            color={systemPrompt ? "success" : "primary"} 
            onClick={toggleSystemPrompt}
            sx={{ mr: 1 }}
          >
            <SettingsIcon />
          </IconButton>
        </Tooltip>
        
        <Tooltip title="Clear Chat">
          <IconButton color="error" onClick={clearChat}>
            <DeleteIcon />
          </IconButton>
        </Tooltip>
      </Box>
    </Box>
  );
};

export default ChatHeader;