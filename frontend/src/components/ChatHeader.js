import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  IconButton, 
  Tooltip, 
  Menu, 
  MenuItem,
  Badge,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import SettingsIcon from '@mui/icons-material/Settings';
import FormatSizeIcon from '@mui/icons-material/FormatSize';
import HistoryIcon from '@mui/icons-material/History';
import AddIcon from '@mui/icons-material/Add';
import RefreshIcon from '@mui/icons-material/Refresh';
import { useChatContext } from '../context/ChatContext';
import useChatActions from '../hooks/useChatActions';
import { useEnvironment } from '../context/EnvironmentContext';

const ChatHeader = () => {
  const { state, sessionManagement } = useChatContext();
  const { 
    responseMode,
    promptStyle, 
    sessionId, 
    systemPrompt, 
    availableSessions,
    sessionMetadata,
    isLoadingHistory 
  } = state;
  const { 
    switchSession, 
    createNewSession, 
    refreshSessionList 
  } = sessionManagement;
  const { setResponseMode, setPromptStyle, clearChat, toggleSystemPrompt } = useChatActions();
  const { COMPANY_NAME } = useEnvironment();
  
  // State for prompt style menu
  const [anchorEl, setAnchorEl] = useState(null);
  const [sessionsMenuAnchor, setSessionsMenuAnchor] = useState(null);
  const [showSessionsDialog, setShowSessionsDialog] = useState(false);
  
  const promptStyleMenuOpen = Boolean(anchorEl);
  const sessionsMenuOpen = Boolean(sessionsMenuAnchor);
  
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
  
  // Session menu handlers
  const handleSessionMenuClick = (event) => {
    setSessionsMenuAnchor(event.currentTarget);
    // Refresh the sessions list when opening the menu
    refreshSessionList();
  };
  
  const handleSessionMenuClose = () => {
    setSessionsMenuAnchor(null);
  };
  
  const handleSessionsDialogOpen = () => {
    setShowSessionsDialog(true);
    handleSessionMenuClose();
  };
  
  const handleSessionsDialogClose = () => {
    setShowSessionsDialog(false);
  };
  
  const handleSwitchSession = (newSessionId) => {
    switchSession(newSessionId);
    handleSessionMenuClose();
  };
  
  const handleCreateNewSession = () => {
    createNewSession();
    handleSessionMenuClose();
  };
  
  const formatDate = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  return (
    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <Typography variant="h6" sx={{ mr: 2 }}>
          Chat with {COMPANY_NAME} Marketing Bot
        </Typography>
        
        {/* Session badge/button */}
        <Tooltip title="Manage Sessions">
          <Button 
            variant="outlined" 
            color="primary" 
            size="small"
            startIcon={<HistoryIcon />}
            onClick={handleSessionMenuClick}
            disabled={isLoadingHistory}
          >
            {isLoadingHistory ? 'Loading...' : 
              sessionId ? `Session: ${sessionId.substring(0, 8)}...` : 'No Session'}
          </Button>
        </Tooltip>
        
        {/* Sessions Menu */}
        <Menu
          anchorEl={sessionsMenuAnchor}
          open={sessionsMenuOpen}
          onClose={handleSessionMenuClose}
        >
          <MenuItem onClick={handleCreateNewSession}>
            <AddIcon fontSize="small" sx={{ mr: 1 }} />
            New Session
          </MenuItem>
          <MenuItem onClick={refreshSessionList}>
            <RefreshIcon fontSize="small" sx={{ mr: 1 }} />
            Refresh List
          </MenuItem>
          <MenuItem onClick={handleSessionsDialogOpen}>
            <HistoryIcon fontSize="small" sx={{ mr: 1 }} />
            View All Sessions
          </MenuItem>
          <MenuItem disabled>
            <Typography variant="overline">Recent Sessions</Typography>
          </MenuItem>
          {availableSessions.slice(0, 5).map(session => (
            <MenuItem 
              key={session.session_id}
              onClick={() => handleSwitchSession(session.session_id)}
              selected={session.session_id === sessionId}
            >
              <Box sx={{ 
                maxWidth: 250, 
                overflow: 'hidden',
                textOverflow: 'ellipsis'
              }}>
                <Typography variant="body2" noWrap>
                  {session.session_id.substring(0, 8)}...
                </Typography>
                <Typography variant="caption" color="text.secondary" noWrap>
                  {formatDate(session.last_updated)} • {session.message_count} msgs
                </Typography>
              </Box>
            </MenuItem>
          ))}
        </Menu>
        
        {/* Session tags if available */}
        {sessionMetadata && sessionMetadata.tags && sessionMetadata.tags.length > 0 && (
          <Box sx={{ ml: 1, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
            {sessionMetadata.tags.map(tag => (
              <Chip 
                key={tag} 
                label={tag} 
                size="small" 
                color="primary" 
                variant="outlined"
              />
            ))}
          </Box>
        )}
      </Box>
      
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
            onClick={() => {
              console.log('RAG button clicked');
              setResponseMode("rag");
            }}
            variant={responseMode === "rag" ? "contained" : "text"}
            color="success"
            sx={{ borderRadius: 0 }}
            disabled={isLoadingHistory}
          >
            RAG {responseMode === "rag" && isLoadingHistory && "..."}
          </Button>
          <Button 
            size="small"
            onClick={() => {
              console.log('Standard button clicked');
              setResponseMode("standard");
            }}
            variant={responseMode === "standard" ? "contained" : "text"}
            color="info"
            sx={{ borderRadius: 0 }}
            disabled={isLoadingHistory}
          >
            Standard {responseMode === "standard" && isLoadingHistory && "..."}
          </Button>
          <Button 
            size="small"
            onClick={() => {
              console.log('Needl button clicked');
              setResponseMode("needl");
            }}
            variant={responseMode === "needl" ? "contained" : "text"}
            color="warning"
            sx={{ borderRadius: 0 }}
            disabled={isLoadingHistory}
          >
            Needl {responseMode === "needl" && isLoadingHistory && "..."}
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
              aria-controls={promptStyleMenuOpen ? "prompt-style-menu" : undefined}
              aria-haspopup="true"
              aria-expanded={promptStyleMenuOpen ? "true" : undefined}
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
          open={promptStyleMenuOpen}
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
      
      {/* Sessions Dialog */}
      <Dialog 
        open={showSessionsDialog} 
        onClose={handleSessionsDialogClose}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h6">Available Sessions</Typography>
            <Box>
              <Button 
                startIcon={<RefreshIcon />} 
                onClick={refreshSessionList}
                size="small"
              >
                Refresh
              </Button>
              <Button 
                startIcon={<AddIcon />} 
                onClick={handleCreateNewSession}
                variant="contained" 
                color="primary"
                size="small"
                sx={{ ml: 1 }}
              >
                New Session
              </Button>
            </Box>
          </Box>
        </DialogTitle>
        <DialogContent>
          <List sx={{ width: '100%' }}>
            {availableSessions.map(session => (
              <ListItem 
                key={session.session_id}
                button
                onClick={() => handleSwitchSession(session.session_id)}
                selected={session.session_id === sessionId}
                sx={{ 
                  borderBottom: '1px solid #eee',
                  backgroundColor: session.session_id === sessionId ? '#f0f7ff' : 'transparent'
                }}
              >
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Typography variant="body1" sx={{ fontWeight: 'medium' }}>
                        {session.session_id}
                      </Typography>
                      {session.tags && session.tags.length > 0 && (
                        <Box sx={{ ml: 2, display: 'flex', gap: 0.5 }}>
                          {session.tags.map(tag => (
                            <Chip key={tag} label={tag} size="small" />
                          ))}
                        </Box>
                      )}
                    </Box>
                  }
                  secondary={
                    <Box>
                      <Typography variant="caption" display="block">
                        Created: {formatDate(session.created_at)}
                      </Typography>
                      <Typography variant="caption" display="block">
                        Last Updated: {formatDate(session.last_updated)}
                      </Typography>
                      <Typography variant="caption">
                        Messages: {session.message_count} • Prompts: {session.prompt_count}
                      </Typography>
                    </Box>
                  }
                />
              </ListItem>
            ))}
          </List>
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default ChatHeader;