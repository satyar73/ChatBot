import React from 'react';
import { 
  Box, 
  Typography, 
  TextField, 
  Button, 
  Paper, 
  IconButton,
  Collapse
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import { useChatContext } from '../context/ChatContext';
import useChatActions from '../hooks/useChatActions';

/**
 * Component for editing the system prompt
 */
const SystemPromptEditor = () => {
  const { state } = useChatContext();
  const { systemPrompt, showSystemPrompt } = state;
  const { setSystemPrompt, toggleSystemPrompt } = useChatActions();

  const handleChange = (e) => {
    setSystemPrompt(e.target.value);
  };

  const handleClear = () => {
    setSystemPrompt('');
  };

  return (
    <Paper 
      elevation={2} 
      sx={{ 
        mb: 2, 
        p: 2, 
        backgroundColor: '#f5f5f5',
        borderLeft: '4px solid #3f51b5'
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="subtitle1" fontWeight="bold">
          System Prompt Editor
        </Typography>
        <IconButton onClick={toggleSystemPrompt} size="small">
          {showSystemPrompt ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        </IconButton>
      </Box>

      <Collapse in={showSystemPrompt}>
        <TextField
          fullWidth
          multiline
          rows={8}
          variant="outlined"
          placeholder="Enter custom system prompt (leave empty to use default)"
          value={systemPrompt}
          onChange={handleChange}
          sx={{ mb: 1 }}
        />
        
        <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button 
            variant="outlined" 
            color="error" 
            onClick={handleClear}
            sx={{ mr: 1 }}
          >
            Clear
          </Button>
          <Button
            variant="contained"
            color="primary"
            onClick={toggleSystemPrompt}
          >
            Close
          </Button>
        </Box>
      </Collapse>
    </Paper>
  );
};

export default SystemPromptEditor;