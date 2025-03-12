import React from 'react';
import { Snackbar, Alert } from '@mui/material';
import { useChatContext } from '../context/ChatContext';
import useChatActions from '../hooks/useChatActions';

const ErrorAlert = () => {
  const { state } = useChatContext();
  const { error } = state;
  const { clearError } = useChatActions();

  return (
    <Snackbar 
      open={!!error} 
      autoHideDuration={6000} 
      onClose={clearError}
      anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
    >
      <Alert onClose={clearError} severity="error">
        {error}
      </Alert>
    </Snackbar>
  );
};

export default ErrorAlert;