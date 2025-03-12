import React from 'react';
import { Alert } from '@mui/material';

const StatusAlerts = ({ error, success, onCloseError, onCloseSuccess }) => {
  return (
    <>
      {error && (
        <Alert severity="error" onClose={onCloseError}>
          {error}
        </Alert>
      )}
      
      {success && (
        <Alert severity="success" onClose={onCloseSuccess}>
          {success}
        </Alert>
      )}
    </>
  );
};

export default StatusAlerts;