import React from 'react';
import { Box, Alert } from '@mui/material';
import { TestingProvider, useTestingContext } from '../context/TestingContext';
import useTestingActions from '../hooks/useTestingActions';
import TestingHeader from '../components/testing/TestingHeader';
import TestingControls from '../components/testing/TestingControls';
import TestingResults from '../components/testing/TestingResults';

/**
 * Inner component that uses the context and hooks
 */
const TestingContent = () => {
  const { state } = useTestingContext();
  const { clearError } = useTestingActions();
  const { error } = state;
  
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
      <TestingHeader />
      <TestingControls />
      
      {/* Error Message */}
      {error && (
        <Alert severity="error" sx={{ my: 2 }} onClose={clearError}>
          {error}
        </Alert>
      )}
      
      <TestingResults />
    </Box>
  );
};

/**
 * Main page component that provides the context
 */
const TestingPage = () => {
  return (
    <TestingProvider>
      <TestingContent />
    </TestingProvider>
  );
};

export default TestingPage;