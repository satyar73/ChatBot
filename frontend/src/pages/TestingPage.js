import React, { useState } from 'react';
import { Box, Alert, Tabs, Tab } from '@mui/material';
import { TestingProvider, useTestingContext } from '../context/TestingContext';
import useTestingActions from '../hooks/useTestingActions';
import TestingHeader from '../components/testing/TestingHeader';
import TestingControls from '../components/testing/TestingControls';
import TestingResults from '../components/testing/TestingResults';
import AsyncTestRunner from '../components/testing/AsyncTestRunner';

/**
 * Inner component that uses the context and hooks
 */
const TestingContent = () => {
  const { state } = useTestingContext();
  const { clearError } = useTestingActions();
  const { error } = state;
  const [activeTab, setActiveTab] = useState(0);
  
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
      <TestingHeader />
      
      {/* Tab Selection */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={activeTab} onChange={handleTabChange} aria-label="testing tabs">
          <Tab label="Quick Tests" />
          <Tab label="Batch Tests" />
        </Tabs>
      </Box>
      
      {/* Error Message */}
      {error && (
        <Alert severity="error" sx={{ my: 2 }} onClose={clearError}>
          {error}
        </Alert>
      )}
      
      {/* Tab Panels */}
      <Box role="tabpanel" hidden={activeTab !== 0}>
        {activeTab === 0 && (
          <>
            <TestingControls />
            <TestingResults />
          </>
        )}
      </Box>
      
      <Box role="tabpanel" hidden={activeTab !== 1}>
        {activeTab === 1 && (
          <>
            <AsyncTestRunner />
            <TestingResults />
          </>
        )}
      </Box>
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