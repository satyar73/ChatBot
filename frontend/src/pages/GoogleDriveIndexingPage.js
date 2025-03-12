import React, { useEffect } from 'react';
import { Box } from '@mui/material';
import {GoogleDriveProvider, useGoogleDriveContext} from '../context/indexing/GoogleDriveContext';
import useGoogleDriveActions from '../hooks/indexing/useGoogleDriveActions';
import GoogleDriveHeader from '../components/indexing/GoogleDriveHeader';
import GoogleDriveStats from '../components/indexing/GoogleDriveStats';
import GoogleDriveControls from '../components/indexing/GoogleDriveControls';
import GoogleDriveFileList from '../components/indexing/GoogleDriveFileList';
import StatusAlerts from '../components/indexing/StatusAlerts';

/**
 * Inner component that uses the context and hooks
 */
const GoogleDriveIndexingContent = () => {
  const { state } = useGoogleDriveContext();
  const { fetchFiles, clearError, clearSuccess } = useGoogleDriveActions();
  const { error, success } = state;
  
  // Fetch files on component mount
  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);
  
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
      <GoogleDriveHeader />
      <GoogleDriveStats />
      <GoogleDriveControls />
      
      <StatusAlerts
        error={error}
        success={success}
        onCloseError={clearError}
        onCloseSuccess={clearSuccess}
      />
      
      <GoogleDriveFileList />
    </Box>
  );
};

/**
 * Main page component that provides the context
 */
const GoogleDriveIndexingPage = () => {
  return (
    <GoogleDriveProvider>
      <GoogleDriveIndexingContent />
    </GoogleDriveProvider>
  );
};

export default GoogleDriveIndexingPage;