import React from 'react';
import { Box, Typography, Button } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { useGoogleDriveContext } from '../../context/indexing/GoogleDriveContext';
import useGoogleDriveActions from '../../hooks/indexing/useGoogleDriveActions';

const GoogleDriveHeader = () => {
  const { state } = useGoogleDriveContext();
  const { loading, indexing } = state;
  const { fetchFiles } = useGoogleDriveActions();

  return (
    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <Typography variant="h5">Google Drive Indexing</Typography>
      <Button
        startIcon={<RefreshIcon />}
        onClick={fetchFiles}
        disabled={loading || indexing}
      >
        Refresh
      </Button>
    </Box>
  );
};

export default GoogleDriveHeader;