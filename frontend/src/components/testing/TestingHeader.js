import React from 'react';
import { Typography, Box } from '@mui/material';

const TestingHeader = () => {
  return (
    <Box sx={{ mb: 1 }}>
      <Typography variant="h5">Chat Testing Interface</Typography>
      <Typography variant="body2" color="text.secondary">
        Run tests to evaluate chat performance with both quick tests and long-running batch operations
      </Typography>
    </Box>
  );
};

export default TestingHeader;