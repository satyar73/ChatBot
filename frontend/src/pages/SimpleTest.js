import React from 'react';
import { Box, Typography, Button, Paper } from '@mui/material';

function SimpleTest() {
  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h4">Simple Test Page</Typography>
      <Typography variant="body1" sx={{ mt: 2 }}>
        This is a simple test page to verify routing is working.
      </Typography>
      
      <Paper sx={{ p: 2, mt: 2 }}>
        <Button 
          variant="contained" 
          onClick={() => alert('Button clicked!')}
        >
          Click Me
        </Button>
      </Paper>
    </Box>
  );
}

export default SimpleTest;