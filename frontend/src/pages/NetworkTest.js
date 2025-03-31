import React, { useState, useEffect } from 'react';
import { Box, Button, Typography, Paper, TextField, Alert, CircularProgress } from '@mui/material';

function NetworkTest() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [testFile, setTestFile] = useState(null);
  
  // Add a log on component mount to verify the component is loading
  useEffect(() => {
    console.log('NetworkTest component mounted');
    // Also update the document title as another way to verify the component loaded
    document.title = 'Network Test Page';
  }, []);

  // Test simple connection
  const testConnection = async () => {
    setLoading(true);
    setError('');
    setResults(null);
    
    try {
      console.log('Testing connection to /chat/test-slides-connection');
      const response = await fetch('/chat/test-slides-connection');
      console.log('Response status:', response.status);
      
      if (!response.ok) {
        throw new Error(`Error ${response.status}: ${await response.text()}`);
      }
      
      const data = await response.json();
      setResults({
        type: 'connection',
        data: data
      });
    } catch (err) {
      console.error('Connection test error:', err);
      setError(`Connection test failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Test file upload
  const testFileUpload = async () => {
    if (!testFile) {
      setError('Please select a file first');
      return;
    }
    
    setLoading(true);
    setError('');
    setResults(null);
    
    const formData = new FormData();
    formData.append('file', testFile);
    
    try {
      console.log('Testing file upload to /chat/test-slides-upload');
      console.log('File:', testFile);
      
      const response = await fetch('/chat/test-slides-upload', {
        method: 'POST',
        body: formData
      });
      
      console.log('Response status:', response.status);
      
      if (!response.ok) {
        throw new Error(`Error ${response.status}: ${await response.text()}`);
      }
      
      const data = await response.json();
      setResults({
        type: 'upload',
        data: data
      });
    } catch (err) {
      console.error('File upload test error:', err);
      setError(`File upload test failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle file selection
  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      setTestFile(e.target.files[0]);
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 800, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom>
        Network Connectivity Tests
      </Typography>
      
      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Test Basic Connection
        </Typography>
        <Button 
          variant="contained" 
          onClick={testConnection}
          disabled={loading}
          sx={{ mt: 1 }}
        >
          {loading && results?.type === 'connection' ? <CircularProgress size={24} /> : 'Test Connection'}
        </Button>
      </Paper>
      
      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Test File Upload
        </Typography>
        <Box sx={{ my: 2 }}>
          <input
            type="file"
            id="test-file-upload"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          <label htmlFor="test-file-upload">
            <Button variant="outlined" component="span">
              Select Test File
            </Button>
          </label>
          {testFile && (
            <Typography variant="body2" sx={{ mt: 1 }}>
              Selected: {testFile.name}
            </Typography>
          )}
        </Box>
        <Button 
          variant="contained" 
          onClick={testFileUpload}
          disabled={loading || !testFile}
          sx={{ mt: 1 }}
        >
          {loading && results?.type === 'upload' ? <CircularProgress size={24} /> : 'Test File Upload'}
        </Button>
      </Paper>
      
      {error && (
        <Alert severity="error" sx={{ mt: 3 }}>
          {error}
        </Alert>
      )}
      
      {results && (
        <Paper sx={{ p: 3, mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Test Results
          </Typography>
          <pre style={{ 
            backgroundColor: '#f5f5f5', 
            padding: '10px', 
            borderRadius: '4px',
            overflow: 'auto',
            maxHeight: '300px'
          }}>
            {JSON.stringify(results.data, null, 2)}
          </pre>
        </Paper>
      )}
    </Box>
  );
}

export default NetworkTest;