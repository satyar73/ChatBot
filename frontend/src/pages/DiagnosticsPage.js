import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Button, 
  Alert, 
  CircularProgress, 
  Grid,
  Divider,
  Card,
  CardContent,
  CardHeader,
  TextField,
  IconButton 
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';

function DiagnosticsPage() {
  // Status test states
  const [statusLoading, setStatusLoading] = useState(false);
  const [statusResult, setStatusResult] = useState(null);
  const [statusError, setStatusError] = useState('');
  
  // File upload test states
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadError, setUploadError] = useState('');
  const [testFile, setTestFile] = useState(null);
  
  // Component initialization
  useEffect(() => {
    console.log('DiagnosticsPage component mounted');
    document.title = 'System Diagnostics';
    
    // Automatically run the status check when the page loads
    testBackendStatus();
  }, []);

  // Test backend status
  const testBackendStatus = async () => {
    setStatusLoading(true);
    setStatusError('');
    setStatusResult(null);
    
    try {
      console.log('Testing backend status at /test/status');
      const response = await fetch('/test/status');
      console.log('Response status:', response.status);
      
      if (!response.ok) {
        throw new Error(`Error ${response.status}: ${await response.text()}`);
      }
      
      const data = await response.json();
      setStatusResult(data);
    } catch (err) {
      console.error('Status test error:', err);
      setStatusError(`Backend status test failed: ${err.message}`);
    } finally {
      setStatusLoading(false);
    }
  };

  // Test file upload
  const testFileUpload = async () => {
    if (!testFile) {
      setUploadError('Please select a file first');
      return;
    }
    
    setUploadLoading(true);
    setUploadError('');
    setUploadResult(null);
    
    const formData = new FormData();
    formData.append('file', testFile);
    
    try {
      console.log('Testing file upload to /test/upload');
      console.log('File:', testFile);
      
      const response = await fetch('/test/upload', {
        method: 'POST',
        body: formData
      });
      
      console.log('Response status:', response.status);
      
      if (!response.ok) {
        throw new Error(`Error ${response.status}: ${await response.text()}`);
      }
      
      const data = await response.json();
      setUploadResult(data);
    } catch (err) {
      console.error('File upload test error:', err);
      setUploadError(`File upload test failed: ${err.message}`);
    } finally {
      setUploadLoading(false);
    }
  };
  
  // Handle file selection
  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      setTestFile(e.target.files[0]);
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom>
        System Diagnostics
      </Typography>
      <Typography variant="body1" sx={{ mb: 4 }}>
        Use this page to verify connectivity and system functionality.
      </Typography>
      
      <Grid container spacing={3}>
        {/* Status Test Card */}
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardHeader 
              title="Backend Status" 
              subheader="Verify the backend service is running properly"
              action={
                <IconButton onClick={testBackendStatus} disabled={statusLoading}>
                  <RefreshIcon />
                </IconButton>
              }
            />
            <Divider />
            <CardContent>
              <Button 
                variant="contained" 
                onClick={testBackendStatus}
                disabled={statusLoading}
                fullWidth
                sx={{ mb: 2 }}
                startIcon={statusLoading ? <CircularProgress size={20} /> : null}
              >
                {statusLoading ? 'Checking Status...' : 'Check Backend Status'}
              </Button>
              
              {statusError && (
                <Alert severity="error" sx={{ mt: 2, mb: 2 }}>
                  {statusError}
                </Alert>
              )}
              
              {statusResult && (
                <Box sx={{ mt: 2 }}>
                  <Alert severity="success" sx={{ mb: 2 }}>
                    {statusResult.message}
                  </Alert>
                  
                  <Typography variant="subtitle2" gutterBottom>
                    System Information:
                  </Typography>
                  
                  <pre style={{ 
                    backgroundColor: '#f5f5f5', 
                    padding: '10px', 
                    borderRadius: '4px',
                    overflow: 'auto',
                    maxHeight: '200px'
                  }}>
                    {JSON.stringify(statusResult.version, null, 2)}
                  </pre>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* File Upload Test Card */}
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardHeader 
              title="File Upload Test" 
              subheader="Verify file upload functionality"
            />
            <Divider />
            <CardContent>
              <Box sx={{ mb: 2 }}>
                <input
                  type="file"
                  id="test-file-upload"
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                />
                <label htmlFor="test-file-upload">
                  <Button variant="outlined" component="span" fullWidth>
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
                disabled={uploadLoading || !testFile}
                fullWidth
                startIcon={uploadLoading ? <CircularProgress size={20} /> : null}
              >
                {uploadLoading ? 'Uploading...' : 'Test File Upload'}
              </Button>
              
              {uploadError && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {uploadError}
                </Alert>
              )}
              
              {uploadResult && (
                <Box sx={{ mt: 2 }}>
                  <Alert severity="success" sx={{ mb: 2 }}>
                    File upload successful!
                  </Alert>
                  
                  <Typography variant="subtitle2" gutterBottom>
                    Upload Results:
                  </Typography>
                  
                  <pre style={{ 
                    backgroundColor: '#f5f5f5', 
                    padding: '10px', 
                    borderRadius: '4px',
                    overflow: 'auto',
                    maxHeight: '200px'
                  }}>
                    {JSON.stringify(uploadResult, null, 2)}
                  </pre>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default DiagnosticsPage;