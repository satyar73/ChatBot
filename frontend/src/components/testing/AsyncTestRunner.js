import React, { useState, useRef, useEffect } from 'react';
import { 
  Box, Paper, Typography, TextField, Button, 
  LinearProgress, CircularProgress, 
  Alert, Card, CardContent, Chip,
  InputAdornment, IconButton
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import FolderIcon from '@mui/icons-material/Folder';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import PendingIcon from '@mui/icons-material/Pending';
import { useTestingContext, ACTIONS } from '../../context/TestingContext';
import useTestingActions from '../../hooks/useTestingActions';

/**
 * Component for handling long-running, asynchronous test execution
 */
const AsyncTestRunner = () => {
  const [file, setFile] = useState(null);
  const [fileError, setFileError] = useState(null);
  const fileInputRef = useRef(null);
  const { state, dispatch } = useTestingContext();
  const { 
    runTests, cleanupTestJob, clearError
  } = useTestingActions();
  
  const { 
    jobStatus, jobProgress, statusMessage, testJobId, loading
  } = state;

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (testJobId) {
        cleanupTestJob();
      }
    };
  }, [testJobId, cleanupTestJob]);
  
  // Function to manually reset the test job state
  const handleReset = () => {
    cleanupTestJob();
    clearError();
    dispatch({ type: ACTIONS.SET_JOB_STATUS, payload: null });
    dispatch({ type: ACTIONS.SET_JOB_PROGRESS, payload: 0 });
    dispatch({ type: ACTIONS.SET_STATUS_MESSAGE, payload: null });
    dispatch({ type: ACTIONS.SET_TEST_JOB_ID, payload: null });
    dispatch({ type: ACTIONS.SET_LOADING, payload: false });
    setFile(null);
  };

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;
    
    // Validate file type (must be CSV)
    if (!selectedFile.name.endsWith('.csv')) {
      setFileError('Only CSV files are supported');
      setFile(null);
      return;
    }
    
    // Validate file size (max 5MB)
    if (selectedFile.size > 5 * 1024 * 1024) {
      setFileError('File size must be less than 5MB');
      setFile(null);
      return;
    }
    
    setFile(selectedFile);
    setFileError(null);
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleStartTest = async () => {
    if (!file) {
      setFileError('Please select a CSV file');
      return;
    }
    
    try {
      // Update the uploadedFile in the context state first
      dispatch({ type: ACTIONS.SET_UPLOADED_FILE, payload: file });
      
      // Run the tests with the file in context
      await runTests(null); 
    } catch (error) {
      setFileError(`Error starting test: ${error.message}`);
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 3, mt: 3 }}>
      <Typography variant="h6" gutterBottom>
        Run Batch Tests
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Upload a CSV file containing test cases for batch processing.
        This uses an asynchronous job system to handle long-running tests.
      </Typography>

      {/* File Upload Section */}
      <Box sx={{ display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, gap: 2, alignItems: 'flex-start', mb: 3 }}>
        <Box sx={{ display: 'flex', flex: 1, width: '100%' }}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="No file selected"
            value={file ? file.name : ''}
            InputProps={{
              readOnly: true,
              startAdornment: file && (
                <InputAdornment position="start">
                  <CloudUploadIcon color="primary" />
                </InputAdornment>
              ),
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton 
                    onClick={handleBrowseClick}
                    disabled={loading}
                  >
                    <FolderIcon />
                  </IconButton>
                </InputAdornment>
              )
            }}
            error={!!fileError}
            helperText={fileError}
            disabled={loading || jobStatus === 'running'}
          />
        </Box>

        <Button
          variant="contained"
          color="primary"
          startIcon={<UploadFileIcon />}
          onClick={handleStartTest}
          disabled={!file || loading || jobStatus === 'running'}
          sx={{ height: '56px', minWidth: '180px' }}
        >
          Upload & Run Tests
        </Button>

        {/* Hidden file input */}
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept=".csv"
          onChange={handleFileChange}
          disabled={loading || jobStatus === 'running'}
        />
      </Box>

      {/* Job Status Section - always visible while running or after completion */}
      {(jobStatus || loading) && (
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Typography variant="body2" color="text.secondary">
              {statusMessage 
                ? statusMessage.includes("\n") 
                  ? statusMessage.split("\n").slice(0, 2).join(" - ") // Show the first two lines of status (test # and current test)
                  : statusMessage
                : `Test job status: ${jobStatus || 'initializing...'}`
              }
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip 
                label={jobStatus || 'initializing'} 
                size="small"
                color={
                  jobStatus === 'completed' ? 'success' : 
                  jobStatus === 'failed' ? 'error' : 
                  jobStatus === 'running' ? 'primary' : 'default'
                }
                icon={
                  jobStatus === 'completed' ? <CheckCircleIcon /> : 
                  jobStatus === 'failed' ? <ErrorIcon /> : 
                  jobStatus === 'running' ? <CircularProgress size={16} /> : 
                  <PendingIcon />
                }
              />
              {/* Reset button for completed or failed jobs */}
              {(jobStatus === 'completed' || jobStatus === 'failed') && (
                <Button 
                  variant="outlined" 
                  size="small" 
                  color="primary" 
                  onClick={handleReset}
                >
                  Reset
                </Button>
              )}
            </Box>
          </Box>
          
          {/* Always show a progress bar */}
          <LinearProgress 
            variant={jobStatus === 'running' && (!jobProgress || jobProgress < 5) ? "indeterminate" : "determinate"}
            value={jobProgress || 0} 
            sx={{ 
              height: 10, 
              borderRadius: 1,
              visibility: 'visible', // Always visible
              '& .MuiLinearProgress-bar': {
                transition: 'transform 0.5s ease' // Smoother transitions
              }
            }} 
          />
          
          {testJobId && (
            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
              Job ID: {testJobId}
            </Typography>
          )}
          
          {/* Display additional status message details if present */}
          {statusMessage && statusMessage.includes("\n") && (
            <Alert severity="info" sx={{ mt: 2, fontSize: '0.875rem' }}>
              {statusMessage.split("\n").slice(1).map((line, index) => (
                // Add style to ensure file paths can be read without being cut off
                <div key={index} style={{ 
                  wordBreak: 'break-word', 
                  marginBottom: '4px',
                  whiteSpace: 'normal' 
                }}>
                  {line}
                </div>
              ))}
            </Alert>
          )}
        </Box>
      )}

      {/* Upload Instructions */}
      {!file && !jobStatus && (
        <Alert severity="info" sx={{ mt: 2 }}>
          <Typography variant="body2">
            Upload a CSV file with the following format:
          </Typography>
          <pre style={{ margin: '10px 0', background: '#f5f5f5', padding: '10px', borderRadius: '4px' }}>
            prompt,expected_result
            "What is your name?","My name is ChatBot."
            "How does this work?","I use RAG to enhance my responses."
          </pre>
        </Alert>
      )}

      {/* File Details */}
      {file && !jobStatus && (
        <Alert severity="info" sx={{ mt: 2 }}>
          Selected file: {file.name} ({(file.size / 1024).toFixed(2)} KB)
        </Alert>
      )}

      {/* Error Display */}
      {jobStatus === 'failed' && state.error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {state.error}
        </Alert>
      )}
    </Paper>
  );
};

export default AsyncTestRunner;