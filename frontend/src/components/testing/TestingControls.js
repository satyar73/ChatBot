import React from 'react';
import { 
  Box, Paper, Typography, TextField, Button, 
  FormControlLabel, Switch, InputAdornment, 
  IconButton, Tooltip, Alert, LinearProgress,
  CircularProgress, Chip
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import FolderIcon from '@mui/icons-material/Folder';
import FileOpenIcon from '@mui/icons-material/FileOpen';
import PendingIcon from '@mui/icons-material/Pending';
import ErrorIcon from '@mui/icons-material/Error';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { useTestingContext } from '../../context/TestingContext';
import useTestingActions from '../../hooks/useTestingActions';

const TestingControls = () => {
  const { state, fileInputRef } = useTestingContext();
  const { 
    testMode, customTestFile, uploadedFile, uploadError, 
    singlePrompt, expectedAnswer, showComparison, loading,
    jobStatus, jobProgress, statusMessage, testJobId
  } = state;
  
  const { 
    setTestMode, setCustomTestFile, setSinglePrompt, 
    setExpectedAnswer, toggleShowComparison, 
    handleFileChange, handleBrowseClick, runTests,
    cleanupTestJob
  } = useTestingActions();
  
  // Cleanup test job polling on unmount
  React.useEffect(() => {
    return () => {
      if (testJobId) {
        cleanupTestJob();
      }
    };
  }, [testJobId, cleanupTestJob]);

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">Run Chat Tests</Typography>
          
          {/* Test Mode Toggle */}
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              variant={testMode === 'file' ? 'contained' : 'outlined'}
              size="small"
              onClick={() => setTestMode('file')}
            >
              File Test
            </Button>
            <Button
              variant={testMode === 'prompt' ? 'contained' : 'outlined'} 
              size="small"
              onClick={() => setTestMode('prompt')}
            >
              Single Prompt
            </Button>
          </Box>
        </Box>
        
        {/* Hidden file input */}
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept=".csv"
          onChange={handleFileChange}
        />
        
        {/* File Test Mode */}
        {testMode === 'file' && (
          <>
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-end' }}>
              <TextField
                label="Test File"
                variant="outlined"
                fullWidth
                value={customTestFile}
                onChange={(e) => setCustomTestFile(e.target.value)}
                helperText={uploadError || "Upload a CSV file or enter a server path"}
                error={!!uploadError}
                disabled={loading}
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <Tooltip title="Browse for CSV file">
                        <IconButton
                          onClick={handleBrowseClick}
                          edge="end"
                          disabled={loading}
                        >
                          <FolderIcon />
                        </IconButton>
                      </Tooltip>
                    </InputAdornment>
                  ),
                  startAdornment: uploadedFile && (
                    <InputAdornment position="start">
                      <FileOpenIcon color="primary" />
                    </InputAdornment>
                  )
                }}
              />
              <Button
                variant="contained"
                color="primary"
                startIcon={uploadedFile ? <UploadFileIcon /> : <PlayArrowIcon />}
                onClick={() => runTests(uploadedFile ? null : customTestFile || null)}
                disabled={loading || !!uploadError}
              >
                {uploadedFile ? 'Upload & Test' : 'Run Tests'}
              </Button>
            </Box>
            
            {/* Upload indicator */}
            {uploadedFile && (
              <Alert severity="info" sx={{ mt: 1 }}>
                File selected: {uploadedFile.name} ({(uploadedFile.size / 1024).toFixed(2)} KB)
              </Alert>
            )}
            
            {/* Job status and progress bar */}
            {jobStatus && (
              <Box sx={{ mt: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    {statusMessage || `Job status: ${jobStatus}`}
                  </Typography>
                  <Chip 
                    label={jobStatus} 
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
                </Box>
                
                <LinearProgress 
                  variant="determinate" 
                  value={jobProgress} 
                  sx={{ height: 10, borderRadius: 1 }} 
                />
                
                {testJobId && (
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                    Job ID: {testJobId}
                  </Typography>
                )}
              </Box>
            )}
          </>
        )}
        
        {/* Single Prompt Test Mode */}
        {testMode === 'prompt' && (
          <>
            <TextField
              label="Prompt"
              variant="outlined"
              fullWidth
              multiline
              rows={3}
              value={singlePrompt}
              onChange={(e) => setSinglePrompt(e.target.value)}
              placeholder="Enter your test prompt here..."
              disabled={loading}
            />
            
            <TextField
              label="Expected Answer (Optional)"
              variant="outlined"
              fullWidth
              multiline
              rows={3}
              value={expectedAnswer}
              onChange={(e) => setExpectedAnswer(e.target.value)}
              placeholder="Enter the expected answer for comparison (optional)"
              disabled={loading}
              helperText="Leave empty to just see the response without comparison"
            />
            
            <Button
              variant="contained"
              color="primary"
              startIcon={<PlayArrowIcon />}
              onClick={() => runTests()}
              disabled={loading || !singlePrompt.trim()}
              sx={{ alignSelf: 'flex-end' }}
            >
              Test Prompt
            </Button>
          </>
        )}
        
        <FormControlLabel
          control={
            <Switch 
              checked={showComparison} 
              onChange={toggleShowComparison} 
            />
          }
          label="Show RAG vs. Standard Comparison"
        />
      </Box>
    </Paper>
  );
};

export default TestingControls;