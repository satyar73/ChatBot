import React, { useState, useRef } from 'react';
import { 
  Box, Paper, Typography, Button, TextField, 
  CircularProgress, Table, TableBody, TableCell, 
  TableContainer, TableHead, TableRow, Alert,
  FormControlLabel, Switch, Card, CardContent,
  InputAdornment, IconButton, Tooltip
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import FileOpenIcon from '@mui/icons-material/FileOpen';
import FolderIcon from '@mui/icons-material/Folder';
import { chatApi } from '../services/api';

const TestingPage = () => {
  const [testResults, setTestResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [customTestFile, setCustomTestFile] = useState('');
  const [showComparison, setShowComparison] = useState(true);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const fileInputRef = useRef(null);
  
  // New state for single prompt testing
  const [testMode, setTestMode] = useState('file'); // 'file' or 'prompt'
  const [singlePrompt, setSinglePrompt] = useState('');
  const [expectedAnswer, setExpectedAnswer] = useState('');

  const runTests = async (testFile = null) => {
    setLoading(true);
    setError(null);
    try {
      let results;
      
      // Handle single prompt test mode
      if (testMode === 'prompt') {
        // Use the single prompt test endpoint
        results = await chatApi.runTests({
          prompt: singlePrompt,
          expected_result: expectedAnswer
        });
      }
      // Handle file upload mode
      else if (uploadedFile) {
        // If we have an uploaded file, use that instead of a filepath
        const formData = new FormData();
        formData.append('csv_file', uploadedFile);
        
        // Add the similarity threshold parameter
        const params = new URLSearchParams({ similarity_threshold: 0.7 });
        
        // Use a direct fetch to handle FormData with file upload
        const response = await fetch(`/chat/batch-test?${params.toString()}`, {
          method: 'POST',
          body: formData,
          // Important: Do not set Content-Type header, browser will set it with boundary
        });
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => null);
          const errorMessage = errorData?.detail || response.statusText;
          throw new Error(`Server returned ${response.status}: ${errorMessage}`);
        }
        
        results = await response.json();
      } 
      // Handle file path mode
      else {
        // Otherwise use the API with filepath
        results = await chatApi.runTests(testFile);
      }
      
      // If we get a single test result (not a batch), convert it to the expected format
      if (results && !results.results && !Array.isArray(results)) {
        results = {
          total_tests: 1,
          passed: results.passed ? 1 : 0,
          failed: results.passed ? 0 : 1,
          pass_rate: results.passed ? 100 : 0,
          results: [results]
        };
      }
      
      setTestResults(results);
    } catch (err) {
      console.error('Test execution error:', err);
      setError('Failed to run tests: ' + (err.message || 'Unknown error'));
      setTestResults(null);
    } finally {
      setLoading(false);
      // Clear the uploaded file after testing
      setUploadedFile(null);
      // Also clear the file input field
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };
  
  // Handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    // Check if the file is a CSV
    if (!file.name.endsWith('.csv')) {
      setUploadError('Please select a CSV file');
      return;
    }
    
    // File size check (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      setUploadError('File size must be less than 5MB');
      return;
    }
    
    setUploadedFile(file);
    setUploadError(null);
    
    // Update the text field with the filename
    setCustomTestFile(file.name + ' (uploaded)');
  };
  
  // Trigger file input click
  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  // Calculate overall metrics if results exist
  const calculateMetrics = () => {
    if (!testResults || !testResults.results) return null;
    
    const results = testResults.results;
    let totalTests = results.length;
    let passedTests = 0;
    
    // Handle different API response formats
    if ('passed' in testResults) {
      // If the API directly provides the pass count
      passedTests = testResults.passed || 0;
    } else {
      // Otherwise calculate from individual results
      results.forEach(test => {
        // Some APIs use test.passed directly
        if (test.passed !== undefined) {
          if (test.passed) passedTests++;
        } 
        // Others require checking the score
        else if (test.similarity_score !== undefined && test.similarity_score >= 0.7) {
          passedTests++;
        }
        // For APIs that use the detailed analysis structure
        else if (test.detailed_analysis?.comparison?.rag_score >= 0.7) {
          passedTests++;
        }
      });
    }
    
    // Calculate pass rate safely
    const passRate = totalTests > 0 ? (passedTests / totalTests * 100) : 0;
    
    return {
      totalTests,
      passedTests,
      failedTests: totalTests - passedTests,
      passRate: passRate.toFixed(1),
      // Fall back to API-provided values if available
      passRateFromApi: testResults.pass_rate !== undefined ? 
                   testResults.pass_rate.toFixed(1) : passRate.toFixed(1),
    };
  };

  const metrics = calculateMetrics();

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
      <Typography variant="h5">Chat Testing Interface</Typography>
      
      {/* Test Controls */}
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
                onChange={(e) => setShowComparison(e.target.checked)} 
              />
            }
            label="Show RAG vs. Standard Comparison"
          />
        </Box>
      </Paper>
      
      {/* Loading State */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}>
          <CircularProgress />
        </Box>
      )}
      
      {/* Error Message */}
      {error && (
        <Alert severity="error" sx={{ my: 2 }}>
          {error}
        </Alert>
      )}
      
      {/* Test Results */}
      {testResults && metrics && (
        <>
          {/* Metrics Summary */}
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <Card sx={{ minWidth: 200, flex: 1 }}>
              <CardContent>
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  Total Tests
                </Typography>
                <Typography variant="h3">{metrics.totalTests}</Typography>
              </CardContent>
            </Card>
            
            <Card sx={{ minWidth: 200, flex: 1, bgcolor: 'success.light', color: 'white' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Tests Passed
                </Typography>
                <Typography variant="h3">{metrics.passRate}%</Typography>
                <Typography variant="body2">
                  {metrics.passedTests} of {metrics.totalTests} tests
                </Typography>
              </CardContent>
            </Card>
            
            <Card sx={{ minWidth: 200, flex: 1, bgcolor: 'error.light', color: 'white' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Tests Failed
                </Typography>
                <Typography variant="h3">{metrics.failedTests}</Typography>
                <Typography variant="body2">
                  {metrics.totalTests > 0 ? 
                    (100 - parseFloat(metrics.passRate)).toFixed(1) + '%' : 
                    '0%'
                  }
                </Typography>
              </CardContent>
            </Card>
          </Box>
          
          {/* Detailed Results Table */}
          <TableContainer component={Paper} sx={{ mt: 3 }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>#</TableCell>
                  <TableCell>Query</TableCell>
                  <TableCell>Expected</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Score</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {testResults.results.map((test, index) => {
                  // Extract fields from the test result based on API format
                  const query = test.query || test.prompt || '';
                  const expected = test.expected_answer || test.expected_result || '';
                  const isPassed = test.passed !== undefined ? test.passed : (test.similarity_score >= 0.7);
                  
                  // Get the score - handle different API formats
                  const score = test.similarity_score !== undefined 
                    ? test.similarity_score 
                    : (test.detailed_analysis?.comparison?.rag_score || 0);
                  
                  // Format the score for display
                  const scoreDisplay = typeof score === 'number' 
                    ? score.toFixed(2) 
                    : 'N/A';
                    
                  return (
                    <TableRow 
                      key={index}
                      sx={{ 
                        '&:nth-of-type(odd)': { bgcolor: 'action.hover' },
                        bgcolor: !isPassed ? 'error.light' : undefined
                      }}
                    >
                      <TableCell>{index + 1}</TableCell>
                      <TableCell>{query}</TableCell>
                      <TableCell>{expected}</TableCell>
                      <TableCell>
                        <Box 
                          sx={{
                            color: isPassed ? 'success.main' : 'error.main',
                            fontWeight: 'bold'
                          }}
                        >
                          {isPassed ? 'PASS' : 'FAIL'}
                        </Box>
                      </TableCell>
                      <TableCell
                        sx={{ 
                          color: isPassed ? 'success.main' : 'error.main',
                          fontWeight: 'bold'
                        }}
                      >
                        {scoreDisplay}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </>
      )}
    </Box>
  );
};

export default TestingPage;