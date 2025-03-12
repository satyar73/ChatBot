import React from 'react';
import { 
  Box, Card, CardContent, Typography, 
  TableContainer, Table, TableHead, TableRow, 
  TableCell, TableBody, Paper, CircularProgress
} from '@mui/material';
import { useTestingContext } from '../../context/TestingContext';
import useTestingActions from '../../hooks/useTestingActions';

const TestingResults = () => {
  const { state } = useTestingContext();
  const { testResults, loading } = state;
  const { calculateMetrics } = useTestingActions();
  
  const metrics = calculateMetrics();
  
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}>
        <CircularProgress />
      </Box>
    );
  }
  
  if (!testResults || !metrics) {
    return null;
  }

  return (
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
  );
};

export default TestingResults;