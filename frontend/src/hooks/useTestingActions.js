import { useCallback } from 'react';
import { chatApi } from '../services/api';
import { useTestingContext, ACTIONS } from '../context/TestingContext';

/**
 * Custom hook for testing actions
 */
const useTestingActions = () => {
  const { state, dispatch, fileInputRef } = useTestingContext();
  
  /**
   * Set the test mode (file or prompt)
   */
  const setTestMode = useCallback((mode) => {
    dispatch({ type: ACTIONS.SET_TEST_MODE, payload: mode });
  }, [dispatch]);
  
  /**
   * Set the custom test file path
   */
  const setCustomTestFile = useCallback((filePath) => {
    dispatch({ type: ACTIONS.SET_CUSTOM_TEST_FILE, payload: filePath });
  }, [dispatch]);
  
  /**
   * Set the single prompt text
   */
  const setSinglePrompt = useCallback((prompt) => {
    dispatch({ type: ACTIONS.SET_SINGLE_PROMPT, payload: prompt });
  }, [dispatch]);
  
  /**
   * Set the expected answer text
   */
  const setExpectedAnswer = useCallback((answer) => {
    dispatch({ type: ACTIONS.SET_EXPECTED_ANSWER, payload: answer });
  }, [dispatch]);
  
  /**
   * Toggle show comparison setting
   */
  const toggleShowComparison = useCallback(() => {
    dispatch({ 
      type: ACTIONS.SET_SHOW_COMPARISON, 
      payload: !state.showComparison 
    });
  }, [state.showComparison, dispatch]);
  
  /**
   * Clear error message
   */
  const clearError = useCallback(() => {
    dispatch({ type: ACTIONS.SET_ERROR, payload: null });
  }, [dispatch]);
  
  /**
   * Handle file selection
   */
  const handleFileChange = useCallback((event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    // Check if the file is a CSV
    if (!file.name.endsWith('.csv')) {
      dispatch({ 
        type: ACTIONS.SET_UPLOAD_ERROR, 
        payload: 'Please select a CSV file' 
      });
      return;
    }
    
    // File size check (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      dispatch({ 
        type: ACTIONS.SET_UPLOAD_ERROR, 
        payload: 'File size must be less than 5MB' 
      });
      return;
    }
    
    dispatch({ type: ACTIONS.SET_UPLOADED_FILE, payload: file });
    dispatch({ type: ACTIONS.SET_UPLOAD_ERROR, payload: null });
    
    // Update the text field with the filename
    dispatch({ 
      type: ACTIONS.SET_CUSTOM_TEST_FILE, 
      payload: file.name + ' (uploaded)' 
    });
  }, [dispatch]);
  
  /**
   * Trigger file input click
   */
  const handleBrowseClick = useCallback(() => {
    fileInputRef.current?.click();
  }, [fileInputRef]);
  
  /**
   * Run tests
   */
  const runTests = useCallback(async (testFile = null) => {
    dispatch({ type: ACTIONS.SET_LOADING, payload: true });
    dispatch({ type: ACTIONS.SET_ERROR, payload: null });
    
    try {
      let results;
      
      // Handle single prompt test mode
      if (state.testMode === 'prompt') {
        // Use the single prompt test endpoint
        results = await chatApi.runTests({
          prompt: state.singlePrompt,
          expected_result: state.expectedAnswer
        });
        
        // Convert single test result to batch format
        results = {
          total_tests: 1,
          passed: results.passed ? 1 : 0,
          failed: results.passed ? 0 : 1,
          pass_rate: results.passed ? 100 : 0,
          results: [results]
        };
        
        dispatch({ type: ACTIONS.SET_TEST_RESULTS, payload: results });
      }
      // Handle file upload mode with long-running job
      else if (state.uploadedFile) {
        // Start a background job for the test
        const jobResponse = await chatApi.startBatchTest(state.uploadedFile, 0.7);
        
        if (!jobResponse || !jobResponse.job_id) {
          throw new Error('Invalid job response from server');
        }
        
        dispatch({ type: ACTIONS.SET_TEST_JOB_ID, payload: jobResponse.job_id });
        dispatch({ type: ACTIONS.SET_JOB_STATUS, payload: 'pending' });
        
        // Show job started message
        dispatch({ 
          type: ACTIONS.SET_STATUS_MESSAGE, 
          payload: `Test job started with ID: ${jobResponse.job_id}. Status will update automatically.` 
        });
        
        // Record start time for progress calculation
        const startTime = Date.now();
        
        // Start polling for job status - set up a polling interval
        const pollInterval = setInterval(async () => {
          try {
            const jobStatus = await chatApi.getTestJobStatus(jobResponse.job_id);
            
            // Update job status in state
            dispatch({ type: ACTIONS.SET_JOB_STATUS, payload: jobStatus.status });
            
            // Debug job status response
            console.log('Job status response:', jobStatus);
            
            // Update progress if available
            if (jobStatus.progress !== undefined) {
              console.log('Setting job progress to:', jobStatus.progress);
              dispatch({ type: ACTIONS.SET_JOB_PROGRESS, payload: jobStatus.progress });
            } else {
              console.log('Progress not available in jobStatus, using default progress indicator');
              // Always set some progress for running jobs, even if backend doesn't provide it
              if (jobStatus.status === 'running') {
                // Just set a default value that increases over time
                const timeBasedProgress = Math.min(
                  10 + Math.floor((Date.now() - startTime) / 1000), // +1% per second
                  90 // Cap at 90%
                );
                console.log('Using time-based progress:', timeBasedProgress);
                dispatch({ type: ACTIONS.SET_JOB_PROGRESS, payload: timeBasedProgress });
              }
            }
            
            // If job complete or failed, stop polling and update results
            if (jobStatus.status === 'completed') {
              clearInterval(pollInterval);
              
              // Set the poll interval to null to prevent memory leaks
              dispatch({ type: ACTIONS.SET_POLL_INTERVAL, payload: null });
              
              // Set the test results from the job result
              if (jobStatus.result) {
                dispatch({ type: ACTIONS.SET_TEST_RESULTS, payload: jobStatus.result });
                
                // Show completion message with any file paths from the message
                const completionMessage = jobStatus.message && typeof jobStatus.message === 'string' 
                  ? jobStatus.message  // Use the message from the server which includes file paths
                  : `Test job completed in ${jobStatus.duration_seconds?.toFixed(1) || '?'} seconds with ${jobStatus.result.passed || 0} passed tests.`;
                
                dispatch({ 
                  type: ACTIONS.SET_STATUS_MESSAGE, 
                  payload: completionMessage
                });
              }
              
              // Mark as not loading
              dispatch({ type: ACTIONS.SET_LOADING, payload: false });
            }
            else if (jobStatus.status === 'failed') {
              clearInterval(pollInterval);
              
              // Set the poll interval to null to prevent memory leaks
              dispatch({ type: ACTIONS.SET_POLL_INTERVAL, payload: null });
              
              // Show error message
              dispatch({ 
                type: ACTIONS.SET_ERROR, 
                payload: `Test job failed: ${jobStatus.error?.message || 'Unknown error'}` 
              });
              
              // Mark as not loading
              dispatch({ type: ACTIONS.SET_LOADING, payload: false });
            }
            else {
              // Update status message with progress
              const statusMessage = jobStatus.message && typeof jobStatus.message === 'string'
                  ? `Current status message: ${jobStatus.message}`
                  : '';
              dispatch({
                type: ACTIONS.SET_STATUS_MESSAGE, 
                payload: `Test job running... (${jobStatus.progress || 0}% complete) - ${statusMessage}`
              });
            }
          } catch (error) {
            console.error('Error polling job status:', error);
            // Don't stop polling on a temporary error
            dispatch({ 
              type: ACTIONS.SET_STATUS_MESSAGE, 
              payload: `Error checking job status: ${error.message}. Will retry...` 
            });
          }
        }, 5000); // Poll every 5 seconds
        
        // Save poll interval for cleanup
        dispatch({ type: ACTIONS.SET_POLL_INTERVAL, payload: pollInterval });
      } 
      // Handle file path mode (direct, not async)
      else {
        // Use the API with filepath directly (shorter tests)
        results = await chatApi.runTests(testFile || state.customTestFile);
        
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
        
        dispatch({ type: ACTIONS.SET_TEST_RESULTS, payload: results });
        dispatch({ type: ACTIONS.SET_LOADING, payload: false });
      }
    } catch (err) {
      console.error('Test execution error:', err);
      dispatch({ 
        type: ACTIONS.SET_ERROR, 
        payload: 'Failed to run tests: ' + (err.message || 'Unknown error') 
      });
      dispatch({ type: ACTIONS.SET_LOADING, payload: false });
    } finally {
      // Reset upload state
      dispatch({ type: ACTIONS.RESET_UPLOAD_STATE });
      // Also clear the file input field
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  }, [
    state.testMode, 
    state.singlePrompt, 
    state.expectedAnswer, 
    state.uploadedFile, 
    state.customTestFile, 
    dispatch, 
    fileInputRef
  ]);
  
  // Cleanup function for test job polling
  const cleanupTestJob = useCallback(() => {
    if (state.pollInterval) {
      clearInterval(state.pollInterval);
      dispatch({ type: ACTIONS.SET_POLL_INTERVAL, payload: null });
    }
  }, [state.pollInterval, dispatch]);
  
  /**
   * Calculate metrics from test results
   */
  const calculateMetrics = useCallback(() => {
    if (!state.testResults || !state.testResults.results) return null;
    
    const results = state.testResults.results;
    let totalTests = results.length;
    let passedTests = 0;
    
    // Handle different API response formats
    if ('passed' in state.testResults) {
      // If the API directly provides the pass count
      passedTests = state.testResults.passed || 0;
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
      passRateFromApi: state.testResults.pass_rate !== undefined ? 
                   state.testResults.pass_rate.toFixed(1) : passRate.toFixed(1),
    };
  }, [state.testResults]);
  
  return {
    setTestMode,
    setCustomTestFile,
    setSinglePrompt,
    setExpectedAnswer,
    toggleShowComparison,
    clearError,
    handleFileChange,
    handleBrowseClick,
    runTests,
    calculateMetrics,
    cleanupTestJob
  };
};

export default useTestingActions;