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
      }
      // Handle file upload mode
      else if (state.uploadedFile) {
        // If we have an uploaded file, use that instead of a filepath
        const formData = new FormData();
        formData.append('csv_file', state.uploadedFile);
        
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
        results = await chatApi.runTests(testFile || state.customTestFile);
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
      
      dispatch({ type: ACTIONS.SET_TEST_RESULTS, payload: results });
    } catch (err) {
      console.error('Test execution error:', err);
      dispatch({ 
        type: ACTIONS.SET_ERROR, 
        payload: 'Failed to run tests: ' + (err.message || 'Unknown error') 
      });
    } finally {
      dispatch({ type: ACTIONS.SET_LOADING, payload: false });
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
    calculateMetrics
  };
};

export default useTestingActions;