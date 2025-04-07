import { useCallback } from 'react';
import { indexApi } from '../../services/api';
import { useGoogleDriveContext, ACTIONS } from '../../context/indexing/GoogleDriveContext';

/**
 * Custom hook for Google Drive indexing actions
 */
const useGoogleDriveActions = () => {
  const { state, dispatch } = useGoogleDriveContext();
  
  /**
   * Fetch the list of indexed files from Google Drive
   */
  const fetchFiles = useCallback(async () => {
    dispatch({ type: ACTIONS.SET_LOADING, payload: true });
    dispatch({ type: ACTIONS.SET_ERROR, payload: null });
    
    try {
      const response = await indexApi.listGoogleDriveFiles();
      dispatch({ type: ACTIONS.SET_FILES, payload: response.files || [] });
      
      // Calculate stats
      const fileTypes = {};
      response.files.forEach(file => {
        const fileName = file.title || file.name || "";
        if (fileName && fileName.includes('.')) {
          const extension = fileName.split('.').pop().toLowerCase();
          fileTypes[extension] = (fileTypes[extension] || 0) + 1;
        } else {
          fileTypes['other'] = (fileTypes['other'] || 0) + 1;
        }
      });
      
      dispatch({ 
        type: ACTIONS.SET_STATS, 
        payload: {
          totalFiles: response.files.length,
          fileTypes
        } 
      });
    } catch (err) {
      dispatch({ 
        type: ACTIONS.SET_ERROR, 
        payload: 'Failed to fetch files: ' + (err.message || 'Unknown error') 
      });
    } finally {
      dispatch({ type: ACTIONS.SET_LOADING, payload: false });
    }
  }, [dispatch]);
  
  /**
   * Set the folder ID
   */
  const setFolderId = useCallback((id) => {
    dispatch({ type: ACTIONS.SET_FOLDER_ID, payload: id });
  }, [dispatch]);
  
  /**
   * Set recursive option
   */
  const setRecursive = useCallback((value) => {
    dispatch({ type: ACTIONS.SET_RECURSIVE, payload: value });
  }, [dispatch]);
  
  /**
   * Set enhanced slides option
   */
  const setEnhancedSlides = useCallback((value) => {
    dispatch({ type: ACTIONS.SET_ENHANCED_SLIDES, payload: value });
  }, [dispatch]);
  
  /**
   * Set namespace
   */
  const setNamespace = useCallback((value) => {
    dispatch({ type: ACTIONS.SET_NAMESPACE, payload: value });
  }, [dispatch]);
  
  /**
   * Clear error message
   */
  const clearError = useCallback(() => {
    dispatch({ type: ACTIONS.SET_ERROR, payload: null });
  }, [dispatch]);
  
  /**
   * Clear success message
   */
  const clearSuccess = useCallback(() => {
    dispatch({ type: ACTIONS.SET_SUCCESS, payload: null });
  }, [dispatch]);
  
  /**
   * Index a Google Drive folder
   * @param {string} namespace - Optional namespace for the index
   */
  const handleIndexFolder = useCallback(async (namespace = null) => {
    dispatch({ type: ACTIONS.SET_INDEXING, payload: true });
    dispatch({ type: ACTIONS.SET_ERROR, payload: null });
    dispatch({ type: ACTIONS.SET_SUCCESS, payload: null });
    
    try {
      const response = await indexApi.indexGoogleDrive(
        state.folderId || null, 
        state.recursive,
        state.enhancedSlides,
        namespace
      );
      const fileCount = response.files_processed || 0;
      const chunkCount = response.chunks_indexed || 0;
      
      // Format the success message including namespace info if present
      let successMessage = `Successfully indexed ${fileCount} files (${chunkCount} chunks) from Google Drive`;
      if (namespace) {
        successMessage += ` in namespace '${namespace}'`;
      }
      successMessage += ".";
      
      dispatch({ 
        type: ACTIONS.SET_SUCCESS, 
        payload: successMessage
      });
      // Refresh the file list
      fetchFiles();
    } catch (err) {
      dispatch({ 
        type: ACTIONS.SET_ERROR, 
        payload: 'Failed to index Google Drive: ' + (err.message || 'Unknown error') 
      });
    } finally {
      dispatch({ type: ACTIONS.SET_INDEXING, payload: false });
    }
  }, [state.folderId, state.recursive, state.enhancedSlides, dispatch, fetchFiles]);
  
  /**
   * Delete a file from the index
   */
  const handleDeleteFile = useCallback(async (fileId) => {
    dispatch({ type: ACTIONS.SET_LOADING, payload: true });
    dispatch({ type: ACTIONS.SET_ERROR, payload: null });
    
    try {
      await indexApi.deleteDocument(fileId);
      dispatch({ type: ACTIONS.REMOVE_FILE, payload: fileId });
      dispatch({ 
        type: ACTIONS.SET_SUCCESS, 
        payload: 'File deleted successfully' 
      });
    } catch (err) {
      dispatch({ 
        type: ACTIONS.SET_ERROR, 
        payload: 'Failed to delete file: ' + (err.message || 'Unknown error') 
      });
    } finally {
      dispatch({ type: ACTIONS.SET_LOADING, payload: false });
    }
  }, [dispatch]);
  
  return {
    fetchFiles,
    setFolderId,
    setRecursive,
    setEnhancedSlides,
    setNamespace,
    clearError,
    clearSuccess,
    handleIndexFolder,
    handleDeleteFile
  };
};

export default useGoogleDriveActions;