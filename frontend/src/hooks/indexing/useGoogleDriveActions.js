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
        const extension = file.name.split('.').pop().toLowerCase();
        fileTypes[extension] = (fileTypes[extension] || 0) + 1;
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
   */
  const handleIndexFolder = useCallback(async () => {
    dispatch({ type: ACTIONS.SET_INDEXING, payload: true });
    dispatch({ type: ACTIONS.SET_ERROR, payload: null });
    dispatch({ type: ACTIONS.SET_SUCCESS, payload: null });
    
    try {
      const response = await indexApi.indexGoogleDrive(state.folderId || null);
      dispatch({ 
        type: ACTIONS.SET_SUCCESS, 
        payload: `Successfully indexed ${response.files_processed || 0} files from Google Drive.` 
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
  }, [state.folderId, dispatch, fetchFiles]);
  
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
    clearError,
    clearSuccess,
    handleIndexFolder,
    handleDeleteFile
  };
};

export default useGoogleDriveActions;