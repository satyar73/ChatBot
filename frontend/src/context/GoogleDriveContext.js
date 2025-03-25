import React, { createContext, useState, useContext, useCallback } from 'react';
import { indexApi } from '../services/api';

// Create the context
const GoogleDriveContext = createContext(null);

// Create a provider component
export const GoogleDriveProvider = ({ children }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [files, setFiles] = useState([]);
  const [indexingResult, setIndexingResult] = useState(null);
  
  // Fetch the list of Google Drive files that have been indexed
  const fetchFiles = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      console.log('Fetching Google Drive files...');
      const result = await indexApi.listGoogleDriveFiles();
      console.log('Fetched Google Drive files:', result);
      
      if (result && result.status === 'success') {
        setFiles(result.files || []);
      } else {
        setError(result?.message || 'Failed to fetch files');
      }
    } catch (err) {
      console.error('Error fetching Google Drive files:', err);
      setError(err?.response?.data?.message || err.message || 'An error occurred while fetching files');
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  // Index a Google Drive folder
  const indexFolder = useCallback(async (folderId, recursive = true, enhancedSlides = false) => {
    setIsLoading(true);
    setError(null);
    setIndexingResult(null);
    
    try {
      console.log(`Indexing Google Drive folder: ${folderId}, recursive: ${recursive}, enhancedSlides: ${enhancedSlides}`);
      const result = await indexApi.indexGoogleDrive(folderId, recursive, enhancedSlides);
      console.log('Indexing result:', result);
      
      setIndexingResult(result);
      
      if (result && result.status === 'success') {
        // Fetch updated files list after successful indexing
        fetchFiles();
      } else {
        setError(result?.message || 'Indexing failed');
      }
    } catch (err) {
      console.error('Error indexing Google Drive:', err);
      setError(err?.response?.data?.message || err.message || 'An error occurred during indexing');
    } finally {
      setIsLoading(false);
    }
  }, [fetchFiles]);
  
  // Clear error state
  const clearError = useCallback(() => {
    setError(null);
  }, []);
  
  // Context value
  const contextValue = {
    isLoading,
    error,
    files,
    indexingResult,
    fetchFiles,
    indexFolder,
    clearError
  };
  
  return (
    <GoogleDriveContext.Provider value={contextValue}>
      {children}
    </GoogleDriveContext.Provider>
  );
};

// Custom hook to use the Google Drive context
export const useGoogleDrive = () => {
  const context = useContext(GoogleDriveContext);
  
  if (!context) {
    throw new Error('useGoogleDrive must be used within a GoogleDriveProvider');
  }
  
  return context;
};

export default GoogleDriveContext;