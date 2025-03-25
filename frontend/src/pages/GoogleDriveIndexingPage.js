import React, { useState, useEffect } from 'react';
import { GoogleDriveProvider, useGoogleDrive } from '../context/GoogleDriveContext';
import { Box, Typography, Paper, Button, TextField, FormControlLabel, Checkbox, 
         Snackbar, Alert, CircularProgress, Divider, List, ListItem, ListItemText, 
         ListItemIcon, Link, Chip } from '@mui/material';
import FolderIcon from '@mui/icons-material/Folder';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import RefreshIcon from '@mui/icons-material/Refresh';

// Main component wrapper with context provider
const GoogleDriveIndexingPage = () => {
  return (
    <GoogleDriveProvider>
      <GoogleDriveIndexingContent />
    </GoogleDriveProvider>
  );
};

// Component that uses the context
const GoogleDriveIndexingContent = () => {
  const { 
    isLoading, 
    error, 
    files, 
    indexingResult, 
    fetchFiles, 
    indexFolder, 
    clearError 
  } = useGoogleDrive();
  
  const [folderId, setFolderId] = useState('');
  const [recursive, setRecursive] = useState(true);
  const [enhancedSlides, setEnhancedSlides] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  
  // Fetch files on component mount
  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);
  
  // Show success message when indexing completes
  useEffect(() => {
    if (indexingResult && indexingResult.status === 'success') {
      setSuccessMessage(`Successfully indexed ${indexingResult.files_processed || 0} files into ${indexingResult.chunks_indexed || 0} chunks.`);
    }
  }, [indexingResult]);
  
  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    indexFolder(folderId || null, recursive, enhancedSlides);
  };
  
  // Handle closing the error alert
  const handleCloseError = () => {
    clearError();
  };
  
  // Handle closing the success message
  const handleCloseSuccess = () => {
    setSuccessMessage('');
  };
  
  return (
    <Box sx={{ padding: 3, maxWidth: 1200, margin: '0 auto' }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Google Drive Indexing
      </Typography>
      
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          Index a Google Drive Folder
        </Typography>
        
        <Box component="form" onSubmit={handleSubmit} sx={{ mt: 2 }}>
          <TextField
            label="Google Drive Folder ID (optional)"
            value={folderId}
            onChange={(e) => setFolderId(e.target.value)}
            fullWidth
            margin="normal"
            helperText="Leave empty to use the default folder configured in the backend"
            disabled={isLoading}
          />
          
          <Box sx={{ mt: 2 }}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={recursive}
                  onChange={(e) => setRecursive(e.target.checked)}
                  disabled={isLoading}
                />
              }
              label="Process subfolders recursively"
            />
          </Box>
          
          <Box sx={{ mt: 1 }}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={enhancedSlides}
                  onChange={(e) => setEnhancedSlides(e.target.checked)}
                  disabled={isLoading}
                />
              }
              label="Use enhanced slide processing with GPT-4 Vision (slower but better quality)"
            />
          </Box>
          
          <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              type="submit"
              variant="contained"
              color="primary"
              disabled={isLoading}
              startIcon={isLoading ? <CircularProgress size={20} /> : null}
            >
              {isLoading ? 'Indexing...' : 'Start Indexing'}
            </Button>
          </Box>
        </Box>
      </Paper>
      
      <Paper elevation={3} sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            Indexed Files
            {files && files.length > 0 && (
              <Chip 
                label={`${files.length} files`} 
                size="small" 
                color="primary" 
                sx={{ ml: 1 }} 
              />
            )}
          </Typography>
          
          <Button
            startIcon={<RefreshIcon />}
            onClick={fetchFiles}
            disabled={isLoading}
          >
            Refresh
          </Button>
        </Box>
        
        <Divider sx={{ mb: 2 }} />
        
        {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
        )}
        
        {!isLoading && files && files.length === 0 && (
          <Typography color="textSecondary" sx={{ my: 2, textAlign: 'center' }}>
            No indexed files found. Start indexing a Google Drive folder to see files here.
          </Typography>
        )}
        
        {!isLoading && files && files.length > 0 && (
          <List>
            {files.map((file) => (
              <ListItem key={file.id} divider>
                <ListItemIcon>
                  <InsertDriveFileIcon />
                </ListItemIcon>
                <ListItemText
                  primary={file.title || 'Untitled'}
                  secondary={
                    <>
                      {`Size: ${Math.round(file.size / 1024)} KB`}
                      {file.url && (
                        <Box component="span" sx={{ ml: 2 }}>
                          <Link 
                            href={file.url} 
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            View in Google Drive
                          </Link>
                        </Box>
                      )}
                    </>
                  }
                />
              </ListItem>
            ))}
          </List>
        )}
      </Paper>
      
      {/* Error Snackbar */}
      <Snackbar open={!!error} autoHideDuration={6000} onClose={handleCloseError}>
        <Alert onClose={handleCloseError} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
      
      {/* Success Snackbar */}
      <Snackbar open={!!successMessage} autoHideDuration={6000} onClose={handleCloseSuccess}>
        <Alert onClose={handleCloseSuccess} severity="success" sx={{ width: '100%' }}>
          {successMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default GoogleDriveIndexingPage;