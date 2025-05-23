import React from 'react';
import { Box, Paper, Typography, TextField, Button, CircularProgress, FormControlLabel, Switch, FormGroup } from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import { useGoogleDriveContext } from '../../context/indexing/GoogleDriveContext';
import useGoogleDriveActions from '../../hooks/indexing/useGoogleDriveActions';

const GoogleDriveControls = () => {
  const { state } = useGoogleDriveContext();
  const { folderId, indexing, recursive, enhancedSlides } = state;
  const { setFolderId, setRecursive, setEnhancedSlides, handleIndexFolder } = useGoogleDriveActions();

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Typography variant="h6">Index Google Drive Folder</Typography>
        
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-end' }}>
          <TextField
            label="Folder ID (Optional)"
            variant="outlined"
            fullWidth
            value={folderId}
            onChange={(e) => setFolderId(e.target.value)}
            helperText="Leave empty to index all accessible files"
            disabled={indexing}
          />
          <Button
            variant="contained"
            color="primary"
            startIcon={<UploadFileIcon />}
            onClick={handleIndexFolder}
            disabled={indexing}
          >
            {indexing ? 'Indexing...' : 'Index Folder'}
          </Button>
        </Box>
        
        <FormGroup>
          <FormControlLabel 
            control={
              <Switch 
                checked={recursive} 
                onChange={(e) => setRecursive(e.target.checked)}
                disabled={indexing}
              />
            } 
            label="Process subfolders recursively" 
          />
          <FormControlLabel 
            control={
              <Switch 
                checked={enhancedSlides} 
                onChange={(e) => setEnhancedSlides(e.target.checked)}
                disabled={indexing}
              />
            } 
            label="Use enhanced slide processing (GPT-4 Vision)" 
          />
        </FormGroup>
        
        {indexing && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <CircularProgress size={24} />
            <Typography>Indexing files from Google Drive. This may take a while...</Typography>
          </Box>
        )}
      </Box>
    </Paper>
  );
};

export default GoogleDriveControls;