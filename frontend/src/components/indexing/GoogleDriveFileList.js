import React from 'react';
import { 
  Paper, Typography, Box, List, ListItem, 
  ListItemIcon, ListItemText, Divider
} from '@mui/material';
import DescriptionIcon from '@mui/icons-material/Description';
import { useGoogleDriveContext } from '../../context/indexing/GoogleDriveContext';

const GoogleDriveFileList = () => {
  // Safe access to state
  const contextValue = useGoogleDriveContext();
  const state = contextValue?.state || { files: [], loading: false };
  
  // Debug output to console
  console.log('GoogleDriveFileList rendering with state:', state);

  return (
    <Paper elevation={3} sx={{ flex: 1, overflow: 'auto', p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Google Drive Files {state.loading && "(Loading...)"}
      </Typography>
      
      {(!state.files || state.files.length === 0) ? (
        <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1 }}>
          <Typography>No files have been indexed yet.</Typography>
        </Box>
      ) : (
        <List>
          {state.files.map((file, index) => (
            <React.Fragment key={index}>
              <ListItem>
                <ListItemIcon>
                  <DescriptionIcon />
                </ListItemIcon>
                <ListItemText
                  primary={file?.title || "Untitled Document"}
                  secondary={
                    <span>
                      File ID: {file?.id || "Unknown"}<br/>
                      {file?.url && <a href={file.url} target="_blank" rel="noopener noreferrer">View in Drive</a>}
                    </span>
                  }
                />
              </ListItem>
              {index < state.files.length - 1 && <Divider />}
            </React.Fragment>
          ))}
        </List>
      )}
    </Paper>
  );
};

export default GoogleDriveFileList;