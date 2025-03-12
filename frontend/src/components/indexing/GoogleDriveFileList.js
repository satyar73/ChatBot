import React from 'react';
import { 
  Paper, Typography, Box, List, ListItem, 
  ListItemIcon, ListItemText, IconButton, 
  Divider, CircularProgress 
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import FolderIcon from '@mui/icons-material/Folder';
import DescriptionIcon from '@mui/icons-material/Description';
import { useGoogleDriveContext } from '../../context/indexing/GoogleDriveContext';
import useGoogleDriveActions from '../../hooks/indexing/useGoogleDriveActions';

const GoogleDriveFileList = () => {
  const { state } = useGoogleDriveContext();
  const { files, loading } = state;
  const { handleDeleteFile } = useGoogleDriveActions();

  const getFileTypeIcon = (fileName) => {
    const extension = fileName.split('.').pop().toLowerCase();
    switch (extension) {
      case 'pdf':
        return <DescriptionIcon color="error" />;
      case 'doc':
      case 'docx':
        return <DescriptionIcon color="primary" />;
      case 'ppt':
      case 'pptx':
        return <DescriptionIcon color="warning" />;
      case 'txt':
      case 'md':
        return <DescriptionIcon color="action" />;
      default:
        return <DescriptionIcon />;
    }
  };

  return (
    <Paper elevation={3} sx={{ flex: 1, overflow: 'auto' }}>
      <Typography variant="h6" sx={{ p: 2, borderBottom: '1px solid #e0e0e0' }}>
        Indexed Files {loading && <CircularProgress size={20} sx={{ ml: 2 }} />}
      </Typography>
      
      {files.length === 0 && !loading ? (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <Typography color="text.secondary">
            No files have been indexed yet. Use the controls above to index your Google Drive files.
          </Typography>
        </Box>
      ) : (
        <List>
          {files.map((file, index) => (
            <React.Fragment key={file.id || index}>
              <ListItem
                secondaryAction={
                  <IconButton edge="end" onClick={() => handleDeleteFile(file.id)}>
                    <DeleteIcon />
                  </IconButton>
                }
              >
                <ListItemIcon>
                  {file.is_folder ? <FolderIcon color="primary" /> : getFileTypeIcon(file.name)}
                </ListItemIcon>
                <ListItemText
                  primary={file.name}
                  secondary={
                    <Box>
                      <Typography variant="body2" component="span">
                        {file.path || 'Root folder'}
                      </Typography>
                      {file.last_modified && (
                        <Typography variant="caption" component="div" color="text.secondary">
                          Modified: {new Date(file.last_modified).toLocaleString()}
                        </Typography>
                      )}
                    </Box>
                  }
                />
              </ListItem>
              {index < files.length - 1 && <Divider />}
            </React.Fragment>
          ))}
        </List>
      )}
    </Paper>
  );
};

export default GoogleDriveFileList;