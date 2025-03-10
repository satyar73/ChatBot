import React, { useState, useEffect } from 'react';
import { 
  Box, Paper, Typography, Button, TextField, 
  CircularProgress, Alert, List, ListItem, 
  ListItemText, ListItemIcon, Divider, IconButton,
  Chip, Grid, Card, CardContent
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import FolderIcon from '@mui/icons-material/Folder';
import DescriptionIcon from '@mui/icons-material/Description';
import DeleteIcon from '@mui/icons-material/Delete';
import RefreshIcon from '@mui/icons-material/Refresh';
import { indexApi } from '../services/api';

const GoogleDriveIndexingPage = () => {
  const [loading, setLoading] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [files, setFiles] = useState([]);
  const [folderId, setFolderId] = useState('');
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [stats, setStats] = useState({
    totalFiles: 0,
    fileTypes: {}
  });

  // Fetch the list of indexed files when the component mounts
  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await indexApi.listGoogleDriveFiles();
      setFiles(response.files || []);
      
      // Calculate stats
      const fileTypes = {};
      response.files.forEach(file => {
        const extension = file.name.split('.').pop().toLowerCase();
        fileTypes[extension] = (fileTypes[extension] || 0) + 1;
      });
      
      setStats({
        totalFiles: response.files.length,
        fileTypes
      });
    } catch (err) {
      setError('Failed to fetch files: ' + (err.message || 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

  const handleIndexFolder = async () => {
    setIndexing(true);
    setError(null);
    setSuccess(null);
    try {
      const response = await indexApi.indexGoogleDrive(folderId || null);
      setSuccess(`Successfully indexed ${response.files_processed || 0} files from Google Drive.`);
      // Refresh the file list
      fetchFiles();
    } catch (err) {
      setError('Failed to index Google Drive: ' + (err.message || 'Unknown error'));
    } finally {
      setIndexing(false);
    }
  };

  const handleDeleteFile = async (fileId) => {
    setLoading(true);
    setError(null);
    try {
      await indexApi.deleteDocument(fileId);
      // Remove the file from the local state
      setFiles(prevFiles => prevFiles.filter(file => file.id !== fileId));
      setSuccess('File deleted successfully');
    } catch (err) {
      setError('Failed to delete file: ' + (err.message || 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

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
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h5">Google Drive Indexing</Typography>
        <Button
          startIcon={<RefreshIcon />}
          onClick={fetchFiles}
          disabled={loading || indexing}
        >
          Refresh
        </Button>
      </Box>
      
      {/* Stats Cards */}
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Indexed Files
              </Typography>
              <Typography variant="h3">{stats.totalFiles}</Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2 }}>
                {Object.entries(stats.fileTypes).map(([type, count]) => (
                  <Chip 
                    key={type} 
                    label={`${type}: ${count}`} 
                    size="small" 
                    color={
                      type === 'pdf' ? 'error' : 
                      type === 'docx' || type === 'doc' ? 'primary' : 
                      type === 'pptx' || type === 'ppt' ? 'warning' : 
                      'default'
                    }
                  />
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Indexing Controls */}
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
          
          {indexing && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <CircularProgress size={24} />
              <Typography>Indexing files from Google Drive. This may take a while...</Typography>
            </Box>
          )}
        </Box>
      </Paper>
      
      {/* Success/Error Messages */}
      {error && (
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      {success && (
        <Alert severity="success" onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}
      
      {/* File List */}
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
    </Box>
  );
};

export default GoogleDriveIndexingPage;