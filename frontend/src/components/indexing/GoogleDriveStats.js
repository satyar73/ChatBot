import React from 'react';
import { Grid, Card, CardContent, Typography, Box, Chip } from '@mui/material';
import { useGoogleDriveContext } from '../../context/indexing/GoogleDriveContext';

const GoogleDriveStats = () => {
  const { state } = useGoogleDriveContext();
  const { stats } = state;

  return (
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
  );
};

export default GoogleDriveStats;