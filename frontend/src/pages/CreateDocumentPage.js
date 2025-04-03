import React, { useState } from 'react';
import { 
  Box, Typography, Button, Paper, Alert, CircularProgress, 
  TextField, Link, FormControl, InputLabel, Select, MenuItem,
  Divider, Tab, Tabs
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import SlideshowIcon from '@mui/icons-material/Slideshow';
import DescriptionIcon from '@mui/icons-material/Description';

function CreateDocumentPage() {
  const [file, setFile] = useState(null);
  const [title, setTitle] = useState('Generated Document');
  const [email, setEmail] = useState('');
  const [authorName, setAuthorName] = useState('');
  const [documentType, setDocumentType] = useState('slides');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(null);
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
    setDocumentType(newValue === 0 ? 'slides' : 'docs');
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    
    // Make sure it's a CSV file
    if (selectedFile && !selectedFile.name.endsWith('.csv')) {
      setError('Please select a CSV file');
      setFile(null);
      return;
    }
    
    setFile(selectedFile);
    setError('');
  };

  const handleTitleChange = (e) => {
    setTitle(e.target.value);
  };

  const handleEmailChange = (e) => {
    setEmail(e.target.value);
  };
  
  const handleAuthorNameChange = (e) => {
    setAuthorName(e.target.value);
  };
  
  const handleDocumentTypeChange = (e) => {
    setDocumentType(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a CSV file');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', title);
    formData.append('document_type', documentType);
    
    // If email is provided, add it to the request
    if (email) {
      formData.append('owner_email', email);
    }
    
    // If author name is provided, add it to the request (useful for slides title page)
    if (authorName) {
      formData.append('author_name', authorName);
    }

    // Log what we're sending
    console.log('Sending file:', file);
    console.log('Title:', title);
    console.log('Email:', email);
    console.log('Author Name:', authorName);
    console.log('Document Type:', documentType);
    
    try {
      // Use more detailed debugging
      console.log('Starting request to /chat/create-document');
      const response = await fetch('/chat/create-document', {
        method: 'POST',
        body: formData,
      });
      console.log('Response status:', response.status);
      console.log('Response headers:', response.headers);

      if (!response.ok) {
        throw new Error(`Error ${response.status}: ${await response.text()}`);
      }

      const data = await response.json();
      setSuccess(data);
    } catch (err) {
      setError(`Failed to create document: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const slidesFormatExample = `question,format
"What is marketing attribution?","Title: Marketing Attribution Overview
Body:
- Definition and importance
- Key metrics
- Implementation steps"

"What models are available?","Title: Attribution Models
Body:
1. First-touch attribution
2. Last-touch attribution
3. Multi-touch attribution"`;

  const docsFormatExample = `question,format
"What is marketing attribution?","# Marketing Attribution Overview

## Definition and Importance
- Key point 1
- Key point 2

## Common Attribution Models
1. First-touch attribution
2. Last-touch attribution"

"How can marketing attribution improve ROI?","# Improving Marketing ROI with Attribution

## Measurement Strategies
- Strategy 1
- Strategy 2"`;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', p: 2 }}>
      <Typography variant="h4" gutterBottom>
        Create Documents with RAG
      </Typography>
      <Typography variant="body1" paragraph>
        Upload a CSV file with questions and format templates to generate Google Slides or Docs with RAG-generated content.
      </Typography>
      
      <Paper elevation={3} sx={{ p: 3, mt: 2, maxWidth: 800 }}>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange} 
          sx={{ mb: 3 }}
          centered
        >
          <Tab 
            icon={<SlideshowIcon />} 
            label="Google Slides" 
            id="document-type-tab-0"
            aria-controls="document-type-tabpanel-0"
          />
          <Tab 
            icon={<DescriptionIcon />} 
            label="Google Docs" 
            id="document-type-tab-1"
            aria-controls="document-type-tabpanel-1"
          />
        </Tabs>

        <form onSubmit={handleSubmit}>
          <Typography variant="h6" gutterBottom>
            CSV File Format:
          </Typography>
          <Typography 
            variant="body2" 
            sx={{ 
              mb: 2, 
              fontFamily: 'monospace', 
              bgcolor: '#f5f5f5', 
              p: 1,
              whiteSpace: 'pre-wrap',
              overflowX: 'auto'
            }}
          >
            {documentType === 'slides' ? slidesFormatExample : docsFormatExample}
          </Typography>
          
          <TextField 
            fullWidth
            label={`${documentType === 'slides' ? 'Presentation' : 'Document'} Title`}
            value={title}
            onChange={handleTitleChange}
            margin="normal"
            variant="outlined"
          />
          
          {documentType === 'slides' && (
            <TextField 
              fullWidth
              label="Author Name (shown on title slide)"
              value={authorName}
              onChange={handleAuthorNameChange}
              margin="normal"
              variant="outlined"
              placeholder="John Doe"
            />
          )}
          
          <TextField 
            fullWidth
            label="Share With (Email)"
            value={email}
            onChange={handleEmailChange}
            margin="normal"
            variant="outlined"
            placeholder="your.email@example.com"
            helperText="Enter your email to get editor access to the document"
          />
          
          <Box sx={{ my: 3, textAlign: 'center' }}>
            <input
              accept=".csv"
              style={{ display: 'none' }}
              id="csv-file-upload"
              type="file"
              onChange={handleFileChange}
            />
            <label htmlFor="csv-file-upload">
              <Button
                variant="contained"
                component="span"
                startIcon={<CloudUploadIcon />}
              >
                Select CSV File
              </Button>
            </label>
            {file && (
              <Typography variant="body2" sx={{ mt: 1 }}>
                Selected: {file.name}
              </Typography>
            )}
          </Box>
          
          <Button
            type="submit"
            variant="contained"
            color="primary"
            fullWidth
            disabled={loading || !file}
            sx={{ mt: 2 }}
          >
            {loading ? (
              <CircularProgress size={24} />
            ) : (
              `Create ${documentType === 'slides' ? 'Slides' : 'Document'}`
            )}
          </Button>
        </form>
        
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
        
        {success && (
          <Alert severity="success" sx={{ mt: 2 }}>
            <Typography variant="body1">
              {success.document_type === 'presentation' ? 'Presentation' : 'Document'} created successfully!
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              Your {success.document_type} is now accessible with the link below.
              {success.shared_with && <Box component="span" display="block" mt={0.5}>
                Shared with: {success.shared_with}
              </Box>}
              {success.author && <Box component="span" display="block" mt={0.5}>
                Author listed as: {success.author}
              </Box>}
              {success.note && <Box component="span" display="block" mt={1} sx={{ fontStyle: 'italic' }}>
                {success.note}
              </Box>}
            </Typography>
            <Link 
              href={success.url} 
              target="_blank" 
              rel="noopener noreferrer"
              sx={{ display: 'block', mt: 1 }}
            >
              Open {success.document_type} in Google Drive
            </Link>
          </Alert>
        )}
      </Paper>
    </Box>
  );
}

export default CreateDocumentPage;