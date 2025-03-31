import React, { useState } from 'react';
import { 
  Box, Typography, Button, Paper, Alert, CircularProgress, 
  TextField, Link, FormControlLabel, Switch, Divider
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

function CreateSlidesPage() {
  const [file, setFile] = useState(null);
  const [title, setTitle] = useState('Q&A Presentation');
  const [email, setEmail] = useState('');
  const [authorName, setAuthorName] = useState('');
  const [useRag, setUseRag] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(null);

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
  
  const handleUseRagChange = (e) => {
    setUseRag(e.target.checked);
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
    formData.append('use_rag', useRag);
    
    // If email is provided, add it to the request
    if (email) {
      formData.append('owner_email', email); // Keep the parameter name for API compatibility
    }
    
    // If author name is provided, add it to the request
    if (authorName) {
      formData.append('author_name', authorName);
    }

    // Log what we're sending
    console.log('Sending file:', file);
    console.log('Title:', title);
    console.log('Email:', email);
    console.log('Author Name:', authorName);
    console.log('Use RAG:', useRag);
    
    try {
      // Use more detailed debugging
      console.log('Starting request to /chat/create-slides');
      const response = await fetch('/chat/create-slides', {
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
      setError(`Failed to create slides: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', p: 2 }}>
      <Typography variant="h4" gutterBottom>
        Create Google Slides from Q&A
      </Typography>
      <Typography variant="body1" paragraph>
        Upload a CSV file with questions and answers to generate a Google Slides presentation.
      </Typography>
      
      <Paper elevation={3} sx={{ p: 3, mt: 2, maxWidth: 600 }}>
        <form onSubmit={handleSubmit}>
          <Typography variant="h6" gutterBottom>
            CSV File Format:
          </Typography>
          <Typography variant="body2" sx={{ mb: 2, fontFamily: 'monospace', bgcolor: '#f5f5f5', p: 1 }}>
            question,answer<br />
            "What is X?","X is Y"<br />
            "How does Z work?","Z works by..."
          </Typography>
          
          <TextField 
            fullWidth
            label="Presentation Title"
            value={title}
            onChange={handleTitleChange}
            margin="normal"
            variant="outlined"
          />
          
          <TextField 
            fullWidth
            label="Author Name (shown on title slide)"
            value={authorName}
            onChange={handleAuthorNameChange}
            margin="normal"
            variant="outlined"
            placeholder="John Doe"
          />
          
          <TextField 
            fullWidth
            label="Share With (Email)"
            value={email}
            onChange={handleEmailChange}
            margin="normal"
            variant="outlined"
            placeholder="your.email@example.com"
            helperText="Enter your email to get editor access to the presentation"
          />
          
          <Box sx={{ mt: 2, mb: 2 }}>
            <Divider sx={{ mb: 2 }} />
            <FormControlLabel
              control={
                <Switch
                  checked={useRag}
                  onChange={handleUseRagChange}
                  color="primary"
                />
              }
              label="Use RAG to generate answers from knowledge base"
            />
            <Typography variant="caption" display="block" sx={{ ml: 2, mt: 0.5 }}>
              When enabled, answers will be generated using our AI with Retrieval Augmented Generation.
              When disabled, the answers from the CSV file will be used directly.
            </Typography>
          </Box>
          
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
            {loading ? <CircularProgress size={24} /> : 'Create Slides'}
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
              Presentation created successfully!
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              The presentation is now publicly accessible with the link below.
              {success.shared_with && <Box component="span" display="block" mt={0.5}>
                Shared with: {success.shared_with}
              </Box>}
              {success.author && <Box component="span" display="block" mt={0.5}>
                Author listed as: {success.author}
              </Box>}
              <Box component="span" display="block" mt={0.5}>
                {success.used_rag ? 
                  "Answers were generated using Retrieval Augmented Generation." :
                  "Answers were taken directly from the CSV file."}
              </Box>
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
              Open presentation in Google Slides
            </Link>
          </Alert>
        )}
      </Paper>
    </Box>
  );
}

export default CreateSlidesPage;