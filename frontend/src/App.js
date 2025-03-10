import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import AppLayout from './components/AppLayout';
import ChatPage from './pages/ChatPage';
import TestingPage from './pages/TestingPage';
import GoogleDriveIndexingPage from './pages/GoogleDriveIndexingPage';
import ShopifyIndexingPage from './pages/ShopifyIndexingPage';

// Create a custom theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#2e7d32',
    },
    secondary: {
      main: '#f50057',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          <Route path="/" element={<AppLayout />}>
            <Route index element={<ChatPage />} />
            <Route path="testing" element={<TestingPage />} />
            <Route path="gdrive-indexing" element={<GoogleDriveIndexingPage />} />
            <Route path="shopify-indexing" element={<ShopifyIndexingPage />} />
          </Route>
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;