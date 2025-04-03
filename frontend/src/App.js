import React from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import ChatPage from './pages/ChatPage';
import TestingPage from './pages/TestingPage';
import GoogleDriveIndexingPage from './pages/GoogleDriveIndexingPage';
import ShopifyIndexingPage from './pages/ShopifyIndexingPage';
import CreateSlidesPage from './pages/CreateSlidesPage';
import CreateDocumentPage from './pages/CreateDocumentPage';
import DiagnosticsPage from './pages/DiagnosticsPage';
import AppLayout from './components/AppLayout';
import { EnvironmentProvider } from './context/EnvironmentContext';

function App() {
  console.log('App rendering with HashRouter');
  
  return (
    <EnvironmentProvider>
      <Router>
        <Routes>
          <Route path="/" element={<AppLayout />}>
            <Route index element={<ChatPage />} />
            <Route path="testing" element={<TestingPage />} />
            <Route path="gdrive-indexing" element={<GoogleDriveIndexingPage />} />
            <Route path="shopify-indexing" element={<ShopifyIndexingPage />} />
            <Route path="create-document" element={<CreateDocumentPage />} />
            {/* Keeping route for backward compatibility, but pointing to new component */}
            <Route path="create-slides" element={<CreateDocumentPage />} />
            <Route path="diagnostics" element={<DiagnosticsPage />} />
            {/* Redirects for backwards compatibility */}
            <Route path="network-test" element={<DiagnosticsPage />} />
            <Route path="simple-test" element={<DiagnosticsPage />} />
          </Route>
        </Routes>
      </Router>
    </EnvironmentProvider>
  );
}

export default App;