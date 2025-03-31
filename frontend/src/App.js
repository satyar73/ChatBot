import React from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import ChatPage from './pages/ChatPage';
import TestingPage from './pages/TestingPage';
import GoogleDriveIndexingPage from './pages/GoogleDriveIndexingPage';
import ShopifyIndexingPage from './pages/ShopifyIndexingPage';
import CreateSlidesPage from './pages/CreateSlidesPage';
import NetworkTest from './pages/NetworkTest';
import SimpleTest from './pages/SimpleTest';
import AppLayout from './components/AppLayout';

function App() {
  console.log('App rendering with HashRouter');
  
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AppLayout />}>
          <Route index element={<ChatPage />} />
          <Route path="testing" element={<TestingPage />} />
          <Route path="gdrive-indexing" element={<GoogleDriveIndexingPage />} />
          <Route path="shopify-indexing" element={<ShopifyIndexingPage />} />
          <Route path="create-slides" element={<CreateSlidesPage />} />
          <Route path="network-test" element={<NetworkTest />} />
          <Route path="simple-test" element={<SimpleTest />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;