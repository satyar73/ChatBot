import React, { useEffect } from 'react';
import { Box } from '@mui/material';
import { ShopifyProvider, useShopifyContext } from '../context/indexing/ShopifyContext';
import useShopifyActions from '../hooks/indexing/useShopifyActions';
import ShopifyHeader from '../components/indexing/ShopifyHeader';
import ShopifyStats from '../components/indexing/ShopifyStats';
import ShopifyControls from '../components/indexing/ShopifyControls';
import ShopifyContentList from '../components/indexing/ShopifyContentList';
import StatusAlerts from '../components/indexing/StatusAlerts';

/**
 * Inner component that uses the context and hooks
 */
const ShopifyIndexingContent = () => {
  const { state } = useShopifyContext();
  const { fetchContent, clearError, clearSuccess } = useShopifyActions();
  const { error, success } = state;
  
  // Fetch content on component mount
  useEffect(() => {
    fetchContent();
  }, [fetchContent]);
  
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
      <ShopifyHeader />
      <ShopifyStats />
      <ShopifyControls />
      
      <StatusAlerts
        error={error}
        success={success}
        onCloseError={clearError}
        onCloseSuccess={clearSuccess}
      />
      
      <ShopifyContentList />
    </Box>
  );
};

/**
 * Main page component that provides the context
 */
const ShopifyIndexingPage = () => {
  return (
    <ShopifyProvider>
      <ShopifyIndexingContent />
    </ShopifyProvider>
  );
};

export default ShopifyIndexingPage;