import React from 'react';
import { Box, Typography, Button } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { useShopifyContext } from '../../context/indexing/ShopifyContext';
import useShopifyActions from '../../hooks/indexing/useShopifyActions';

const ShopifyHeader = () => {
  const { state } = useShopifyContext();
  const { loading, indexing } = state;
  const { fetchContent } = useShopifyActions();

  return (
    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <Typography variant="h5">Shopify Indexing</Typography>
      <Button
        startIcon={<RefreshIcon />}
        onClick={fetchContent}
        disabled={loading || indexing}
      >
        Refresh
      </Button>
    </Box>
  );
};

export default ShopifyHeader;