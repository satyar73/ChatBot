import React from 'react';
import { Box, Paper, Typography, TextField, Button, CircularProgress } from '@mui/material';
import StorefrontIcon from '@mui/icons-material/Storefront';
import { useShopifyContext } from '../../context/indexing/ShopifyContext';
import useShopifyActions from '../../hooks/indexing/useShopifyActions';

const ShopifyControls = () => {
  const { state } = useShopifyContext();
  const { shopifyDomain, indexing } = state;
  const { setShopifyDomain, handleIndexShopify } = useShopifyActions();

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <Typography variant="h6">Index Shopify Store</Typography>
        
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-end' }}>
          <TextField
            label="Shopify Domain"
            variant="outlined"
            fullWidth
            value={shopifyDomain}
            onChange={(e) => setShopifyDomain(e.target.value)}
            helperText="Enter your Shopify store domain (e.g., your-store.myshopify.com)"
            disabled={indexing}
            required
          />
          <Button
            variant="contained"
            color="primary"
            startIcon={<StorefrontIcon />}
            onClick={handleIndexShopify}
            disabled={indexing}
          >
            {indexing ? 'Indexing...' : 'Index Store'}
          </Button>
        </Box>
        
        {indexing && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <CircularProgress size={24} />
            <Typography>Indexing content from Shopify. This may take a while...</Typography>
          </Box>
        )}
      </Box>
    </Paper>
  );
};

export default ShopifyControls;