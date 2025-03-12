import React from 'react';
import { Grid, Card, CardContent, Typography } from '@mui/material';
import { useShopifyContext } from '../../context/indexing/ShopifyContext';

const ShopifyStats = () => {
  const { state } = useShopifyContext();
  const { stats } = state;

  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Typography variant="h6" color="text.secondary" gutterBottom>
              Total Items
            </Typography>
            <Typography variant="h3">{stats.totalItems}</Typography>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={6} md={4}>
        <Card sx={{ bgcolor: 'primary.light', color: 'white' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Products
            </Typography>
            <Typography variant="h3">{stats.products}</Typography>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={6} md={4}>
        <Card sx={{ bgcolor: 'secondary.light', color: 'white' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Blog Articles
            </Typography>
            <Typography variant="h3">{stats.articles}</Typography>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default ShopifyStats;