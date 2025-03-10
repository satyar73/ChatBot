import React, { useState, useEffect } from 'react';
import { 
  Box, Paper, Typography, Button, TextField, 
  CircularProgress, Alert, List, ListItem, 
  ListItemText, ListItemIcon, Divider, IconButton,
  Chip, Grid, Card, CardContent, Tabs, Tab
} from '@mui/material';
import StorefrontIcon from '@mui/icons-material/Storefront';
import ArticleIcon from '@mui/icons-material/Article';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import DeleteIcon from '@mui/icons-material/Delete';
import RefreshIcon from '@mui/icons-material/Refresh';
import { indexApi } from '../services/api';

const ShopifyIndexingPage = () => {
  const [loading, setLoading] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [content, setContent] = useState([]);
  const [shopifyDomain, setShopifyDomain] = useState('');
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [tab, setTab] = useState(0);
  const [stats, setStats] = useState({
    totalItems: 0,
    products: 0,
    articles: 0
  });

  // Fetch the list of indexed content when the component mounts
  useEffect(() => {
    fetchContent();
  }, []);

  const fetchContent = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await indexApi.listShopifyContent();
      setContent(response.content || []);
      
      // Calculate stats
      const products = response.content.filter(item => item.type === 'product').length;
      const articles = response.content.filter(item => item.type === 'article').length;
      
      setStats({
        totalItems: response.content.length,
        products,
        articles
      });
    } catch (err) {
      setError('Failed to fetch Shopify content: ' + (err.message || 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

  const handleIndexShopify = async () => {
    if (!shopifyDomain && !shopifyDomain.includes('.')) {
      setError('Please enter a valid Shopify domain');
      return;
    }
    
    setIndexing(true);
    setError(null);
    setSuccess(null);
    try {
      const response = await indexApi.indexShopify(shopifyDomain);
      setSuccess(
        `Successfully indexed ${response.products_indexed || 0} products and ${response.articles_indexed || 0} articles from Shopify.`
      );
      // Refresh the content list
      fetchContent();
    } catch (err) {
      setError('Failed to index Shopify: ' + (err.message || 'Unknown error'));
    } finally {
      setIndexing(false);
    }
  };

  const handleDeleteItem = async (itemId) => {
    setLoading(true);
    setError(null);
    try {
      await indexApi.deleteDocument(itemId);
      // Remove the item from the local state
      setContent(prevContent => prevContent.filter(item => item.id !== itemId));
      setSuccess('Item deleted successfully');
    } catch (err) {
      setError('Failed to delete item: ' + (err.message || 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

  // Filter content based on the selected tab
  const filteredContent = content.filter(item => {
    if (tab === 0) return true; // All
    if (tab === 1) return item.type === 'product';
    if (tab === 2) return item.type === 'article';
    return true;
  });

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
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
      
      {/* Stats Cards */}
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
      
      {/* Indexing Controls */}
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
      
      {/* Success/Error Messages */}
      {error && (
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      {success && (
        <Alert severity="success" onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}
      
      {/* Content List */}
      <Paper elevation={3} sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={tab} 
            onChange={(e, newValue) => setTab(newValue)}
            indicatorColor="primary"
            textColor="primary"
          >
            <Tab label={`All (${stats.totalItems})`} />
            <Tab label={`Products (${stats.products})`} />
            <Tab label={`Articles (${stats.articles})`} />
          </Tabs>
        </Box>
        
        <Box sx={{ flex: 1, overflow: 'auto', p: 0 }}>
          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
              <CircularProgress />
            </Box>
          )}
          
          {!loading && filteredContent.length === 0 ? (
            <Box sx={{ p: 3, textAlign: 'center' }}>
              <Typography color="text.secondary">
                No content found. Use the controls above to index your Shopify store.
              </Typography>
            </Box>
          ) : (
            <List>
              {filteredContent.map((item, index) => (
                <React.Fragment key={item.id || index}>
                  <ListItem
                    secondaryAction={
                      <IconButton edge="end" onClick={() => handleDeleteItem(item.id)}>
                        <DeleteIcon />
                      </IconButton>
                    }
                  >
                    <ListItemIcon>
                      {item.type === 'product' ? (
                        <ShoppingCartIcon color="primary" />
                      ) : (
                        <ArticleIcon color="secondary" />
                      )}
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {item.title || item.name}
                          <Chip 
                            size="small" 
                            label={item.type} 
                            color={item.type === 'product' ? 'primary' : 'secondary'}
                          />
                        </Box>
                      }
                      secondary={
                        <>
                          <Typography variant="body2" component="span">
                            {item.description ? `${item.description.substring(0, 120)}...` : 'No description'}
                          </Typography>
                          {item.updated_at && (
                            <Typography variant="caption" component="div" color="text.secondary">
                              Updated: {new Date(item.updated_at).toLocaleString()}
                            </Typography>
                          )}
                        </>
                      }
                    />
                  </ListItem>
                  {index < filteredContent.length - 1 && <Divider />}
                </React.Fragment>
              ))}
            </List>
          )}
        </Box>
      </Paper>
    </Box>
  );
};

export default ShopifyIndexingPage;