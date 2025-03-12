import React from 'react';
import { 
  Paper, Box, Tabs, Tab, List, ListItem, 
  ListItemIcon, ListItemText, IconButton, Chip,
  Divider, CircularProgress, Typography
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import ArticleIcon from '@mui/icons-material/Article';
import { useShopifyContext } from '../../context/indexing/ShopifyContext';
import useShopifyActions from '../../hooks/indexing/useShopifyActions';

const ShopifyContentList = () => {
  const { state } = useShopifyContext();
  const { tab, loading, stats } = state;
  const { setTab, handleDeleteItem, getFilteredContent } = useShopifyActions();
  
  const filteredContent = getFilteredContent();

  return (
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
  );
};

export default ShopifyContentList;