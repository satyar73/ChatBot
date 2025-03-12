import { useCallback } from 'react';
import { indexApi } from '../../services/api';
import { useShopifyContext, ACTIONS } from '../../context/indexing/ShopifyContext';

/**
 * Custom hook for Shopify indexing actions
 */
const useShopifyActions = () => {
  const { state, dispatch } = useShopifyContext();
  
  /**
   * Fetch the list of indexed content from Shopify
   */
  const fetchContent = useCallback(async () => {
    dispatch({ type: ACTIONS.SET_LOADING, payload: true });
    dispatch({ type: ACTIONS.SET_ERROR, payload: null });
    
    try {
      const response = await indexApi.listShopifyContent();
      dispatch({ type: ACTIONS.SET_CONTENT, payload: response.content || [] });
      
      // Calculate stats
      const products = response.content.filter(item => item.type === 'product').length;
      const articles = response.content.filter(item => item.type === 'article').length;
      
      dispatch({ 
        type: ACTIONS.SET_STATS, 
        payload: {
          totalItems: response.content.length,
          products,
          articles
        } 
      });
    } catch (err) {
      dispatch({ 
        type: ACTIONS.SET_ERROR, 
        payload: 'Failed to fetch Shopify content: ' + (err.message || 'Unknown error') 
      });
    } finally {
      dispatch({ type: ACTIONS.SET_LOADING, payload: false });
    }
  }, [dispatch]);
  
  /**
   * Set the Shopify domain
   */
  const setShopifyDomain = useCallback((domain) => {
    dispatch({ type: ACTIONS.SET_SHOPIFY_DOMAIN, payload: domain });
  }, [dispatch]);
  
  /**
   * Set the active tab
   */
  const setTab = useCallback((tabIndex) => {
    dispatch({ type: ACTIONS.SET_TAB, payload: tabIndex });
  }, [dispatch]);
  
  /**
   * Clear error message
   */
  const clearError = useCallback(() => {
    dispatch({ type: ACTIONS.SET_ERROR, payload: null });
  }, [dispatch]);
  
  /**
   * Clear success message
   */
  const clearSuccess = useCallback(() => {
    dispatch({ type: ACTIONS.SET_SUCCESS, payload: null });
  }, [dispatch]);
  
  /**
   * Index a Shopify store
   */
  const handleIndexShopify = useCallback(async () => {
    if (!state.shopifyDomain && !state.shopifyDomain.includes('.')) {
      dispatch({ 
        type: ACTIONS.SET_ERROR, 
        payload: 'Please enter a valid Shopify domain' 
      });
      return;
    }
    
    dispatch({ type: ACTIONS.SET_INDEXING, payload: true });
    dispatch({ type: ACTIONS.SET_ERROR, payload: null });
    dispatch({ type: ACTIONS.SET_SUCCESS, payload: null });
    
    try {
      const response = await indexApi.indexShopify(state.shopifyDomain);
      dispatch({ 
        type: ACTIONS.SET_SUCCESS, 
        payload: `Successfully indexed ${response.products_indexed || 0} products and ${response.articles_indexed || 0} articles from Shopify.` 
      });
      // Refresh the content list
      fetchContent();
    } catch (err) {
      dispatch({ 
        type: ACTIONS.SET_ERROR, 
        payload: 'Failed to index Shopify: ' + (err.message || 'Unknown error') 
      });
    } finally {
      dispatch({ type: ACTIONS.SET_INDEXING, payload: false });
    }
  }, [state.shopifyDomain, dispatch, fetchContent]);
  
  /**
   * Delete an item from the index
   */
  const handleDeleteItem = useCallback(async (itemId) => {
    dispatch({ type: ACTIONS.SET_LOADING, payload: true });
    dispatch({ type: ACTIONS.SET_ERROR, payload: null });
    
    try {
      await indexApi.deleteDocument(itemId);
      dispatch({ type: ACTIONS.REMOVE_ITEM, payload: itemId });
      dispatch({ 
        type: ACTIONS.SET_SUCCESS, 
        payload: 'Item deleted successfully' 
      });
    } catch (err) {
      dispatch({ 
        type: ACTIONS.SET_ERROR, 
        payload: 'Failed to delete item: ' + (err.message || 'Unknown error') 
      });
    } finally {
      dispatch({ type: ACTIONS.SET_LOADING, payload: false });
    }
  }, [dispatch]);
  
  /**
   * Get filtered content based on tab selection
   */
  const getFilteredContent = useCallback(() => {
    return state.content.filter(item => {
      if (state.tab === 0) return true; // All
      if (state.tab === 1) return item.type === 'product';
      if (state.tab === 2) return item.type === 'article';
      return true;
    });
  }, [state.content, state.tab]);
  
  return {
    fetchContent,
    setShopifyDomain,
    setTab,
    clearError,
    clearSuccess,
    handleIndexShopify,
    handleDeleteItem,
    getFilteredContent
  };
};

export default useShopifyActions;