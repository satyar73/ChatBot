import React, { createContext, useReducer, useContext } from 'react';

// Initial state
const initialState = {
  loading: false,
  indexing: false,
  content: [],
  shopifyDomain: '',
  error: null,
  success: null,
  tab: 0, // 0: All, 1: Products, 2: Articles
  stats: {
    totalItems: 0,
    products: 0,
    articles: 0
  }
};

// Action types
export const ACTIONS = {
  SET_LOADING: 'SET_LOADING',
  SET_INDEXING: 'SET_INDEXING',
  SET_CONTENT: 'SET_CONTENT',
  SET_SHOPIFY_DOMAIN: 'SET_SHOPIFY_DOMAIN',
  SET_ERROR: 'SET_ERROR',
  SET_SUCCESS: 'SET_SUCCESS',
  SET_TAB: 'SET_TAB',
  SET_STATS: 'SET_STATS',
  REMOVE_ITEM: 'REMOVE_ITEM'
};

// Reducer for state management
const shopifyReducer = (state, action) => {
  switch (action.type) {
    case ACTIONS.SET_LOADING:
      return { ...state, loading: action.payload };
    
    case ACTIONS.SET_INDEXING:
      return { ...state, indexing: action.payload };
    
    case ACTIONS.SET_CONTENT:
      return { ...state, content: action.payload };
    
    case ACTIONS.SET_SHOPIFY_DOMAIN:
      return { ...state, shopifyDomain: action.payload };
    
    case ACTIONS.SET_ERROR:
      return { ...state, error: action.payload };
    
    case ACTIONS.SET_SUCCESS:
      return { ...state, success: action.payload };
    
    case ACTIONS.SET_TAB:
      return { ...state, tab: action.payload };
    
    case ACTIONS.SET_STATS:
      return { ...state, stats: action.payload };
    
    case ACTIONS.REMOVE_ITEM:
      return { 
        ...state, 
        content: state.content.filter(item => item.id !== action.payload) 
      };
    
    default:
      return state;
  }
};

// Create the context
export const ShopifyContext = createContext();

// Provider component
export const ShopifyProvider = ({ children }) => {
  const [state, dispatch] = useReducer(shopifyReducer, initialState);

  return (
    <ShopifyContext.Provider value={{ state, dispatch }}>
      {children}
    </ShopifyContext.Provider>
  );
};

// Custom hook for using the context
export const useShopifyContext = () => {
  const context = useContext(ShopifyContext);
  if (!context) {
    throw new Error('useShopifyContext must be used within a ShopifyProvider');
  }
  return context;
};