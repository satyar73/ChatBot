import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || '';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const chatApi = {
  sendMessage: async (message, responseMode = "rag", sessionId = null) => {
    try {
      // Generate a random session ID if not provided
      const session_id = sessionId || `session_${Math.random().toString(36).substring(2, 15)}`;
      
      console.log(`Sending chat request with session_id: ${session_id}, mode: ${responseMode}`);
      
      // Map UI response mode to backend mode parameter
      let mode = "rag";
      if (responseMode === "standard") {
        mode = "no_rag";
      } else if (responseMode === "compare") {
        mode = "compare";
      }
      
      const response = await api.post('/chat', { 
        message,
        session_id,
        mode
      });
      
      // Return the response data along with the session ID for future requests
      return { 
        ...response.data,
        session_id 
      };
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  },
  
  runTests: async (testData = null) => {
    try {
      // Case 1: Single prompt test with expected result
      if (testData && typeof testData === 'object' && testData.prompt) {
        const response = await api.post('/chat/test', {
          prompt: testData.prompt,
          expected_result: testData.expected_result || ""
        });
        return response.data;
      }
      // Case 2: String filepath
      else if (typeof testData === 'string') {
        // Single test case with a string path
        if (!testData.includes('.csv')) {
          const response = await api.post('/chat/test', { test_file: testData });
          return response.data;
        } 
        // Batch test with filepath
        else {
          const response = await api.post('/chat/batch-test', 
            { test_file: testData },
            { params: { similarity_threshold: 0.7 } }
          );
          return response.data;
        }
      }
      // Case 3: No test data (default test)
      else {
        const response = await api.post('/chat/test');
        return response.data;
      }
    } catch (error) {
      console.error('Error running chat tests:', error);
      throw error;
    }
  }
};

export const indexApi = {
  // Cache Management
  clearCache: async (olderThanDays = null) => {
    try {
      const params = olderThanDays ? { older_than_days: olderThanDays } : {};
      const response = await api.delete('/chat/cache', { params });
      return response.data;
    } catch (error) {
      console.error('Error clearing cache:', error);
      throw error;
    }
  },
  
  getCacheStats: async () => {
    try {
      const response = await api.get('/chat/cache/stats');
      return response.data;
    } catch (error) {
      console.error('Error fetching cache stats:', error);
      throw error;
    }
  },
  
  // Google Drive Indexing
  indexGoogleDrive: async (folderId = null) => {
    try {
      const response = await api.post('/index/google-drive', { folder_id: folderId });
      return response.data;
    } catch (error) {
      console.error('Error indexing Google Drive:', error);
      throw error;
    }
  },
  
  listGoogleDriveFiles: async () => {
    try {
      const response = await api.get('/index/google-drive/files');
      return response.data;
    } catch (error) {
      console.error('Error listing Google Drive files:', error);
      throw error;
    }
  },
  
  // Shopify Indexing
  indexShopify: async (shopifyDomain = null) => {
    try {
      const response = await api.post('/index/shopify', { domain: shopifyDomain });
      return response.data;
    } catch (error) {
      console.error('Error indexing Shopify:', error);
      throw error;
    }
  },
  
  listShopifyContent: async () => {
    try {
      const response = await api.get('/index/shopify/content');
      return response.data;
    } catch (error) {
      console.error('Error listing Shopify content:', error);
      throw error;
    }
  },
  
  // General index operations
  deleteDocument: async (documentId) => {
    try {
      const response = await api.delete(`/index/document/${documentId}`);
      return response.data;
    } catch (error) {
      console.error('Error deleting document:', error);
      throw error;
    }
  },
  
  healthCheck: async () => {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }
};

export default { chatApi, indexApi };