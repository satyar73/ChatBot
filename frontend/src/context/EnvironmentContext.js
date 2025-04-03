import React, { createContext, useContext, useEffect } from 'react';
import config from '../config/config.js';

// Create context
const EnvironmentContext = createContext(config);

/**
 * Provider component for environment variables
 * Also handles updating document title and other environment-specific elements
 */
export const EnvironmentProvider = ({ children }) => {
  useEffect(() => {
    // Update document title
    document.title = `${config.COMPANY_NAME} ChatBot`;
    
    // Update any manifest.json properties that depend on environment variables
    // This is only for development purposes, as the manifest.json file should be
    // generated during the build process with the correct values
    try {
      const manifestLink = document.querySelector('link[rel="manifest"]');
      if (manifestLink && manifestLink.href.startsWith('http')) {
        fetch(manifestLink.href)
          .then(response => response.json())
          .then(manifest => {
            manifest.short_name = `${config.COMPANY_NAME} ChatBot`;
            manifest.name = `${config.COMPANY_NAME} RAG-enabled Marketing ChatBot`;
            
            // Create a blob URL for the updated manifest
            const blob = new Blob([JSON.stringify(manifest)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            // Update the manifest link
            const newLink = document.createElement('link');
            newLink.rel = 'manifest';
            newLink.href = url;
            manifestLink.parentNode.replaceChild(newLink, manifestLink);
          })
          .catch(error => console.error('Error updating manifest:', error));
      }
    } catch (error) {
      console.error('Error updating environment-specific elements:', error);
    }
  }, []);

  return (
    <EnvironmentContext.Provider value={config}>
      {children}
    </EnvironmentContext.Provider>
  );
};

// Custom hook to use the environment context
export const useEnvironment = () => useContext(EnvironmentContext);

export default EnvironmentContext;