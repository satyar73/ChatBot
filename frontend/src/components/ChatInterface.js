/**
 * ChatInterface Component - A dynamic chat application built with React and Material-UI.
 * 
 * This component serves as the entry point for the chat interface,
 * but uses a more modular approach with separation of concerns:
 * 
 * 1. Context (ChatContext) - Manages state and state transitions
 * 2. Custom Hooks (useChatActions) - Contains business logic
 * 3. Component files - Pure UI components with minimal logic
 * 
 * The code is structured according to best practices:
 * - State management is centralized in a context
 * - Business logic is in custom hooks
 * - UI components are focused on rendering
 * - Each file has a single responsibility
 */

import React from 'react';
import { Box } from '@mui/material';
import { ChatProvider, useChatContext } from '../context/ChatContext';
import ErrorAlert from './ErrorAlert';
import ChatHeader from './ChatHeader';
import SystemPromptEditor from './SystemPromptEditor';
import ChatMessageList from './ChatMessageList';
import ChatInput from './ChatInput';

/**
 * Main ChatInterface component that brings together all the chat components
 * This component acts primarily as a composition layer with minimal logic
 */
const ChatInterfaceContent = () => {
  const { state } = useChatContext();

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <ErrorAlert />
      <ChatHeader />
      {state.showSystemPrompt && <SystemPromptEditor />}
      <ChatMessageList />
      <ChatInput />
    </Box>
  );
};

/**
 * Wrapper component that provides context to the chat interface
 */
const ChatInterface = () => {
  return (
    <ChatProvider>
      <ChatInterfaceContent />
    </ChatProvider>
  );
};

export default ChatInterface;