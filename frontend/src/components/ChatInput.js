import React from 'react';
import { Paper, Grid, TextField, Button } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import { useChatContext } from '../context/ChatContext';
import useChatActions from '../hooks/useChatActions';

const ChatInput = () => {
  const { state } = useChatContext();
  const { input, loading } = state;
  const { sendMessage, setInput } = useChatActions();

  return (
    <Paper elevation={3} sx={{ p: 1 }}>
      <form onSubmit={sendMessage}>
        <Grid container spacing={1}>
          <Grid item xs>
            <TextField
              fullWidth
              placeholder="Type your message here..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              variant="outlined"
              disabled={loading}
              autoComplete="off"
            />
          </Grid>
          <Grid item>
            <Button
              type="submit"
              variant="contained"
              color="primary"
              endIcon={<SendIcon />}
              disabled={loading || !input.trim()}
            >
              Send
            </Button>
          </Grid>
        </Grid>
      </form>
    </Paper>
  );
};

export default ChatInput;