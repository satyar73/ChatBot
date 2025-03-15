import React, { useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { 
  AppBar, Box, Drawer, Toolbar, Typography, IconButton, 
  List, ListItem, ListItemButton, ListItemIcon, ListItemText, 
  Divider, Container
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import ChatIcon from '@mui/icons-material/Chat';
import SpeedIcon from '@mui/icons-material/Speed';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';

// Width of the side bar, change if the length of the menu items become too
// big
const drawerWidth = 240;

function AppLayout() {
  // State to manage the open/close toggle for mobile drawer
  const [mobileOpen, setMobileOpen] = useState(false);

  // React Router hooks to get the current location and navigate to a new page.
  const navigate = useNavigate();
  const location = useLocation();

  // Toggle the state of the mobile drawer (open/close)
  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  // Menu item definitions for the side drawer
  // Each item includes text, an icon, and a corresponding route path
  const menuItems = [
    { text: 'Chat', icon: <ChatIcon />, path: '/' },
    { text: 'Test Chat', icon: <SpeedIcon />, path: '/testing' },
    { text: 'Google Drive Indexing', icon: <InsertDriveFileIcon />, path: '/gdrive-indexing' },
    { text: 'Shopify Indexing', icon: <ShoppingCartIcon />, path: '/shopify-indexing' },
  ];

  // Drawer component containing the list of menu items
  const drawer = (
    <div>
        {/*
          ** App title displayed above the list Items
          */}
      <Toolbar>
        <Typography variant="h6" noWrap component="div">
          MSquared Marketing
        </Typography>
      </Toolbar>
      <Divider />
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton 
              selected={location.pathname === item.path}
              onClick={() => {
                navigate(item.path);
                setMobileOpen(false);
              }}
            >
              <ListItemIcon>
                {item.icon}
              </ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </div>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div">
            {menuItems.find(item => item.path === location.pathname)?.text || 'MSquared Chat'}
          </Typography>
        </Toolbar>
      </AppBar>
      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile.
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      <Box
        component="main"
        sx={{ 
          flexGrow: 1, 
          p: 3, 
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          height: '100vh',
          display: 'flex',
          flexDirection: 'column'
        }}
      >
        <Toolbar />
        <Container maxWidth="xl" sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <Outlet />
        </Container>
      </Box>
    </Box>
  );
}

export default AppLayout;