import React, { useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { 
  AppBar, Box, Drawer, Toolbar, Typography, IconButton, 
  List, ListItem, ListItemButton, ListItemIcon, ListItemText, 
  Divider, Container, ListSubheader
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import ChatIcon from '@mui/icons-material/Chat';
import SpeedIcon from '@mui/icons-material/Speed';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import SlideshowIcon from '@mui/icons-material/Slideshow';
import CreateIcon from '@mui/icons-material/Create';
import NetworkCheckIcon from '@mui/icons-material/NetworkCheck';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import AdminPanelSettingsIcon from '@mui/icons-material/AdminPanelSettings';
import config from '../config/config.js';

// Width of the side bar, change if the length of the menu items become too
// big
const drawerWidth = 240;
const collapsedDrawerWidth = 60;

function AppLayout() {
  // State to manage the open/close toggle for mobile drawer
  const [mobileOpen, setMobileOpen] = useState(false);
  // State to manage the permanent drawer expanded/collapsed state
  const [drawerExpanded, setDrawerExpanded] = useState(true);

  // React Router hooks to get the current location and navigate to a new page.
  const navigate = useNavigate();
  const location = useLocation();

  // Toggle the state of the mobile drawer (open/close)
  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  // Toggle the state of the permanent drawer (expanded/collapsed)
  const toggleDrawerExpanded = () => {
    setDrawerExpanded(!drawerExpanded);
  };

  // User-facing menu items (first section)
  const userMenuItems = [
    { text: 'Chat', icon: <ChatIcon />, path: '/' },
    { text: 'Create Documents', icon: <CreateIcon />, path: '/create-document' },
  ];

  // Admin menu items (second section)
  const adminMenuItems = [
    { text: 'Test Chat', icon: <SpeedIcon />, path: '/testing' },
    { text: 'Google Drive Indexing', icon: <InsertDriveFileIcon />, path: '/gdrive-indexing' },
    { text: 'Shopify Indexing', icon: <ShoppingCartIcon />, path: '/shopify-indexing' },
    { text: 'System Diagnostics', icon: <NetworkCheckIcon />, path: '/diagnostics' },
  ];

  // Drawer component containing the list of menu items
  const drawer = (
    <div>
      {/* App title and collapse button */}
      <Toolbar sx={{ justifyContent: 'space-between' }}>
        {drawerExpanded ? (
          <Typography variant="h6" noWrap component="div">
            {config.COMPANY_NAME} Marketing
          </Typography>
        ) : null}
        <IconButton onClick={toggleDrawerExpanded}>
          {drawerExpanded ? <ChevronLeftIcon /> : <ChevronRightIcon />}
        </IconButton>
      </Toolbar>
      <Divider />

      {/* User section */}
      <List
        subheader={
          drawerExpanded ? (
            <ListSubheader component="div" id="user-section-subheader">
              User Tools
            </ListSubheader>
          ) : null
        }
      >
        {userMenuItems.map((item) => (
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
              {drawerExpanded && <ListItemText primary={item.text} />}
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      
      <Divider />
      
      {/* Admin section */}
      <List
        subheader={
          drawerExpanded ? (
            <ListSubheader component="div" id="admin-section-subheader">
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <AdminPanelSettingsIcon fontSize="small" sx={{ mr: 1 }} />
                Admin Tools
              </Box>
            </ListSubheader>
          ) : (
            <ListSubheader component="div" id="admin-section-icon">
              <AdminPanelSettingsIcon />
            </ListSubheader>
          )
        }
      >
        {adminMenuItems.map((item) => (
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
              {drawerExpanded && <ListItemText primary={item.text} />}
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </div>
  );

  // Determine current width based on expanded state
  const currentDrawerWidth = drawerExpanded ? drawerWidth : collapsedDrawerWidth;

  // Find current page title from all menu items
  const allMenuItems = [...userMenuItems, ...adminMenuItems];
  const currentPageTitle = allMenuItems.find(item => item.path === location.pathname)?.text || `${config.COMPANY_NAME} Chat`;

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${currentDrawerWidth}px)` },
          ml: { sm: `${currentDrawerWidth}px` },
          transition: theme => theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
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
            {currentPageTitle}
          </Typography>
        </Toolbar>
      </AppBar>
      <Box
        component="nav"
        sx={{ 
          width: { sm: currentDrawerWidth }, 
          flexShrink: { sm: 0 },
          transition: theme => theme.transitions.create('width', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
          }),
        }}
      >
        {/* Mobile drawer */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile.
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { 
              boxSizing: 'border-box', 
              width: drawerWidth 
            },
          }}
        >
          {drawer}
        </Drawer>
        {/* Desktop drawer */}
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { 
              boxSizing: 'border-box', 
              width: currentDrawerWidth,
              overflowX: 'hidden',
              transition: theme => theme.transitions.create('width', {
                easing: theme.transitions.easing.sharp,
                duration: theme.transitions.duration.enteringScreen,
              }),
            },
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
          width: { sm: `calc(100% - ${currentDrawerWidth}px)` },
          height: '100vh',
          display: 'flex',
          flexDirection: 'column',
          transition: theme => theme.transitions.create('width', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
          }),
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