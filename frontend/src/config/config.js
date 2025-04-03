/**
 * Frontend configuration file for environment variables
 */

const config = {
    // Company configuration
    COMPANY_NAME: process.env.REACT_APP_COMPANY_NAME || "SVV",

    // API configuration
    API_BASE_URL: process.env.REACT_APP_API_BASE_URL || "http://localhost:8005",

    // Feature flags
    ENABLE_DEBUG_MODE: process.env.REACT_APP_ENABLE_DEBUG_MODE === "true" || false,
};

export default config;
