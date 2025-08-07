// client/src/services/api.js
import axios from 'axios';

// Dynamically determine the base URL for the main backend API
const getApiBaseUrl = () => {
    // The port your main Python/Flask server is running on
    const backendPort = process.env.REACT_APP_BACKEND_PORT || 5003; 
    const hostname = window.location.hostname;
    // Use the same protocol (http or https) as the frontend
    const protocol = window.location.protocol; 
    const backendHost = (hostname === 'localhost' || hostname === '127.0.0.1') ? 'localhost' : hostname;
    return `${protocol}//${backendHost}:${backendPort}/api`;
};

// Create a single, unified Axios instance for all API communication
const apiClient = axios.create({
    baseURL: getApiBaseUrl(),
});

// --- Interceptors ---

// Interceptor 1: Attach the user ID and auth token to every outgoing request
apiClient.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('token');
        const userId = localStorage.getItem('userId');

        if (token) {
            config.headers['Authorization'] = `Bearer ${token}`;
        }
        
        if (userId) {
            config.headers['x-user-id'] = userId;
        } else if (!config.url.includes('/auth/')) {
            // Log a warning if a non-authentication request is made without a user ID
            console.warn("API Interceptor: userId not found for non-auth request to", config.url);
        }
        
        // Ensure Content-Type is set to JSON, but not for file uploads (which use FormData)
        if (!(config.data instanceof FormData)) {
            config.headers['Content-Type'] = 'application/json';
        }
        return config;
    },
    (error) => Promise.reject(error)
);

// Interceptor 2: Handle 401 Unauthorized errors by logging the user out
apiClient.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response && error.response.status === 401) {
            console.warn("API Interceptor: 401 Unauthorized. Clearing session and redirecting to login.");
            localStorage.clear(); // Clear user ID and any other session data
            // Redirect to the login page if not already there
            if (!window.location.pathname.includes('/login')) {
                window.location.href = '/login?sessionExpired=true';
            }
        }
        return Promise.reject(error);
    }
);


// --- API Function Definitions ---

// --- AUTHENTICATION ---
export const signupUser = (userData) => apiClient.post('/auth/signup', userData);
export const signinUser = (userData) => apiClient.post('/auth/signin', userData);
export const requestAdminKeyAccess = () => apiClient.post('/auth/request-access');

// --- USER SETTINGS ---
export const getUserSettings = () => apiClient.get('/settings');
export const saveUserSettings = (settingsData) => apiClient.post('/settings', settingsData);

// --- ADMIN PANEL ---
export const getAdminAccessRequests = () => apiClient.get('/admin/requests');
export const processAdminRequest = (userId, isApproved) => apiClient.post('/admin/approve', { userId, isApproved });
export const getAcceptedUsers = () => apiClient.get('/admin/accepted');

// --- CHAT & AGENTIC SEARCH ---
const AGENT_TIMEOUT = 300000; // 5 minute timeout for long-running agentic tasks

export const sendMessage = (messageData) => apiClient.post('/chat/message', messageData);

export const sendAgenticMessage = (message) => {
    const payload = { message };
    // This calls the agentic endpoint that performs web searches
    return apiClient.post('/chat/agentic', payload, {
        timeout: AGENT_TIMEOUT
    });
};

export const refinePrompt = (raw_prompt) => {
    return apiClient.post('/chat/refine-prompt', { raw_prompt }, {
        timeout: 30000 // 30 second timeout
    });
};

// --- CHAT HISTORY ---
export const saveChatHistory = (historyData) => apiClient.post('/chat/history', historyData);
export const getChatSessions = () => apiClient.get('/chat/sessions');
export const getSessionDetails = (sessionId) => apiClient.get(`/chat/session/${sessionId}`);
export const deleteChatSession = (sessionId) => apiClient.delete(`/chat/session/${sessionId}`);

// --- FILE UPLOAD & MANAGEMENT ---
export const uploadFile = (formData) => apiClient.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
});
export const getUserFiles = () => apiClient.get('/files');
export const renameUserFile = (serverFilename, newOriginalName) => apiClient.patch(`/files/${serverFilename}`, { newOriginalName });
export const deleteUserFile = (serverFilename) => apiClient.delete(`/files/${serverFilename}`);

// --- DOCUMENT ANALYSIS & CONTENT GENERATION ---
export const analyzeDocument = (analysisData) => apiClient.post('/analysis/document', analysisData);

// This is the definitive function for generating reports (e.g., PDFs)
export const generateReport = (reportData) => {
    console.log("API Service: Calling /generation/report with payload:", reportData);
    // This calls the report generation endpoint and expects a file blob in return
    return apiClient.post('/generation/report', reportData, {
        responseType: 'blob', // Critical for handling file downloads
        timeout: AGENT_TIMEOUT, // Use a long timeout as report generation can be slow
    });
};

// --- QUIZ GENERATION ---
export const generateQuizFromFile = (data) => apiClient.post('/analysis/document', data);

// <<< ADD THIS ENTIRE NEW FUNCTION FOR STREAMING >>>
export const streamMessage = async (messageData, onChunk) => {
    const token = localStorage.getItem('token');
    const userId = localStorage.getItem('userId');
    const apiBaseUrl = getApiBaseUrl(); // Use the helper to get the base URL

    try {
        const response = await fetch(`${apiBaseUrl}/chat/message-stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
                'x-user-id': userId
            },
            body: JSON.stringify(messageData),
        });

        if (!response.body) {
            throw new Error("Response body is missing.");
        }
        
        if (!response.ok) {
            // Handle HTTP errors like 503 or 400
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let done = false;

        while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            if (value) {
                const chunk = decoder.decode(value, { stream: true });
                onChunk(chunk); // Call the provided callback with the new piece of text
            }
        }
    } catch (error) {
        console.error("Streaming API call failed:", error);
        // Provide an error chunk to the callback to display in the UI
        onChunk(`\n\n--- \nSorry, an error occurred while streaming the response: ${error.message}`);
    }
};
// <<< END OF NEW FUNCTION >>>


// --- DEFAULT EXPORT ---
// Export the configured instance for use in other parts of the app
export default apiClient;