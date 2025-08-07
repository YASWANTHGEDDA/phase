// client/src/components/MainLayout.js
import React, { useState, Suspense, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom'; // <-- IMPORT hooks
import Sidebar from './Sidebar';
import Modal from './Modal';
import QuizModal from './QuizModal';
import CompilerView from './CompilerView';

// Lazy load ALL page-level components that MainLayout can render
const ChatPage = React.lazy(() => import('./ChatPage'));
const PodcastGeneratorPage = React.lazy(() => import('./PodcastGenerator/PodcastGeneratorPage'));
const SettingsPage = React.lazy(() => import('./SettingsPage')); // <-- LAZY LOAD SETTINGS PAGE
const QuizView = React.lazy(() => import('./QuizView'));

const LoadingFallback = () => <div style={{width: '100%', padding: '20px', textAlign: 'center', color: 'white'}}>Loading...</div>;

const MainLayout = ({ performLogout }) => {
    // --- ROUTING LOGIC ---
    const location = useLocation(); // Gets the current URL path, e.g., "/chat", "/settings"
    const navigate = useNavigate(); // Function to programmatically change the URL

    // This function determines the "active view" based on the current URL path
    const getCurrentView = () => {
        const path = location.pathname;
        if (path.startsWith('/files')) return 'files';
        if (path.startsWith('/podcast')) return 'podcast';
        if (path.startsWith('/settings')) return 'settings';
        // Any other path (like "/" or "/chat") defaults to 'chat'
        return 'chat';
    };
    const currentView = getCurrentView();

    // This function is passed to the Sidebar to handle navigation clicks
    const handleNavigation = (view) => {
        navigate(`/${view}`);
    };

    // --- MODAL LOGIC (Unchanged) ---
    const [isQuizModalOpen, setIsQuizModalOpen] = useState(false);
    const [isCompilerModalOpen, setIsCompilerModalOpen] = useState(false);
    const openQuizModal = useCallback(() => setIsQuizModalOpen(true), []);
    const closeQuizModal = useCallback(() => setIsQuizModalOpen(false), []);
    const openCompilerModal = useCallback(() => setIsCompilerModalOpen(true), []);
    const closeCompilerModal = useCallback(() => setIsCompilerModalOpen(false), []);

    // This function now renders based on the URL-derived 'currentView'
    const renderCurrentView = () => {
        switch(currentView) {
            case 'chat':
            case 'files':
                // ChatPage handles both 'chat' (Assistant) and 'files' (Data Sources) panels
                return <ChatPage performLogout={performLogout} initialPanel={currentView} />;
            case 'podcast':
                return <PodcastGeneratorPage />;
            case 'settings':
                return <SettingsPage />; // <-- RENDER SETTINGS PAGE
            default:
                return <ChatPage performLogout={performLogout} initialPanel="chat" />;
        }
    };

    return (
        <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden' }}>
            <Sidebar 
                currentView={currentView}
                setCurrentView={handleNavigation} // Pass the navigation function
                openCompilerModal={openCompilerModal}
                openQuizModal={openQuizModal}
            />
            <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                <Suspense fallback={<LoadingFallback />}>
                    {renderCurrentView()}
                </Suspense>
            </main>
            <QuizModal isOpen={isQuizModalOpen} onClose={closeQuizModal}>
                <Suspense fallback={<LoadingFallback />}><QuizView /></Suspense>
            </QuizModal>
            <Modal isOpen={isCompilerModalOpen} onClose={closeCompilerModal}>
                <CompilerView />
            </Modal>
        </div>
    );
};

export default MainLayout;