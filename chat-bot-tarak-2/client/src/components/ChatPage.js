import './ChatPage.css';
// client/src/components/ChatPage.js
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import { v4 as uuidv4 } from 'uuid';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { jsPDF } from 'jspdf';

// --- Services & Configuration ---
import { refinePrompt, sendMessage, sendAgenticMessage, saveChatHistory, getUserFiles, getUserSettings, streamMessage } from '../services/api';
import { LLM_OPTIONS } from '../config/constants';
import { useTheme } from '../context/ThemeContext';

// --- Child Components & Utilities ---
import SystemPromptWidget, { getPromptTextById } from './SystemPromptWidget';
import HistoryModal from './HistoryModal';
import FileUploadWidget from './FileUploadWidget';
import FileManagerWidget from './FileManagerWidget';
import AnalysisResultModal from './AnalysisResultModal';
import VoiceInputButton from './VoiceInputButton';
import QuizView from './QuizView';
import QuizModal from './QuizModal';
import FlashcardSuggestion from './FlashcardSuggestion';
import FlashcardList from './FlashcardList';
import { shouldSuggestFlashcards } from '../utils/flashcardDetection';
import { generateFlashcards } from '../api/generateFlashcards';
import { parseFlashcards } from '../utils/parseFlashcards';
import CompilerView from './CompilerView';

// --- Icons ---
import { FiFileText, FiMessageSquare, FiDatabase, FiSettings, FiLogOut, FiSun, FiMoon, FiSend, FiPlus, FiArchive, FiShield, FiDownload, FiHelpCircle, FiCode, FiStar, FiX, FiMic } from 'react-icons/fi';
import { PiMagicWand } from 'react-icons/pi';
import { BsMicFill } from 'react-icons/bs'; // Add this import for a more attractive podcast icon

// ===================================================================================
//  UI Sub-Components
// ===================================================================================

const ActivityBar = ({ activeView, setActiveView, onQuizClick, onCompilerClick, onPodcastClick }) => (
    <div className="activity-bar">
        <button className={`activity-button ${activeView === 'ASSISTANT' ? 'active' : ''}`} onClick={() => setActiveView('ASSISTANT')} title="Assistant Settings">
            <FiSettings size={24} />
        </button>
        <button className={`activity-button ${activeView === 'DATA' ? 'active' : ''}`} onClick={() => setActiveView('DATA')} title="Data Sources">
            <FiDatabase size={24} />
        </button>
        <button className="activity-button podcast-icon-button" onClick={onPodcastClick} title="Podcast Generator">
            <BsMicFill size={26} />
        </button>
        <button className={`activity-button ${activeView === 'QUIZ' ? 'active' : ''}`} onClick={() => { setActiveView('QUIZ'); if (onQuizClick) onQuizClick(); }} title="Interactive Quizzing">
            <FiHelpCircle size={24} />
        </button>
        <button className="activity-button" onClick={onCompilerClick} title="Compiler">
            <FiCode size={24} />
        </button>
    </div>
);

const AssistantSettingsPanel = (props) => (
    <div className="sidebar-panel">
        <h3 className="sidebar-header">Assistant Settings</h3>
        <SystemPromptWidget
            selectedPromptId={props.currentSystemPromptId}
            promptText={props.editableSystemPromptText}
            onSelectChange={props.handlePromptSelectChange}
            onTextChange={props.handlePromptTextChange}
        />
        <div className="llm-settings-widget">
            <h4>AI Settings</h4>
            <div className="setting-item">
                <label htmlFor="llm-provider-select">Provider:</label>
                <select id="llm-provider-select" value={props.llmProvider} onChange={props.handleLlmProviderChange} disabled={props.isProcessing}>
                    {Object.keys(LLM_OPTIONS).map(key => (
                        <option key={key} value={key}>{LLM_OPTIONS[key].name}</option>
                    ))}
                </select>
            </div>
            {LLM_OPTIONS[props.llmProvider]?.models.length > 0 && (
                <div className="setting-item">
                    <label htmlFor="llm-model-select">Model:</label>
                    <select id="llm-model-select" value={props.llmModelName} onChange={props.handleLlmModelChange} disabled={props.isProcessing}>
                        {LLM_OPTIONS[props.llmProvider].models.map(model => <option key={model} value={model}>{model}</option>)}
                        <option value="">Provider Default</option>
                    </select>
                </div>
            )}
            <div className="setting-item rag-toggle-container" title="Enable Multi-Query for RAG">
                <label htmlFor="multi-query-toggle">Multi-Query (RAG)</label>
                <input type="checkbox" id="multi-query-toggle" checked={props.enableMultiQuery} onChange={props.handleMultiQueryToggle} disabled={props.isProcessing || !props.isRagEnabled} />
            </div>
        </div>
    </div>
);

const DataSourcePanel = (props) => (
    <div className="sidebar-panel">
        <h3 className="sidebar-header">Data Sources</h3>
        <FileUploadWidget onUploadSuccess={props.triggerFileRefresh} />
        <FileManagerWidget refreshTrigger={props.refreshTrigger} onAnalysisComplete={props.onAnalysisComplete} setHasFiles={props.setHasFiles} onFileSelect={props.onFileSelect} activeFile={props.activeFile}/>
    </div>
);


const ThemeToggleButton = () => {
    const { theme, toggleTheme } = useTheme();
    return (
        <button onClick={toggleTheme} className="header-button theme-toggle-button" title={`Switch to ${theme === 'light' ? 'Dark' : 'Light'} Mode`}>
            {theme === 'light' ? <FiMoon size={20} /> : <FiSun size={20} />}
        </button>
    );
};


// ===================================================================================
//  Main ChatPage Component
// ===================================================================================

// MODIFICATION: Component signature updated
const ChatPage = ({ performLogout, initialPanel }) => {
    // --- State Management ---
    // MODIFICATION: activeView state initialization updated
    const [activeView, setActiveView] = useState(initialPanel === 'files' ? 'DATA' : 'ASSISTANT');
    const [isInitialLoad, setIsInitialLoad] = useState(true);
    const [showGradient, setShowGradient] = useState(true);
    
    useEffect(() => {
    // This effect runs whenever the initialPanel prop changes from MainLayout.
    // It ensures the panel displayed inside ChatPage always matches the
    // main navigation selection.
    setActiveView(initialPanel === 'files' ? 'DATA' : 'ASSISTANT');
    }, [initialPanel]);

    // Handle initial load animation
    useEffect(() => {
        const timer = setTimeout(() => {
            setShowGradient(false);
            setIsInitialLoad(false);
            console.log('Gradient animation completed, transitioning to normal shadow');
        }, 2000); // Show gradient for 2 seconds

        return () => clearTimeout(timer);
    }, []);
    useEffect(() => {
        const timer = setTimeout(() => {
            setShowGradient(false);
            setIsInitialLoad(false);
        }, 2000); // Show gradient for 2 seconds

        return () => clearTimeout(timer);
    }, []);

    const [messages, setMessages] = useState([]);
    const [inputText, setInputText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [sessionId, setSessionId] = useState('');
    const [username, setUsername] = useState('');
    const [userRole, setUserRole] = useState(null);
    const [currentSystemPromptId, setCurrentSystemPromptId] = useState('friendly');
    const [editableSystemPromptText, setEditableSystemPromptText] = useState(() => getPromptTextById('friendly'));
    const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);
    const [fileRefreshTrigger, setFileRefreshTrigger] = useState(0);
    const [hasFiles, setHasFiles] = useState(false);
    const [isRagEnabled, setIsRagEnabled] = useState(false);
    const [llmProvider, setLlmProvider] = useState('gemini');
    const [llmModelName, setLlmModelName] = useState(LLM_OPTIONS['gemini']?.models[0] || '');
    const [enableMultiQuery, setEnableMultiQuery] = useState(true);
    const [isAnalysisModalOpen, setIsAnalysisModalOpen] = useState(false);
    const [analysisModalData, setAnalysisModalData] = useState(null);
    const [activeFile, setActiveFile] = useState(localStorage.getItem('activeFile') || null);
    const [isQuizModalOpen, setIsQuizModalOpen] = useState(false);
    const [flashcards, setFlashcards] = useState([]);
    const [showFlashcardSuggestion, setShowFlashcardSuggestion] = useState(false);
    const [alreadySuggested, setAlreadySuggested] = useState({});
    const [isCompilerPopupOpen, setIsCompilerPopupOpen] = useState(false);
    const [isAgentMode, setIsAgentMode] = useState(false);
    const [isRefining, setIsRefining] = useState(false);

    // --- Refs & Hooks ---
    const messagesEndRef = useRef(null);
    const navigate = useNavigate();
    const {
        transcript,
        listening,
        resetTranscript,
        browserSupportsSpeechRecognition
    } = useSpeechRecognition();

    useEffect(() => {
        setInputText(transcript);
    }, [transcript]);

    // MODIFICATION: performLogoutCleanup hook removed.

    useEffect(() => {
        const initializeApp = async () => {
            try {
                const storedSessionId = localStorage.getItem('sessionId') || uuidv4();
                if (!localStorage.getItem('sessionId')) localStorage.setItem('sessionId', storedSessionId);
                setSessionId(storedSessionId);

                const userRole = localStorage.getItem('userRole');
                const username = localStorage.getItem('username');

                if (!userRole || !username) {
                    performLogout();
                    return;
                }
                setUserRole(userRole);
                setUsername(username);

                const storedKeys = localStorage.getItem('userApiKeys');
                if (!storedKeys || storedKeys === '{}') {
                    const settingsResponse = await getUserSettings();
                    const settings = settingsResponse.data;
                    if (settings && (settings.geminiApiKey || settings.grokApiKey)) {
                        const keysToStore = { gemini: settings.geminiApiKey, groq: settings.grokApiKey, ollama_host: settings.ollamaHost };
                        localStorage.setItem('userApiKeys', JSON.stringify(keysToStore));
                    }
                }
            } catch (error) {
                console.error("Error during app initialization:", error);
                setError("Could not validate user settings.");
            }
        };
        initializeApp();
    }, [performLogout]);

    const handlePromptSelectChange = useCallback((newId) => {
        setCurrentSystemPromptId(newId);
        setEditableSystemPromptText(getPromptTextById(newId));
    }, []);
    
    const saveAndReset = useCallback(async (isLoggingOut = false, onCompleteCallback = null) => {
        const messagesToSave = messages.filter(m => m.role && m.parts);
        if (messagesToSave.length > 0) {
            setIsLoading(true);
            setError('');
            try {
                await saveChatHistory({ sessionId: localStorage.getItem('sessionId'), messages: messagesToSave });
            } catch (err)
                {
                setError(`Session Error: ${err.response?.data?.message || 'Failed to save session.'}`);
            }
        }
        
        const newSessionId = uuidv4();
        localStorage.setItem('sessionId', newSessionId);
        setSessionId(newSessionId);
        setMessages([]);
        if (!isLoggingOut) handlePromptSelectChange('friendly');
        setIsLoading(false);
        if (onCompleteCallback) onCompleteCallback();
    }, [messages, handlePromptSelectChange]);
    
    // MODIFICATION: handleLogout hook removed.

    const handleSendMessage = useCallback(async (e) => {
        if (e) e.preventDefault();
        const textToSend = inputText.trim();
        if (!textToSend || isLoading) return;
    
        if (listening) SpeechRecognition.stopListening();
        setIsLoading(true);
        setError('');
        
        const newUserMessage = { role: 'user', parts: [{ text: textToSend }], id: uuidv4(), timestamp: new Date().toISOString() };
        setMessages(prev => [...prev, newUserMessage]);
        setInputText('');
        resetTranscript();
    
        const isRAGOrAgent = isRagEnabled || isAgentMode;
    
        try {
            if (isRAGOrAgent) {
                console.log(`Executing in non-streaming mode: RAG=${isRagEnabled}, Agent=${isAgentMode}`);
                let response;
                if (isAgentMode) {
                    response = await sendAgenticMessage(textToSend);
                } else { 
                    const messageData = {
                        message: textToSend,
                        history: [...messages, newUserMessage],
                        sessionId: localStorage.getItem('sessionId'),
                        systemPrompt: editableSystemPromptText,
                        isRagEnabled,
                        llmProvider,
                        llmModelName,
                        enableMultiQuery,
                        activeFile,
                    };
                    response = await sendMessage(messageData);
                }
    
                if (!response.data) throw new Error("Invalid response from server.");
    
                const replyData = response.data.reply || {
                    role: 'model',
                    parts: [{ text: response.data.agent_response || "Agent task completed." }],
                    agentTrace: response.data.agent_trace,
                    references: response.data.references || [],
                    thinking: response.data.thinking_content || null,
                    provider: response.data.provider_used,
                    model: response.data.model_used,
                    context_source: response.data.context_source
                };
                const aiReply = { ...replyData, id: uuidv4(), timestamp: new Date().toISOString() };
                setMessages(prev => [...prev, aiReply]);
    
            } else {
                console.log("Executing in streaming mode.");
                const aiMessageId = uuidv4();
                const emptyAiMessage = { 
                    role: 'model', 
                    parts: [{ text: '' }], 
                    id: aiMessageId, 
                    timestamp: new Date().toISOString(),
                    provider: llmProvider,
                    model: llmModelName,
                    context_source: 'Conversational'
                };
                setMessages(prev => [...prev, emptyAiMessage]);
    
                const messageData = {
                    message: textToSend,
                    history: [...messages, newUserMessage],
                    sessionId: localStorage.getItem('sessionId'),
                    systemPrompt: editableSystemPromptText,
                    llmProvider,
                    llmModelName,
                };
    
                await streamMessage(messageData, (chunk) => {
                    setMessages(prev =>
                        prev.map(msg =>
                            msg.id === aiMessageId
                                ? { ...msg, parts: [{ text: msg.parts[0].text + chunk }] }
                                : msg
                        )
                    );
                });
            }
    
        } catch (err) {
            const errorMessage = err.response?.data?.message || err.message || 'Failed to get response.';
            setError(`Chat Error: ${errorMessage}`);
            setMessages(prev => [...prev, { role: 'model', parts: [{ text: `Error: ${errorMessage}` }], isError: true, id: uuidv4(), timestamp: new Date().toISOString() }]);
        } finally {
            setIsLoading(false);
        }
    }, [
        inputText, isLoading, listening, messages, resetTranscript,
        isRagEnabled, isAgentMode, editableSystemPromptText, llmProvider, 
        llmModelName, enableMultiQuery, activeFile, sessionId
    ]);

    const handleAcceptFlashcards = useCallback(async () => {
        setShowFlashcardSuggestion(false);
        const lastMessage = messages.slice().reverse().find(m => m.role === 'model' && !m.isError);
        if (!lastMessage || !lastMessage.parts[0]?.text) return;

        setIsLoading(true);
        setError('');
        try {
            const response = await generateFlashcards(lastMessage.parts[0].text);
            const parsed = parseFlashcards(response.data.flashcards);
            if (parsed.length > 0) {
                setFlashcards(prev => [...prev, ...parsed]);
                setMessages(prev => [...prev, {
                    role: 'model',
                    parts: [{ text: `I've created ${parsed.length} flashcards for you. You can view them in the Quiz tab.` }],
                    isFlashcardNotification: true,
                    timestamp: new Date().toISOString()
                }]);
            }
        } catch (err) {
            setError("Sorry, I couldn't generate flashcards from that text.");
        } finally {
            setIsLoading(false);
        }
    }, [messages]);

    const handleRefinePrompt = useCallback(async () => {
        const textToRefine = inputText.trim();
        if (!textToRefine || isRefining || isLoading) return;
        setIsRefining(true);
        setError('');
        try {
            const response = await refinePrompt(textToRefine);
            if (response.data && response.data.refined_prompt) {
                setInputText(response.data.refined_prompt);
            } else {
                throw new Error("The AI failed to refine the prompt.");
            }
        } catch (err) {
            setError(`Refine Error: ${err.response?.data?.message || 'Failed to refine prompt.'}`);
        } finally {
            setIsRefining(false);
        }
    }, [inputText, isRefining, isLoading]);

    const triggerFileRefresh = useCallback(() => {
        setFileRefreshTrigger(p => p + 1);
        setIsRagEnabled(true);
        setHasFiles(true);
        getUserFiles().then(response => {
            const files = response.data || [];
            if (files.length > 0) {
                const latestFile = files.reduce((a, b) => (new Date(a.lastModified) > new Date(b.lastModified) ? a : b));
                setActiveFile(latestFile.relativePath);
                localStorage.setItem('activeFile', latestFile.relativePath);
            }
        });
    }, []);

    const handleNewChat = useCallback(() => { 
        if (!isLoading) { 
            resetTranscript(); 
            saveAndReset(false); 
        } 
    }, [isLoading, saveAndReset, resetTranscript]);

    const handleEnterKey = useCallback((e) => { if (e.key === 'Enter' && !e.shiftKey && !isLoading) { e.preventDefault(); handleSendMessage(e); } }, [handleSendMessage, isLoading]);
    const handlePromptTextChange = useCallback((newText) => { setEditableSystemPromptText(newText); }, []);
    const handleLlmProviderChange = (e) => { const newProvider = e.target.value; setLlmProvider(newProvider); setLlmModelName(LLM_OPTIONS[newProvider]?.models[0] || ''); };
    const handleLlmModelChange = (e) => { setLlmModelName(e.target.value); };
    const handleRagToggle = (e) => setIsRagEnabled(e.target.checked);
    const handleMultiQueryToggle = (e) => setEnableMultiQuery(e.target.checked);
    const handleHistory = useCallback(() => setIsHistoryModalOpen(true), []);
    const closeHistoryModal = useCallback(() => setIsHistoryModalOpen(false), []);
    const handleSessionSelectForContinuation = useCallback((sessionData) => {
        if (sessionData && sessionData.sessionId && sessionData.messages) {
            localStorage.setItem('sessionId', sessionData.sessionId);
            setSessionId(sessionData.sessionId);
            setMessages(sessionData.messages);
            setError('');
            closeHistoryModal();
        }
    }, [closeHistoryModal]);

    const onAnalysisComplete = useCallback((data) => { setAnalysisModalData(data); setIsAnalysisModalOpen(true); }, []);
    const closeAnalysisModal = useCallback(() => { setAnalysisModalData(null); setIsAnalysisModalOpen(false); }, []);
    
    const handleToggleListen = () => {
        if (listening) {
            SpeechRecognition.stopListening();
        } else {
            resetTranscript();
            SpeechRecognition.startListening({ continuous: true });
        }
    };

    const handleDownloadChat = useCallback(() => {
        if (messages.length === 0) return;
        const doc = new jsPDF();
        let y = 10;
        doc.setFontSize(12);
        messages.forEach((msg) => {
            const sender = msg.role === 'user' ? username || 'User' : 'Assistant';
            const text = msg.parts.map(part => part.text).join(' ');
            const lines = doc.splitTextToSize(`${sender}: ${text}`, 180);
            if (y + (lines.length * 10) > 280) {
                doc.addPage();
                y = 10;
            }
            doc.text(lines, 10, y);
            y += lines.length * 10;
        });
        doc.save('chat_history.pdf');
    }, [messages, username]);

    const handleFileSelect = useCallback((filePath) => {
        setActiveFile(filePath);
        localStorage.setItem('activeFile', filePath);
    }, []);

    const sidebarProps = {
        currentSystemPromptId, editableSystemPromptText,
        handlePromptSelectChange, handlePromptTextChange,
        llmProvider, handleLlmProviderChange,
        isProcessing: isLoading, llmModelName, handleLlmModelChange,
        enableMultiQuery, handleMultiQueryToggle, isRagEnabled,
        triggerFileRefresh, refreshTrigger: fileRefreshTrigger, onAnalysisComplete,
        setHasFiles, onFileSelect: handleFileSelect, activeFile
    };

    return (
        <div className={`chat-page-container ${showGradient ? 'gradient-loading' : 'normal-shadow'}`}>
            <div className="chat-page-sidebar">
                {activeView === 'ASSISTANT' && <AssistantSettingsPanel {...sidebarProps} />}
                {activeView === 'DATA' && <DataSourcePanel {...sidebarProps} />}
            </div>
            <div className="chat-view">
                <header className="chat-header">
                  <div className="header-left">
                    <h1 className="header-title">FusedChat</h1>
                    {showGradient && <span className="loading-indicator" style={{marginLeft: '10px', fontSize: '0.8rem', opacity: 0.7}}>Loading...</span>}
                  </div>
                  <div className="header-right">
                    <span className="username-display">Hi, {username}</span>
                    <ThemeToggleButton />
                    <button onClick={handleHistory} className="header-button" title="Chat History" disabled={isLoading}><FiArchive size={20} /></button>
                    <button onClick={() => navigate('/settings')} className="header-button" title="Settings" disabled={isLoading}><FiSettings size={20} /></button>
                    <button onClick={handleDownloadChat} className="header-button" title="Download Chat" disabled={messages.length === 0}><FiDownload size={20} /></button>
                    <button onClick={handleNewChat} className="header-button" title="New Chat" disabled={isLoading}><FiPlus size={20} /></button>
                    {userRole === 'admin' && (
                      <button onClick={() => navigate('/admin')} className="header-button" title="Admin Panel">
                        <FiShield size={20} />
                      </button>
                    )}
                    {/* MODIFICATION: Logout button uses performLogout prop */}
                    <button onClick={performLogout} className="header-button" title="Logout" disabled={isLoading}><FiLogOut size={20} /></button>
                  </div>
                </header>
                <main className="messages-area" ref={messagesEndRef}>
                    {messages.length === 0 && !isLoading && (
                         <div className="welcome-screen">
                            <FiMessageSquare size={48} className="welcome-icon" />
                            <h2>Start a conversation</h2>
                            <p>Ask a question, upload a document, or select a model to begin.</p>
                         </div>
                    )}
                    {messages.map((msg) => (
                        <div key={msg.id} className={`message ${msg.role.toLowerCase()}${msg.isError ? '-error-message' : ''}`}>
                            <div className="message-content-wrapper">
                                <p className="message-sender-name">{msg.role === 'user' ? username : 'Assistant'}</p>
                                <div className="message-text"><ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.parts[0].text}</ReactMarkdown></div>
                                
                                {msg.thinking && <details className="message-thinking-trace"><summary>Thinking Process</summary><pre>{msg.thinking}</pre></details>}
                                
                                {msg.agentTrace && (
                                    <details className="message-thinking-trace agent-trace">
                                        <summary>Agent Steps</summary>
                                        <pre>{msg.agentTrace}</pre>
                                    </details>
                                )}

                                {msg.role === 'model' && msg.provider && (
                                    <div className="message-metadata">
                                        <span>Provider: {msg.provider} | Model: {msg.model || 'Default'}</span>
                                    </div>
                                )}
                                {msg.references?.length > 0 && <div className="message-references"><strong>References:</strong><ul>{msg.references.map((ref, i) => <li key={i} title={ref.preview_snippet}>{ref.documentName} (Score: {ref.score?.toFixed(2)})</li>)}</ul></div>}
                            </div>
                        </div>
                    ))}
                </main>
                
                {showFlashcardSuggestion && (
                    <FlashcardSuggestion onAccept={handleAcceptFlashcards} onDecline={() => setShowFlashcardSuggestion(false)} />
                )}
                {flashcards.length > 0 && (
                    <FlashcardList cards={flashcards} />
                )}

                <div className="indicator-container">
                    {isLoading && <div className="loading-indicator"><span>Thinking...</span></div>}
                    {!isLoading && error && <div className="error-indicator">{error}</div>}
                </div>
                {inputText.match(/pdf|topics|headings|subheadings/i) && !activeFile && (
                    <div className="fm-error" style={{margin:'10px',textAlign:'center'}}>Please activate a file in the file manager to ask questions about a PDF.</div>
                )}
                <div className="chat-input-area">
                    <textarea value={inputText} onChange={(e) => setInputText(e.target.value)} onKeyDown={handleEnterKey} placeholder="Type or say something..." rows="1" disabled={isLoading || isRefining} />
                    <VoiceInputButton isListening={listening} onToggleListen={handleToggleListen} isSupported={browserSupportsSpeechRecognition} />
                    
                    <button onClick={handleRefinePrompt} disabled={isLoading || isRefining || !inputText.trim()} title="Refine Prompt" className={`prompt-refine-button magic-wand-button${inputText ? ' active' : ''}`}> 
                        {isRefining ? <div className="spinner" /> : <PiMagicWand size={22} />}
                    </button>

                    <div className="agent-toggle-container" title="Toggle Agent Mode for complex, multi-step tasks">
                        <label htmlFor="agent-toggle">Agent</label>
                        <input
                            type="checkbox"
                            id="agent-toggle"
                            checked={isAgentMode}
                            onChange={(e) => setIsAgentMode(e.target.checked)}
                            disabled={isLoading}
                        />
                    </div>

                    <div className="rag-toggle-container" title={!hasFiles ? "Upload files to enable RAG" : "Toggle RAG"}>
                        <label htmlFor="rag-toggle">RAG</label>
                        <input type="checkbox" id="rag-toggle" checked={isRagEnabled} onChange={handleRagToggle} disabled={!hasFiles || isLoading} />
                    </div>
                    <button onClick={handleSendMessage} disabled={isLoading || isRefining || !inputText.trim()} title="Send Message" className="send-button">
                        <FiSend size={20} />
                    </button>
                </div>
            </div>
            {isCompilerPopupOpen && (
                <div className="compiler-popup-overlay" onClick={() => setIsCompilerPopupOpen(false)}>
                    <div className="compiler-popup-container" onClick={(e) => e.stopPropagation()}>
                        <button className="compiler-popup-close-button" onClick={() => setIsCompilerPopupOpen(false)} title="Close"><FiX size={28} /></button>
                        <CompilerView />
                    </div>
                </div>
            )}
            <HistoryModal isOpen={isHistoryModalOpen} onClose={closeHistoryModal} onSessionSelect={handleSessionSelectForContinuation} />
            {analysisModalData && <AnalysisResultModal isOpen={isAnalysisModalOpen} onClose={closeAnalysisModal} analysisData={analysisModalData} />}
            <QuizModal isOpen={isQuizModalOpen} onClose={() => setIsQuizModalOpen(false)}>
                <QuizView flashcards={flashcards} />
            </QuizModal>
        </div>
    );
};

export default ChatPage;