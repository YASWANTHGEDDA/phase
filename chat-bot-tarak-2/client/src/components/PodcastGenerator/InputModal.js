// client/src/components/PodcastGenerator/InputModal.js
import React, { useState, useRef } from 'react';
import './InputModal.css';
import { UploadFile, Link, Title, YouTube, Close } from '@mui/icons-material';

const InputModal = ({ isOpen, onClose, onGenerate }) => {
    const [activeTab, setActiveTab] = useState('upload');
    const [rawText, setRawText] = useState('');
    const [youtubeUrl, setYoutubeUrl] = useState('');
    const [selectedFile, setSelectedFile] = useState(null);
    const [podcastTitle, setPodcastTitle] = useState('');
    const fileInputRef = useRef(null);

    if (!isOpen) return null;

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setSelectedFile(file);
            if (!podcastTitle) {
                setPodcastTitle(file.name.replace(/\.[^/.]+$/, ""));
            }
        }
    };

    const handleGenerateClick = () => {
        let inputData;
        let inputType;

        switch(activeTab) {
            case 'upload':
                if (!selectedFile) return;
                inputData = selectedFile;
                inputType = 'file';
                break;
            case 'youtube':
                if (!youtubeUrl) return;
                inputData = youtubeUrl;
                inputType = 'youtube_url';
                break;
            case 'text':
                if (!rawText) return;
                inputData = rawText;
                inputType = 'raw_text';
                break;
            default:
                return;
        }
        onGenerate({ inputType, inputData, title: podcastTitle || 'AI Podcast' });
    };

    const renderContent = () => {
        switch (activeTab) {
            case 'upload':
                return (
                    <div className="tab-content" onClick={() => fileInputRef.current.click()}>
                        <UploadFile style={{ fontSize: 60, color: '#aaa' }} />
                        <p>{selectedFile ? selectedFile.name : 'Click or Drag and Drop to Upload'}</p>
                        <span className="file-types">Supported: .pdf, .txt</span>
                        <input type="file" ref={fileInputRef} onChange={handleFileChange} style={{ display: 'none' }} accept=".pdf,.txt" />
                    </div>
                );
            case 'youtube':
                return (
                    <div className="tab-content">
                        <YouTube style={{ fontSize: 60, color: '#aaa' }} />
                        <input
                            type="text"
                            className="url-input"
                            placeholder="Paste YouTube URL here..."
                            value={youtubeUrl}
                            onChange={(e) => setYoutubeUrl(e.target.value)}
                        />
                    </div>
                );
            case 'text':
                return (
                    <div className="tab-content">
                        <textarea
                            className="text-input-area"
                            placeholder="Paste your text here..."
                            value={rawText}
                            onChange={(e) => setRawText(e.target.value)}
                        />
                    </div>
                );
            default: return null;
        }
    };

    return (
        <div className="modal-backdrop">
            <div className="modal-content">
                <button className="modal-close-btn" onClick={onClose}><Close /></button>
                <h2>Add Source</h2>
                <p className="modal-subtitle">Generate a podcast from your own content.</p>
                
                <div className="modal-body">
                    {renderContent()}
                </div>

                <div className="modal-footer">
                    <div className="title-input-wrapper">
                        <Title style={{ color: '#888' }} />
                        <input 
                            type="text" 
                            placeholder="Enter podcast title..." 
                            className="title-input"
                            value={podcastTitle}
                            onChange={(e) => setPodcastTitle(e.target.value)}
                        />
                    </div>
                    <button className="generate-btn" onClick={handleGenerateClick}>
                        Generate Podcast
                    </button>
                </div>

                <div className="tabs">
                    <button className={`tab ${activeTab === 'upload' ? 'active' : ''}`} onClick={() => setActiveTab('upload')}>
                        <UploadFile /> Upload File
                    </button>
                    <button className={`tab ${activeTab === 'youtube' ? 'active' : ''}`} onClick={() => setActiveTab('youtube')}>
                        <YouTube /> YouTube
                    </button>
                    <button className={`tab ${activeTab === 'text' ? 'active' : ''}`} onClick={() => setActiveTab('text')}>
                        <Link /> Paste Text
                    </button>
                </div>
            </div>
        </div>
    );
};

export default InputModal;