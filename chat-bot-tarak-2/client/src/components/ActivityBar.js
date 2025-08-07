// client/src/components/ActivityBar.js

import React from 'react';
// Import the icons you want to use. 'Mic' is a good choice for a podcast.
import { QuestionAnswer, Code, Mic } from '@mui/icons-material'; // Or your preferred icons
import './ActivityBar.css';

// The props now include onPodcastClick
const ActivityBar = ({ onQuizClick, onCompilerClick, onPodcastClick }) => {
    return (
        <div className="activity-bar">
            {/* Your existing icons */}
            <button onClick={onQuizClick} className="activity-bar-button" title="Start a Quiz">
                <QuestionAnswer />
            </button>
            <button onClick={onCompilerClick} className="activity-bar-button" title="Open Compiler">
                <Code />
            </button>
            
            {/* THIS IS THE NEW ICON BUTTON */}
            <button onClick={onPodcastClick} className="activity-bar-button" title="AI Podcast Generator">
                <Mic />
            </button>
        </div>
    );
};

export default ActivityBar;