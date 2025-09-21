import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

// Configure API base URL from environment variables
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Check API status on component mount
    checkApiStatus();
    // Load conversations
    loadConversations();
  }, []);

  useEffect(() => {
    // Load messages when conversation changes
    if (currentConversationId) {
      loadConversationMessages(currentConversationId);
    } else {
      // Show welcome message for new conversation
      setMessages([{
        text: "Hello! I'm an AI chatbot powered by DialoGPT. How can I help you today?",
        sender: 'bot',
        timestamp: new Date()
      }]);
    }
  }, [currentConversationId]);

  const checkApiStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`);
      setApiStatus(response.data);
    } catch (error) {
      console.error('Failed to check API status:', error);
      setApiStatus({ status: 'error', message: 'API not available' });
    }
  };

  const loadConversations = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/conversations`);
      setConversations(response.data.conversations);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  };

  const loadConversationMessages = async (conversationId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/conversations/${conversationId}/messages`);
      const formattedMessages = response.data.messages.map(msg => ({
        text: msg.content,
        sender: msg.role === 'user' ? 'user' : 'bot',
        timestamp: new Date(msg.created_at),
        processingTime: msg.processing_time
      }));
      setMessages(formattedMessages);
    } catch (error) {
      console.error('Failed to load conversation messages:', error);
    }
  };

  const createNewConversation = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/conversations`, {
        title: "New Conversation"
      });
      const newConversation = response.data;
      setConversations(prev => [newConversation, ...prev]);
      setCurrentConversationId(newConversation.id);
      setMessages([{
        text: "Hello! I'm an AI chatbot powered by DialoGPT. How can I help you today?",
        sender: 'bot',
        timestamp: new Date()
      }]);
    } catch (error) {
      console.error('Failed to create new conversation:', error);
    }
  };

  const selectConversation = (conversationId) => {
    setCurrentConversationId(conversationId);
  };

  const deleteConversation = async (conversationId) => {
    try {
      await axios.delete(`${API_BASE_URL}/conversations/${conversationId}`);
      setConversations(prev => prev.filter(conv => conv.id !== conversationId));
      if (currentConversationId === conversationId) {
        setCurrentConversationId(null);
        setMessages([]);
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error);
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const userMessage = { text: inputMessage, sender: 'user', timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        message: inputMessage,
        conversation_id: currentConversationId
      });

      const botMessage = {
        text: response.data.response,
        sender: 'bot',
        timestamp: new Date(),
        processingTime: response.data.processing_time
      };

      setMessages(prev => [...prev, botMessage]);

      // Update current conversation ID if it was created
      if (!currentConversationId && response.data.conversation_id) {
        setCurrentConversationId(response.data.conversation_id);
        // Reload conversations to include the new one
        loadConversations();
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'bot',
        timestamp: new Date(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-left">
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            ☰
          </button>
          <h1>AI Chatbot</h1>
        </div>
        <div className="api-status">
          {apiStatus && (
            <span className={`status ${apiStatus.status}`}>
              API: {apiStatus.status}
              {apiStatus.gpu_available && ' | GPU Available'}
            </span>
          )}
        </div>
      </header>

      <div className="app-body">
        {/* Sidebar */}
        <aside className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
          <div className="sidebar-header">
            <button className="new-conversation-btn" onClick={createNewConversation}>
              + New Conversation
            </button>
          </div>
          <div className="conversations-list">
            {conversations.map(conversation => (
              <div
                key={conversation.id}
                className={`conversation-item ${currentConversationId === conversation.id ? 'active' : ''}`}
                onClick={() => selectConversation(conversation.id)}
              >
                <div className="conversation-title">{conversation.title}</div>
                <div className="conversation-meta">
                  {conversation.message_count} messages
                  {conversation.last_message_at && (
                    <span className="last-message-time">
                      {new Date(conversation.last_message_at).toLocaleDateString()}
                    </span>
                  )}
                </div>
                <button
                  className="delete-conversation-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteConversation(conversation.id);
                  }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </aside>

        {/* Main chat area */}
        <main className="chat-container">
        <div className="messages">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.sender}`}>
              <div className="message-content">
                <p>{message.text}</p>
                <small className="timestamp">
                  {message.timestamp.toLocaleTimeString()}
                  {message.processingTime && ` (${(message.processingTime * 1000).toFixed(0)}ms)`}
                </small>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="message bot">
              <div className="message-content">
                <p className="typing">Thinking...</p>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={sendMessage} className="message-form">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Type your message..."
            disabled={isLoading}
            className="message-input"
          />
          <button type="submit" disabled={isLoading || !inputMessage.trim()}>
            Send
          </button>
        </form>
        </main>
      </div>
    </div>
  );
}

export default App;
