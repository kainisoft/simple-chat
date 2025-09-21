# Chatbot Frontend

This is the React frontend for the AI Chatbot API.

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

The frontend will be available at http://localhost:3000

### Backend Integration

The frontend is configured to proxy API requests to the backend running on http://localhost:8000.

Make sure the backend is running before starting the frontend:

```bash
# From the project root
docker compose -f docker-compose.dev.yml up
```

### Available Scripts

- `npm start` - Runs the app in development mode
- `npm test` - Launches the test runner
- `npm run build` - Builds the app for production
- `npm run eject` - Ejects from Create React App (one-way operation)

### Features

- Real-time chat interface
- Message history
- Responsive design
- API integration with the backend chatbot service
