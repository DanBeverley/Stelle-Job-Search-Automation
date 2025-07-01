import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';

import Navbar from './components/layout/Navbar';
import Home from './pages/Home';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import JobSearch from './pages/JobSearch';
import AuthTest from './pages/AuthTest';
import APITest from './pages/APITest';

import { AuthProvider } from './contexts/AuthContext';
import ProtectedRoute from './components/auth/ProtectedRoute';

function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="min-h-screen bg-dark-950 text-gray-100">
          <Navbar />
          
          <main className="pt-16">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />
              <Route path="/auth-test" element={<AuthTest />} />
              <Route path="/api-test" element={<APITest />} />
              
              <Route path="/dashboard" element={
                <ProtectedRoute>
                  <Dashboard />
                </ProtectedRoute>
              } />
              
              <Route path="/job-search" element={
                <ProtectedRoute>
                  <JobSearch />
                </ProtectedRoute>
              } />
            </Routes>
          </main>

          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#0f172a',
                color: '#f1f5f9',
                border: '1px solid #334155',
              },
              success: {
                iconTheme: {
                  primary: '#10b981',
                  secondary: '#f1f5f9',
                },
              },
              error: {
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#f1f5f9',
                },
              },
            }}
          />
        </div>
      </Router>
    </AuthProvider>
  );
}

export default App;