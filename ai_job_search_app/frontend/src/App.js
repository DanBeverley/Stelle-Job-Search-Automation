import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Toaster } from 'react-hot-toast';

import Navbar from './components/layout/Navbar';
import Home from './pages/Home';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import CVUpload from './pages/CVUpload';
import JobSearch from './pages/JobSearch';
import Applications from './pages/Applications';
import SalaryPrediction from './pages/SalaryPrediction';
import SkillAnalysis from './pages/SkillAnalysis';
import CoverLetter from './pages/CoverLetter';
import InterviewPrep from './pages/InterviewPrep';
import Profile from './pages/Profile';

import { AuthProvider } from './contexts/AuthContext';
import ProtectedRoute from './components/auth/ProtectedRoute';

function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="min-h-screen bg-dark-950 text-gray-100">
          <Navbar />
          
          <main className="pt-16">
            <AnimatePresence mode="wait">
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/login" element={<Login />} />
                <Route path="/register" element={<Register />} />
                
                <Route path="/dashboard" element={
                  <ProtectedRoute>
                    <Dashboard />
                  </ProtectedRoute>
                } />
                
                <Route path="/cv-upload" element={
                  <ProtectedRoute>
                    <CVUpload />
                  </ProtectedRoute>
                } />
                
                <Route path="/job-search" element={
                  <ProtectedRoute>
                    <JobSearch />
                  </ProtectedRoute>
                } />
                
                <Route path="/applications" element={
                  <ProtectedRoute>
                    <Applications />
                  </ProtectedRoute>
                } />
                
                <Route path="/salary-prediction" element={
                  <ProtectedRoute>
                    <SalaryPrediction />
                  </ProtectedRoute>
                } />
                
                <Route path="/skill-analysis" element={
                  <ProtectedRoute>
                    <SkillAnalysis />
                  </ProtectedRoute>
                } />
                
                <Route path="/cover-letter" element={
                  <ProtectedRoute>
                    <CoverLetter />
                  </ProtectedRoute>
                } />
                
                <Route path="/interview-prep" element={
                  <ProtectedRoute>
                    <InterviewPrep />
                  </ProtectedRoute>
                } />
                
                <Route path="/profile" element={
                  <ProtectedRoute>
                    <Profile />
                  </ProtectedRoute>
                } />
              </Routes>
            </AnimatePresence>
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